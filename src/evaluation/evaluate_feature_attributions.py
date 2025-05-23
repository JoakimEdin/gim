import pandas as pd
import torch
from rich.progress import track
from transformers import AutoTokenizer
from pathlib import Path

import numpy as np
from transformer_lens import HookedTransformer
from src.feature_attribution_methods.feature_attribution_methods import (
    get_deeplift_callable,
    get_gradient_x_input_callable,
    get_integrated_gradient_callable,
    get_kernelshap_callable,
    get_lime_callable,
    get_occlusion_1_callable,
    get_random_baseline_callable,
    get_attnlrp_callable,
    get_atpstar_callable,
    get_gim_callable,
    get_grad_freeze_norm_callable,
    get_grad_norm_callable,
    get_grad_freeze_callable,
    get_grad_tsg_callable,
    get_grad_baseline_callable,
    get_transformerlrp_callable,
)
from src.models import MODELS
from src.utils.dataset_factory import DATASETS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = Path("results")
EXPLANATION_METHODS = {
    "gradient_x_input": get_gradient_x_input_callable,
    "deeplift": get_deeplift_callable,
    "integrated_gradient": get_integrated_gradient_callable,
    "lime": get_lime_callable,
    "kernelshap": get_kernelshap_callable,
    "occlusion_1": get_occlusion_1_callable,
    "random": get_random_baseline_callable,
    "atpstar": get_atpstar_callable,
    "attnlrp": get_attnlrp_callable,
    "transformerlrp": get_transformerlrp_callable,
    "gim": get_gim_callable,
    "grad_freeze_norm": get_grad_freeze_norm_callable,
    "grad_norm": get_grad_norm_callable,
    "grad_freeze": get_grad_freeze_callable,
    "grad_tsg": get_grad_tsg_callable,
    "grad_baseline": get_grad_baseline_callable,
}

model_names = MODELS.keys()
dataset_names = DATASETS.keys()


explanation_method_names = [
    "gradient_x_input",
    "grad_norm",
    "grad_freeze",
    "grad_tsg",
    "grad_freeze_norm",
    "transformerlrp",
    "attnlrp",
    "gim",
    "integrated_gradient",
    "deeplift",
]

masking_ratio_sufficiency = [0.5, 0.8, 0.9, 0.95, 0.99]
model = None

for model_name in model_names:
    if model is not None:
        del model
        torch.cuda.empty_cache()
    model_path = MODELS[model_name]
    model = HookedTransformer.from_pretrained(
        model_path,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=True,
    )

    model.cfg.use_attn_result = False
    model.cfg.use_attn_in = False
    model.cfg.use_split_qkv_input = False
    model.cfg.use_hook_mlp_in = False

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenization_spaces=False
    )
    baseline_token_id = tokenizer(" ", return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0].to(device)
    start_token_id = tokenizer.cls_token_id
    end_token_id = tokenizer.sep_token_id

    for dataset_name in dataset_names:
        dataset_config = DATASETS[dataset_name]

        true_positives = pd.read_csv(
            results_dir / dataset_name / model_name / "correct_positive_ids.csv",
            dtype={"id": str},
        )
        true_positives_ids = set(true_positives["id"].values)

        dataset = dataset_config.loader()

        dataset = dataset.filter(lambda x: x["annotation_id"] in true_positives_ids)
        print(f"Number of true positives: {len(dataset)}")

        create_prompt = dataset_config.get_model_specific_prompt(model_name)
        options = dataset_config.get_model_specific_options(model_name)

        if "query" in dataset.column_names:
            dataset = dataset.map(
                lambda x: create_prompt(context=x["context"], query=x["query"])
            )
        else:
            dataset = dataset.map(
                lambda x: create_prompt(context=x["context"]),
            )

        # Update the evidence spans to account for the prompt
        dataset = dataset.map(
            lambda x: {
                "evidence_spans_prompt": None
                if x["evidence_spans"] is None
                else [
                    [
                        evidence_span[0] + x["n_characters_before_context"],
                        evidence_span[1] + x["n_characters_before_context"],
                    ]
                    for evidence_span in x["evidence_spans"]
                ]
            },
        )

        dataset = dataset.map(
            lambda x: tokenizer(x["prompt"], return_tensors="pt", truncation=True),
            remove_columns=["context"],
            num_proc=8,
            batch_size=1000,
        )
        dataset = dataset.filter(lambda x: len(x["input_ids"][0]) <= 1024)
        # dataset = dataset.select(range(10))

        # Get token IDs for the options
        options_token_id = [
            token_id
            for token_id in tokenizer(options, add_special_tokens=False)["input_ids"]
        ]

        dataset = dataset.map(
            lambda x: {"target_token": options_token_id[x["label"]]},
        )

        for explanation_name in explanation_method_names:
            explanation_method = EXPLANATION_METHODS[explanation_name]
            explanation_method_callable = explanation_method(
                model,
                baseline_token_id=baseline_token_id,
                cls_token_id=start_token_id,
                eos_token_id=end_token_id,
            )
            print(f"Evaluating {explanation_name} on {dataset_name} dataset")
            sufficiency_list = []
            comprehensiveness_list = []
            explanation_names = []
            id_list = []
            prob_list = []
            attribution_list = []
            for example in track(
                dataset, description="Evaluating explanations", total=len(dataset)
            ):
                input_ids = torch.tensor(example["input_ids"]).to(device)
                answer = torch.tensor(example["target_token"]).to(device)
                token_attributions = (
                    explanation_method_callable(
                        input_ids=input_ids,
                        answer=answer,
                        device=device,
                    )
                    .squeeze()
                    .cpu()
                )
                with torch.no_grad():
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                    ):
                        full_output = (
                            model(input_ids)[:, -1, answer].squeeze().cpu().item()
                        )
                        token_ranking = (
                            torch.argsort(token_attributions[1:-1], descending=True) + 1
                        )
                        permutation_input_ids = input_ids.clone()

                        comprehensiveness = np.zeros(50)
                        for idx, token_idx_to_patch in enumerate(token_ranking[:50]):
                            permutation_input_ids[
                                :, token_idx_to_patch
                            ] = baseline_token_id
                            comprehensiveness[idx] = (
                                full_output
                                - model(permutation_input_ids)[:, -1, answer]
                                .squeeze()
                                .cpu()
                                .item()
                            )

                        # # calculate sufficiency
                        sufficiency = np.zeros(len(masking_ratio_sufficiency))
                        permutation_input_ids = input_ids.clone()
                        token_ranking_flipped = token_ranking.flip(0)
                        for idx, mask_ratio in enumerate(masking_ratio_sufficiency):
                            n_tokens_to_mask = int(
                                mask_ratio * len(token_ranking_flipped)
                            )
                            token_idx_to_patch = token_ranking_flipped[
                                :n_tokens_to_mask
                            ]
                            permutation_input_ids[
                                :, token_idx_to_patch
                            ] = baseline_token_id
                            sufficiency[idx] = (
                                full_output
                                - model(permutation_input_ids)[:, -1, answer]
                                .squeeze()
                                .cpu()
                                .item()
                            )

                        explanation_names.append(explanation_name)
                        comprehensiveness_list.append(comprehensiveness)
                        sufficiency_list.append(sufficiency)
                        prob_list.append(full_output)
                        id_list.append(example["annotation_id"])
                        attribution_list.append(token_attributions.cpu().numpy())

                        print(
                            f"Comprehensiveness: {comprehensiveness.mean() / full_output} - Sufficiency: {sufficiency.mean() / full_output} - Prob: {full_output}"
                        )

            df = pd.DataFrame(
                {
                    "id": id_list,
                    "explanation_method": explanation_names,
                    "comprehensiveness": comprehensiveness_list,
                    "sufficiency": sufficiency_list,
                    "attributions": attribution_list,
                    "prob": prob_list,
                }
            )
            path = results_dir / dataset_name / model_name / explanation_name
            path.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path / "aopc.parquet")
