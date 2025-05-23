# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
import torch
from rich.progress import track
from transformers import AutoTokenizer
from pathlib import Path

import numpy as np
from transformer_lens import HookedTransformer
from fancy_einsum import einsum
import einops

from src.models import MODELS
from src.utils.dataset_factory import DATASETS
from src.gim.transpose_trace import TransformerTrace
from src.gim.config import TransformerTraceConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = Path("results")
IG_STEPS = 50


# Create the configuration
def get_config(explanation_name: str) -> TransformerTraceConfig:
    if explanation_name == "gim":
        return TransformerTraceConfig(
            layernorm_backward="freeze",
            activation_backward="grad",
            query_value_key_divide_grad_strategy=True,
            abs_edge_strength=False,
            value_only=False,
            softmax_backward="tsg",
            log_level=0,
            scale_mlp_gate=True,
        )

    if explanation_name == "grad" or explanation_name == "ig":
        return TransformerTraceConfig(
            layernorm_backward="grad",
            activation_backward="grad",
            query_value_key_divide_grad_strategy=False,
            abs_edge_strength=False,
            value_only=False,
            softmax_backward="grad",
            log_level=0,
            scale_mlp_gate=False,
        )

    if explanation_name == "atpstar":
        return TransformerTraceConfig(
            layernorm_backward="grad",
            activation_backward="grad",
            query_value_key_divide_grad_strategy=False,
            abs_edge_strength=False,
            value_only=False,
            softmax_backward="atpstar",
            log_level=0,
            scale_mlp_gate=False,
        )
    raise ValueError(f"Unknown explanation name: {explanation_name}")


model_names = [
    "gemma2-2b-it",
    "llama3-1b-it",
    "llama3-3b-it",
    "qwen2-1.5b-it",
    "qwen2-3b-it",
]

dataset_names = [
    "fever",
    "twitter",
    "movie",
    "hatexplain",
    "scifact",
    "boolq",
]

masking_ratio_sufficiency = [0.5, 0.8, 0.9, 0.95, 0.99]
model = None
gradient_manager = None

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

    trace = TransformerTrace(model)

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

        # Get token IDs for the options
        options_token_id = [
            token_id
            for token_id in tokenizer(options, add_special_tokens=False)["input_ids"]
        ]

        dataset = dataset.map(
            lambda x: {"target_token": options_token_id[x["label"]]},
        )

        for explanation_name in [
            "gim",
            "atpstar",
            "grad",
            "ig",
        ]:
            config = get_config(explanation_name)

            sufficiency_list = []
            comprehensiveness_list = []
            explanation_names = []
            id_list = []
            prob_list = []
            attribution_list = []

            for example in track(
                dataset, description="Evaluating explanations", total=len(dataset)
            ):
                gradient_manager = None
                torch.cuda.empty_cache()
                input_ids = torch.tensor(example["input_ids"]).to(device)
                answer = torch.tensor(example["target_token"]).to(device)

                model.cfg.ungroup_grouped_query_attention = True

                logits, clean_cache = model.run_with_cache(
                    input_ids,
                )

                full_output = logits[:, -1, answer].squeeze().cpu().item()

                output_embeddings_grad = trace.answer_direction(answer)

                # Run the backward pass with the new API
                if explanation_name == "atpstar":
                    clean_embeddings = clean_cache["blocks.0.hook_resid_pre"]
                    corrupt_embeddings = einops.repeat(
                        clean_embeddings.mean(1),
                        "batch d_model -> batch seq_len d_model",
                        seq_len=clean_embeddings.shape[1],
                    )
                    _, corrupt_cache = model.run_with_cache(
                        corrupt_embeddings, start_at_layer=0
                    )
                    gradient_manager = trace.backward(
                        output_embeddings_grad=output_embeddings_grad,
                        clean_cache=clean_cache,
                        corrupted_cache=corrupt_cache,
                        config=config,
                        return_prune_scores=False,
                    )
                elif explanation_name != "ig":
                    gradient_manager = trace.backward(
                        output_embeddings_grad=output_embeddings_grad,
                        clean_cache=clean_cache,
                        config=config,
                        return_prune_scores=False,
                    )
                model.cfg.ungroup_grouped_query_attention = False

                comprehensiveness_layers = []
                sufficiency_layers = []
                for layer in range(0, model.cfg.n_layers):
                    clean_embedding = clean_cache[f"blocks.{layer}.hook_resid_pre"]

                    if explanation_name == "ig":
                        corrupted_embedding = clean_embedding.mean(1, keepdim=True)

                        grad = torch.zeros_like(clean_embedding)
                        for step in range(0, IG_STEPS):
                            alpha = step / (IG_STEPS - 1)
                            interpolated_embedding = (
                                1 - alpha
                            ) * clean_embedding + alpha * corrupted_embedding
                            interpolated_embedding.requires_grad_()
                            output = model(
                                interpolated_embedding, start_at_layer=layer
                            )[0, -1, answer]
                            output.backward()
                            grad += interpolated_embedding.grad
                            interpolated_embedding.grad = None
                            torch.cuda.empty_cache()
                        grad /= IG_STEPS
                    else:
                        if gradient_manager is None:
                            raise ValueError(
                                "Gradient manager is None. Please check the explanation name."
                            )
                        grad = gradient_manager.mlp_and_token_grad[:, layer]

                    layer_attributions = einsum(
                        "batch seq_len d_model, batch seq_len d_model -> seq_len",
                        (clean_embedding - clean_embedding.mean(1, keepdim=True)),
                        grad,
                    ).cpu()

                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        with torch.autocast(
                            device_type="cuda", dtype=torch.float16, enabled=True
                        ):
                            layer_ranking = (
                                torch.argsort(layer_attributions[1:-1], descending=True)
                                + 1
                            )
                            permutation_embedding = clean_embedding.clone()

                            comprehensiveness = np.zeros(50)
                            for idx, token_idx_to_patch in enumerate(
                                layer_ranking[:50]
                            ):
                                permutation_embedding[:, token_idx_to_patch] = (
                                    clean_embedding.mean(1)
                                )

                                comprehensiveness[idx] = (
                                    full_output
                                    - model(
                                        permutation_embedding, start_at_layer=layer
                                    )[:, -1, answer]
                                    .squeeze()
                                    .cpu()
                                    .item()
                                )

                            # # calculate sufficiency
                            sufficiency = np.zeros(len(masking_ratio_sufficiency))
                            permutation_embedding = clean_embedding.clone()
                            token_ranking_flipped = layer_ranking.flip(0)
                            for idx, mask_ratio in enumerate(masking_ratio_sufficiency):
                                n_tokens_to_mask = int(
                                    mask_ratio * len(token_ranking_flipped)
                                )
                                token_idx_to_patch = token_ranking_flipped[
                                    :n_tokens_to_mask
                                ]
                                permutation_embedding[:, token_idx_to_patch] = (
                                    clean_embedding.mean(1)
                                )
                                sufficiency[idx] = (
                                    full_output
                                    - model(
                                        permutation_embedding, start_at_layer=layer
                                    )[:, -1, answer]
                                    .squeeze()
                                    .cpu()
                                    .item()
                                )
                            print(
                                f"Comprehensiveness: {comprehensiveness.mean() / full_output} - Sufficiency: {sufficiency.mean() / full_output} - Layer: {layer}"
                            )
                            comprehensiveness_layers.append(
                                comprehensiveness.mean() / full_output
                            )
                            sufficiency_layers.append(sufficiency.mean() / full_output)

                explanation_names.append(explanation_name)
                comprehensiveness_list.append(comprehensiveness_layers)
                sufficiency_list.append(sufficiency_layers)
                prob_list.append(full_output)
                id_list.append(example["annotation_id"])
                attribution_list.append(layer_attributions.cpu().numpy())

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
            print("saving to", path)
            path.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path / "layer_aopc.parquet")
