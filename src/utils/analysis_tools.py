from pathlib import Path

import pandas as pd
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from datasets import Dataset


from src.utils.tensor import get_device
from src.models import MODELS
from src.utils.dataset_factory import DATASETS
from src.utils.tokenizer import (
    map_character_spans_to_token_ids,
    map_n_characters_to_n_tokens,
)


def load_model_dataset(
    model_name: str, dataset_name: str
) -> tuple[HookedTransformer, Dataset, AutoTokenizer]:
    model_path = MODELS[model_name]
    print(f"Loading model: {model_path}...")
    device = get_device()
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
    model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    results_dir = Path("results")
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

    dataset = dataset.map(
        lambda x: tokenizer(
            x["prompt"],
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
        ),
        num_proc=8,
        batch_size=1000,
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"][0]) <= 1024)

    # # Update the evidence spans to account for the prompt
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

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "offset_mapping"],
        output_all_columns=True,
    )

    dataset = dataset.map(
        lambda x: {
            "evidence_token_ids_prompt": map_character_spans_to_token_ids(
                x["offset_mapping"], x["evidence_spans_prompt"]
            ),
        },
    )

    dataset = dataset.map(
        lambda x: {
            "n_tokens_before_context": map_n_characters_to_n_tokens(
                x["offset_mapping"], x["n_characters_before_context"]
            )
            - 1
        }
    )
    dataset = dataset.map(
        lambda x: {
            "n_tokens_after_context": map_n_characters_to_n_tokens(
                (x["offset_mapping"].max() - x["offset_mapping"]).flip(1),
                x["n_characters_after_context"],
            )
            + 1
        }
    )
    dataset = dataset.map(
        lambda x: {
            "evidence_token_ids": [
                token_id - x["n_tokens_before_context"]
                for token_id in x["evidence_token_ids_prompt"]
            ]
        },
    )

    # Get token IDs for the options
    options_token_id = [
        token_id
        for token_id in tokenizer(options, add_special_tokens=False)["input_ids"]
    ]
    dataset = dataset.map(
        lambda x: {"answer": options_token_id[x["label"]]},
    )
    dataset = dataset.map(
        lambda x: {"wrong_answer": options_token_id[1 - x["label"]]},
    )
    return model, dataset, tokenizer
