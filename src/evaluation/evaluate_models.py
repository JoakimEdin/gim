import torch
import pandas as pd
import os
import tqdm.auto as tqdm
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.dataset_factory import DATASETS
from src.models import MODELS
from src.utils.tensor import get_device


# ============= Batching Utility =============
def create_batches(dataset, batch_size=8, max_length_diff=50):
    """Create batches of similar lengths for efficient processing"""
    examples = list(dataset)

    # Sort by input length
    examples.sort(key=lambda x: len(x["input_ids"]))

    batches = []
    current_batch = []
    base_length = None

    for example in examples:
        if not current_batch:
            current_batch = [example]
            base_length = len(example["input_ids"])
        elif (
            len(current_batch) < batch_size
            and abs(len(example["input_ids"]) - base_length) <= max_length_diff
        ):
            current_batch.append(example)
        else:
            batches.append(current_batch)
            current_batch = [example]
            base_length = len(example["input_ids"])

    if current_batch:
        batches.append(current_batch)

    return batches


def collate_batch(batch, tokenizer, device):
    """Prepare a batch with padding and attention masks using torch padding"""
    max_length = max(len(item["input_ids"][0]) for item in batch)

    input_ids_batch = (
        torch.ones(len(batch), max_length, dtype=torch.long, device=device)
        * tokenizer.pad_token_id
    )
    attention_mask_batch = torch.zeros(
        len(batch), max_length, dtype=torch.long, device=device
    )
    input_length_batch = torch.zeros(len(batch), dtype=torch.long, device=device)

    labels_list = []
    ids_list = []

    for idx, item in enumerate(batch):
        input_ids = torch.tensor(item["input_ids"][0])
        input_ids_batch[idx, : len(input_ids)] = input_ids
        attention_mask_batch[idx, : len(input_ids)] = 1
        labels_list.append(item["label"])
        ids_list.append(item["annotation_id"])
        input_length_batch[idx] = len(input_ids)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": torch.tensor(labels_list),
        "ids": ids_list,
        "lengths": input_length_batch,
    }


# ============= Evaluation Functions =============
def evaluate_model(
    model_name, dataset_config, batch_size=8, max_length_diff=50, output_dir="results"
):
    device = get_device()
    model_path = MODELS[model_name]

    print(f"Loading model: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, clean_up_tokenization_spaces=False
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {dataset_config.name}...")
    dataset = dataset_config.loader(use_subset=False)

    # Create prompt function with examples
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

    # Get token IDs for the options
    options_token_id = tokenizer(
        options, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].squeeze()

    # Create directory for results
    os.makedirs(output_dir, exist_ok=True)

    # Prepare results storage
    metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    prediction_results = []

    # Create batches
    batches = create_batches(dataset, batch_size, max_length_diff)

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for batch_idx, batch_examples in enumerate(tqdm.tqdm(batches)):
                batch = collate_batch(batch_examples, tokenizer, device)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )

                # Get the last non-padding token position for each sequence
                lengths = batch["lengths"] - 1  # Subtract 1 since indices are 0-based

                # Create a batch index tensor
                batch_indices = torch.arange(len(batch["lengths"]), device=device)

                # Extract the logits at the last position for each sequence
                # Shape: [batch_size, vocab_size]
                last_token_logits = outputs.logits[batch_indices, lengths]

                # Get the probabilities for each option token
                # Shape: [batch_size, num_options]
                option_logits = last_token_logits[:, options_token_id]

                # Get the predicted class
                predictions = option_logits.argmax(-1).cpu().numpy()
                labels = batch["labels"]

                for i, (pred, label, example_id) in enumerate(
                    zip(predictions, labels, batch["ids"])
                ):
                    # Store prediction result
                    prediction_results.append(
                        {
                            "id": example_id,
                            "true_label": label.item(),
                            "predicted_label": pred.item(),
                            "true_class": label,
                        }
                    )

                    # Update metrics
                    if pred == 1 and label == 1:
                        metrics["tp"] += 1
                    elif pred == 1 and label == 0:
                        metrics["fp"] += 1
                    elif pred == 0 and label == 0:
                        metrics["tn"] += 1
                    else:
                        metrics["fn"] += 1

    # Create dataframe with all predictions
    results_df = pd.DataFrame(prediction_results)

    # Calculate final metrics
    tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp + 1e-11)
    recall = tp / (tp + fn + 1e-11)
    f1 = 2 * precision * recall / (precision + recall + 1e-11)
    specificity = tn / (tn + fp + 1e-11)

    # Create metrics dataframe
    metrics_df = pd.DataFrame(
        [
            {
                "Accuracy": f"{accuracy:.4f}",
                "Precision": f"{precision:.4f}",
                "Recall": f"{recall:.4f}",
                "F1": f"{f1:.4f}",
                "Specificity": f"{specificity:.4f}",
                "True Positives": tp,
                "False Positives": fp,
                "True Negatives": tn,
                "False Negatives": fn,
                "Positive Examples": tp + fn,
                "Negative Examples": fp + tn,
            }
        ]
    )

    metrics_df.to_csv(output_dir / "results.csv", index=False)

    # Save correctly predicted positive example IDs
    correct_positives_df = results_df[
        (results_df["predicted_label"] == 1) & (results_df["true_label"] == 1)
    ]
    correct_positives_df[["id"]].to_csv(
        output_dir / "correct_positive_ids.csv", index=False
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Number of positive examples: {tp + fn}")
    print(f"Number of negative examples: {fp + tn}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# ============= Main Execution =============
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate models on different datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=MODELS.keys(),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=DATASETS.keys(),
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )

    args = parser.parse_args()

    if args.model == "all":
        model_names = MODELS.keys()
    else:
        model_names = [args.model]

    if args.dataset == "all":
        dataset_names = DATASETS.keys()
    else:
        dataset_names = [args.dataset]

    for dataset_name in dataset_names:
        for model_name in model_names:
            dataset_config = DATASETS[dataset_name]
            print(f"Evaluating {model_name} on {dataset_name}")
            output_dir = Path(args.output_dir) / dataset_name / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            evaluate_model(
                model_name, dataset_config, args.batch_size, output_dir=output_dir
            )


if __name__ == "__main__":
    main()
