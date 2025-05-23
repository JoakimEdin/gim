from pathlib import Path
import random

import pandas as pd

# some datasets are too large. This script will create a subset of the ids that all models correctly predicts

results_dir = Path("results")


dataset_names = ["hatexplain", "twitter", "fever"]
model_names = [
    "llama3-1b-it",
    "llama3-3b-it",
    "llama3-8b-it",
    "gemma2-2b-it",
    "gemma2-9b-it",
    "qwen2-1.5b-it",
    "qwen2-3b-it",
]
subset_size = 300

for dataset_name in dataset_names:
    overlapping_ids = None
    for model_name in model_names:
        file_path = results_dir / dataset_name / model_name / "correct_positive_ids.csv"
        df = pd.read_csv(file_path)
        if overlapping_ids is None:
            overlapping_ids = set(df.id.tolist())
        else:
            overlapping_ids = overlapping_ids.intersection(set(df.id.tolist()))

    if overlapping_ids is None:
        print(f"No overlapping ids for dataset: {dataset_name}")
        continue

    subset_ids = random.sample(list(overlapping_ids), subset_size)
    print(f"Dataset: {dataset_name}")
    print(f"subset size: {subset_size}")
    print(f"Overlapping ids: {len(overlapping_ids)}")
    print(f"subset ids: {len(subset_ids)}")

    # save subset at data/raw/datasetname/subset.csv
    subset_df = pd.DataFrame(subset_ids, columns=["id"])
    subset_df.to_csv(f"data/raw/{dataset_name}/subset.csv", index=False)

# print(dataset_overlapping_ids)
