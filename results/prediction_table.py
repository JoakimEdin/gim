# type: ignore
# ruff: noqa: F841
from pathlib import Path

import pandas as pd

from src.utils.dataset_factory import DATASETS

results_dir = Path("results")
model_names = [
    "gemma2-2b-it",
    "gemma2-9b-it",
    "llama3-1b-it",
    "llama3-3b-it",
    "llama3-8b-it",
    "qwen2-1.5b-it",
    "qwen2-3b-it",
]

results = []
for dataset_name, dataset_desc in DATASETS.items():
    for model_name in model_names:
        print(f"{model_name} on {dataset_name}")
        file_path = results_dir / dataset_name / model_name / "results.csv"
        df = pd.read_csv(file_path)
        df["model"] = model_name
        df["dataset"] = dataset_name
        results.append(df)

results = pd.concat(results).reset_index(drop=True)
results = results.drop(
    columns=[
        "False Positives",
        "False Negatives",
        "True Negatives",
        "Positive Examples",
        "Negative Examples",
        "True Positives",
    ]
)
results["model"] = results["model"].replace(
    {
        "gemma2-2b-it": "Gemma2.2 2B",
        "gemma2-9b-it": "Gemma2.2 9B",
        "llama3-1b-it": "Llama3.2 1B",
        "llama3-3b-it": "Llama3.2 3B",
        "llama3-8b-it": "Llama3.1 8B",
        "qwen2-1.5b-it": "Qwen2 1.5B",
        "qwen2-3b-it": "Qwen2 3B",
    }
)
results["dataset"] = results["dataset"].replace(
    {
        "fever": "FEVER",
        "scifact": "SciFact",
        "boolq": "BoolQ",
        "hatexplain": "HateXplain",
        "twitter": "Twitter",
        "movie": "Movie",
    }
)

results["model"] = pd.Categorical(
    results["model"],
    categories=[
        "Gemma2.2 2B",
        "Gemma2.2 9B",
        "Llama3.2 1B",
        "Llama3.2 3B",
        "Llama3.1 8B",
        "Qwen2 1.5B",
        "Qwen2 3B",
    ],
    ordered=True,
)
results["dataset"] = pd.Categorical(
    results["dataset"],
    categories=[
        "BoolQ",
        "FEVER",
        "HateXplain",
        "Movie",
        "SciFact",
        "Twitter",
    ],
    ordered=True,
)
results[["Accuracy", "Precision", "Recall", "F1", "Specificity"]] = (
    results[["Accuracy", "Precision", "Recall", "F1", "Specificity"]] * 100
)
index_tuples = [
    (dataset, model) for dataset, model in zip(results["dataset"], results["model"])
]
results.index = pd.MultiIndex.from_tuples(index_tuples, names=["dataset", "model"])

results = results.drop(columns=["dataset", "model"])

results.columns = results.columns.str.title()
results = results.style.format(
    decimal=".",
    thousands=" ",
    precision=2,
)
print(results.to_latex(hrules=True, multicol_align=True))
