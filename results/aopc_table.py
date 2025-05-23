# type: ignore
# ruff: noqa: F841
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.dataset_factory import DATASETS
from src.models import MODELS

results_dir = Path("results")

dataset_names = list(DATASETS.keys())
model_names = list(MODELS.keys())
print(model_names)
explanation_names = [
    "random",
    "grad_baseline",
    "integrated_gradient",
    "integrated_gradients",
    "deeplift",
    "attnlrp_baseline",
    "transformerlrp",
    "grad_ln_freeze_scale_baseline",
    "softmax_t2",  # The names I used for my experiments made no sense. Change these.
]
results = []
for dataset_name in dataset_names:
    for model_name in model_names:
        for explanation_name in explanation_names:
            file_path = (
                results_dir
                / dataset_name
                / model_name
                / explanation_name
                / "aopc.parquet"
            )
            if (model_name == "qwen2-1.5b-it") and (
                explanation_name == "attnlrp_baseline"
            ):
                print(model_name, explanation_name, dataset_name)
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df = df[df["explanation_method"] == explanation_name]
                df["model"] = model_name
                df["dataset"] = dataset_name
                results.append(df)

results = pd.concat(results).reset_index(drop=True)
print(results["model"].unique())

results["comprehensiveness"] = (
    results["comprehensiveness"].apply(lambda x: np.mean(x)) / results["prob"]
)
results["sufficiency"] = (
    results["sufficiency"].apply(lambda x: np.mean(x)) / results["prob"]
)
results = results.drop(columns=["prob"])
results = results.drop(columns=["id"])
results = results.drop(columns=["attributions"])

results["model"] = results["model"].replace(
    {
        "gemma2-2b-it": "Gemma2.2 2B",
        "gemma2-9b-it": "Gemma2.2 9B",
        "llama3-1b-it": "Llama3.2 1B",
        "llama3-3b-it": "Llama3.2 3B",
        "llama3-8b-it": "Llama3.1 8B",
        "qwen2-1.5b-it": "Qwen2 1.5B",
        "qwen2-3b-it": "Qwen2 3B",
        "qwen2-7b-it": "Qwen2 7B",
    }
)
print(results["model"].unique())
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

results["explanation_method"] = results["explanation_method"].replace(
    {
        "grad_baseline": "GxI",
        "attnlrp_baseline": "AttnLRP",
        "random": "Random",
        "gim": "DecompGrad",
        "deeplift": "DeepLIFT",
        "integrated_gradient": "IG",
        "integrated_gradients": "IG",
        "transformerlrp": "TransformerLRP",
        "atpstar": "ATP*",
        "grad_ln_freeze_scale_baseline": "Grad LN Freeze Scale Baseline",
        "softmax_t2": "GIM",
    }
)
# sort rows by the explanation_method list
explanation_method_order = [
    "GxI",
    "IG",
    "DeepLIFT",
    "TransformerLRP",
    "AttnLRP",
    "GIM",
]

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
        # "Qwen2 7B",
    ],
    ordered=True,
)
results["explanation_method"] = pd.Categorical(
    results["explanation_method"],
    categories=explanation_method_order,
    ordered=True,
)
print(results["model"].unique())
# First group to get the mean values
results_grouped = (
    results.groupby(["dataset", "model", "explanation_method"]).mean().reset_index()
)

# Set the attnlrp results for Gemma to NaN
results_grouped.loc[
    (results_grouped["model"] == "Gemma2.2 2B")
    & (results_grouped["explanation_method"] == "AttnLRP"),
    ["comprehensiveness", "sufficiency"],
] = np.nan

results_grouped.loc[
    (results_grouped["model"] == "Gemma2.2 9B")
    & (results_grouped["explanation_method"] == "AttnLRP"),
    ["comprehensiveness", "sufficiency"],
] = np.nan

print(results_grouped["model"].unique())


# Then pivot the table to have datasets as columns and (model, explanation_method) as rows
# Each dataset column will split into comprehensiveness and sufficiency
results_pivot = results_grouped.pivot_table(
    index=["dataset", "explanation_method"],
    columns=["model"],
    values=["comprehensiveness", "sufficiency"],
)

# Print column structure to debug
print("Pivot table columns:")
print(results_pivot.columns.tolist())

datasets = results["dataset"].cat.categories
explanations = results["explanation_method"].cat.categories

# Create a multi-index with the desired order
idx = pd.MultiIndex.from_product(
    [datasets, explanations], names=["dataset", "explanation_method"]
)

# Reindex the pivot table to enforce the order
results_pivot = results_pivot.reindex(idx)
# Rename columns to title case
results_pivot.columns = results_pivot.columns.map(lambda x: (x[0].title(), x[1]))


# Create a styling function to highlight highest comprehensiveness and lowest sufficiency
def highlight_best(results_pivot):
    styles = pd.DataFrame("", index=results_pivot.index, columns=results_pivot.columns)

    # Process each dataset separately
    for dataset in datasets:
        dataset_slice = results_pivot.loc[dataset]

        # For each model and metric, highlight the best score
        for model in results_pivot.columns.get_level_values(1).unique():
            # Best comprehensiveness (highest value)
            comp_col = ("Comprehensiveness", model)
            if comp_col in results_pivot.columns:
                max_val = dataset_slice[comp_col].max()
                max_idx = dataset_slice[dataset_slice[comp_col] == max_val].index
                styles.loc[(dataset, max_idx), comp_col] = "font-weight: bold;"

            # Best sufficiency (lowest value)
            suff_col = ("Sufficiency", model)
            if suff_col in results_pivot.columns:
                min_val = dataset_slice[suff_col].min()
                min_idx = dataset_slice[dataset_slice[suff_col] == min_val].index
                styles.loc[(dataset, min_idx), suff_col] = "font-weight: bold;"

    return styles


# Apply formatting and highlighting
results_styled = results_pivot.style.format(
    decimal=".", thousands=" ", precision=2, na_rep="N/A"
).apply(highlight_best, axis=None)


# Generate LaTeX with midrules and bold formatting
latex_table = results_styled.to_latex(
    hrules=True,
    multicol_align=True,
    convert_css=True,  # This ensures CSS styles like font-weight: bold are converted to LaTeX
)

latex_table = latex_table.replace("\multirow", "\\midrule\n\multirow")


print(latex_table)
