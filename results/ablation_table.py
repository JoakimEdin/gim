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
model_names = [
    "gemma2-2b-it",
    "gemma2-9b-it",
    "llama3-1b-it",
    "llama3-3b-it",
    "llama3-8b-it",
    "qwen2-1.5b-it",
    "qwen2-3b-it",
]
explanation_names = [
    # "random",
    # "gradient_x_input",
    # "integrated_gradients",
    # "deeplift",
    # "lime",
    # "attnlrp",
    # "decompgrad",
    "gim_query_subtract",
    "softmax_smoothgrad_b2_x2",
    "grad_weight-0.5",
    "gim",
    "grad_ln_freeze_scale",
    "grad_ln_freeze_scale_baseline",
    "grad_direct",
    "grad_ln_freeze",
    "grad_scale",
    "grad_baseline",
    "grad_norm_baseline",
    "grad_freeze_baseline",
    "grad_freeze_norm_baseline",
    "grad_norm",
    "softmax_t2_norm",
    "softmax_t2_freeze",
    "softmax_t2_only",
    "softmax_t1.5",
    "softmax_t2.5",
    "softmax_t_sample",
    "softmax_t2",
    "softmax_t3",
    "softmax_t4",
    "softmax_t5",
    "softmax_t2_ln_skip",
]  # The names I used for my experiments made no sense. Change these.
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
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df["model"] = model_name
                df["dataset"] = dataset_name
                results.append(df)

results = pd.concat(results).reset_index(drop=True)

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

results["explanation_method"] = results["explanation_method"].replace(
    {
        "integrated_gradient": "IG",
        "grad_direct": "DirectGrad",
        "grad_weight-0.5": "Grad-T0.1",
        "gim_query_subtract": "DecompGradQ",
        "softmax_t1.5": "+ Temperature 1.5",
        "softmax_t2.5": "+ Temperature 2.5",
        "softmax_t2": "+ Grad norm + LN freeze + TGM",
        "softmax_t3": "+ Temperature 3",
        "softmax_t4": "+ Temperature 4",
        "softmax_t5": "+ Temperature 5",
        "grad_baseline": "Gradient X Input",
        "grad_freeze_baseline": "+ LN freeze",
        "grad_ln_freeze_scale_baseline": "+ Grad norm + LN freeze",
        "grad_freeze_norm_baseline": "+ Grad norm + LN freeze",
        "grad_norm": "+ Grad norm",
        "grad_scale": "+ Grad norm",
        "softmax_t2_norm": "+ Grad norm + TGM",
        "softmax_t2_only": "+ TGM",
        "softmax_t2_freeze": "+ LN freeze + TGM",
        "softmax_t2_ln_skip": "+ Temperature 2 ln skip",
        "softmax_t_sample": "+ Temperature sample",
    }
)

# sort rows by the explanation_method list
explanation_method_order = [
    "Gradient X Input",
    "+ LN freeze",
    "+ Grad norm",
    "+ TGM",
    "+ Grad norm + TGM",
    "+ LN freeze + TGM",
    "+ Grad norm + LN freeze",
    "+ Grad norm + LN freeze + TGM",
    # "+ Temperature 1.5",
    # "+ Temperature 2.5",
    # "+ Temperature sample",
    # "+ Temperature 2 ln skip",
    # "+ Temperature 1",
    # "+ Temperature 1.5",
    # "+ Temperature 2",
    # "+ Temperature 3",
    # "+ Temperature 5",
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
        "Llama3.2 1B",
        "Llama3.2 3B",
        "Qwen2 1.5B",
        "Qwen2 3B",
    ],
    ordered=True,
)
results["explanation_method"] = pd.Categorical(
    results["explanation_method"],
    categories=explanation_method_order,
    ordered=True,
)
# First group to get the mean values
results_grouped = (
    results.groupby(["dataset", "model", "explanation_method"]).mean().reset_index()
)

# Then pivot the table to have datasets as columns and (model, explanation_method) as rows
# Each dataset column will split into comprehensiveness and sufficiency
results_pivot = results_grouped.pivot_table(
    index=["dataset", "explanation_method"],
    columns=["model"],
    values=["comprehensiveness", "sufficiency"],
)

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

# Generate LaTeX with midrules before each model
latex_table = results_styled.to_latex(
    hrules=True, multicol_align=True, convert_css=True
)

latex_table = latex_table.replace("\multirow", "\\midrule\n\multirow")


print(latex_table)
