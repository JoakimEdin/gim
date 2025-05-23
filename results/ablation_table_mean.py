# type: ignore
# ruff: noqa: F841
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.dataset_factory import DATASETS
from src.models import MODELS


def subtract_grad_scores(rows):
    rows[["comprehensiveness", "sufficiency"]] -= rows[
        rows["explanation_method"] == "Gradient X Input"
    ][["comprehensiveness", "sufficiency"]].values
    return rows


results_dir = Path("results")

dataset_names = list(DATASETS.keys())
model_names = list(MODELS.keys())
model_names = [
    "gemma2-2b-it",
    "llama3-1b-it",
    "llama3-3b-it",
    "qwen2-1.5b-it",
    "qwen2-3b-it",
]
explanation_names = [
    "grad_ln_freeze_scale_baseline",
    "grad_freeze_norm_baseline",
    "softmax_t2",
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
        "gemma2-2b-it": "Gemma-2B",
        "gemma2-9b-it": "Gemma 9B",
        "llama3-1b-it": "Llama-1B",
        "llama3-3b-it": "Llama-3B",
        "llama3-8b-it": "Llama 8B",
        "qwen2-1.5b-it": "Qwen-1.5B",
        "qwen2-3b-it": "Qwen-3B",
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
        "softmax_t2": "+ Grad norm + LN freeze + TSG",
        "grad_ln_freeze_scale_baseline": "+ Grad norm + LN freeze",
        "grad_freeze_norm_baseline": "+ Grad norm + LN freeze",
    }
)


# sort rows by the explanation_method list
explanation_method_order = [
    "+ Grad norm + LN freeze",
    "+ Grad norm + LN freeze + TSG",
]


# First group to get the mean values
results_shifted_list = []
for explanation_method in explanation_method_order:
    out = results.groupby(["dataset", "model"]).apply(
        lambda x: (
            x[x["explanation_method"] == explanation_method][
                ["comprehensiveness", "sufficiency"]
            ]
            - x[x["explanation_method"] == "Gradient X Input"][
                ["comprehensiveness", "sufficiency"]
            ].values
        )
    )
    out["explanation_method"] = explanation_method
    results_shifted_list.append(out)
# how do I change the groupby index names to column names?
results_shifted = pd.concat(results_shifted_list).reset_index()
results_shifted = (
    results_shifted.groupby(["dataset", "model", "explanation_method"])
    .mean()
    .reset_index()
)
# set latex
plt.rcParams.update({"text.usetex": True})

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.5, context="paper")
sns.catplot(
    data=results_shifted,
    x="model",
    y="comprehensiveness",
    hue="explanation_method",
    dodge=True,
    jitter=False,
    kind="strip",
    legend_out=False,
    aspect=1.5,
)
sns.boxplot(
    data=results_shifted,
    x="model",
    y="comprehensiveness",
    hue="explanation_method",
    fill=False,
    legend=False,
)
plt.ylabel("$\\rightarrow$ \nComprehensiveness")
plt.xlabel("")
plt.legend(title=None)
plt.savefig("figures/ablation_bar_comprehensiveness.png", dpi=300, bbox_inches="tight")
plt.clf()

sns.catplot(
    data=results_shifted,
    x="model",
    y="sufficiency",
    hue="explanation_method",
    dodge=True,
    jitter=False,
    kind="strip",
    legend_out=False,
    aspect=1.5,
)
sns.boxplot(
    data=results_shifted,
    x="model",
    y="sufficiency",
    hue="explanation_method",
    fill=False,
    legend=False,
)
plt.ylabel("$\leftarrow$ \n Sufficiency")
plt.xlabel("")
plt.legend(title=None)
plt.savefig("figures/ablation_bar_sufficiency.png", dpi=300, bbox_inches="tight")
