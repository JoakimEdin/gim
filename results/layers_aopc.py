# type: ignore
# ruff: noqa: F841
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.dataset_factory import DATASETS
from src.models import MODELS

plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": "serif"})
results_dir = Path("results")
figures_dir = Path("figures/layer_aopc")

dataset_names = list(DATASETS.keys())
model_names = list(MODELS.keys())
explanation_names = [
    "grad",
    "ig",
    "atpstar",
    "t2",
]

for dataset_name in dataset_names:
    for model_name in model_names:
        results = []
        for explanation_name in explanation_names:
            file_path = (
                results_dir
                / dataset_name
                / model_name
                / explanation_name
                / "layer_aopc.parquet"
            )
            if file_path.exists():
                df = pd.read_parquet(file_path)

                df["model"] = (
                    model_name.replace("gemma2-2b-it", "Gemma2.2 2B")
                    .replace("llama3-1b-it", "Llama3.2 1B")
                    .replace("llama3-3b-it", "Llama3.2 3B")
                    .replace("llama3-8b-it", "Llama3.1 8B")
                )
                df["dataset"] = (
                    dataset_name.replace("fever", "FEVER")
                    .replace("scifact", "SciFact")
                    .replace("boolq", "BoolQ")
                    .replace("hatexplain", "HateXplain")
                    .replace("twitter", "Twitter")
                    .replace("movie", "Movie")
                )

                results.append(df)
        if len(results) == 0:
            continue
        results = pd.concat(results).reset_index(drop=True)
        results["explanation_method"] = results["explanation_method"].replace(
            {
                "grad": "Attribution Patching",
                "atpstar": "ATP*",
                "ig": "Integrated Gradients",
                "grad_freeze": "+ LN freeze",
                "grad_norm": "+ Grad norm",
                "grad_freeze_norm": "+ LN freeze + Grad norm",
                "t2": "GIM",
            }
        )
        results = results.drop(columns=["attributions"])
        results = results.drop_duplicates(["id", "explanation_method"])
        results["Layer"] = results["comprehensiveness"].apply(
            lambda x: np.arange(len(x))
        )
        results = results.explode(["comprehensiveness", "sufficiency", "Layer"])

        sns.set(context="paper", style="whitegrid", font_scale=3)
        sns.set_palette("colorblind")
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=results,
            x="Layer",
            y="comprehensiveness",
            hue="explanation_method",
            errorbar="ci",
            markers=True,
            dashes=False,
        )
        plt.legend(title=None)
        plt.ylabel("$\\rightarrow$\nComprehensiveness")

        figure_path = figures_dir / dataset_name / model_name
        figure_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            figure_path / "layer_aopc_comprehensiveness.png",
            bbox_inches="tight",
            dpi=300,
        )

        plt.clf()
        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=results,
            x="Layer",
            y="sufficiency",
            hue="explanation_method",
            errorbar="ci",
            markers=True,
            dashes=False,
        )
        # remove legend title
        plt.legend(title=None)
        plt.ylabel("$\\leftarrow$\nSufficiency")
        # save figure
        plt.savefig(
            figure_path / "layer_aopc_sufficiency.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.clf()
