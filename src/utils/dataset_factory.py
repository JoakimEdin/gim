from typing import Callable, List
from dataclasses import dataclass

from datasets import Dataset

from src.utils.get_datasets import (
    load_boolq_dataset,
    load_fever_dataset,
    load_hatexplain_dataset,
    load_movies_dataset,
    load_scifact_dataset,
    load_twitter_dataset,
)

# Import configuration system for prompts and options
from src.configs.prompt_config import get_prompt_generator, get_options, DEFAULT_CONFIGS


@dataclass
class DatasetConfig:
    """Configuration for a dataset"""

    name: str
    loader: Callable[[], Dataset]

    @property
    def default_options(self) -> List[str]:
        """Get the default options for this dataset"""
        return DEFAULT_CONFIGS[self.name].options

    @property
    def default_prompt_generator(self) -> Callable:
        """Get the default prompt generator for this dataset"""
        return DEFAULT_CONFIGS[self.name].prompt

    def get_model_specific_prompt(self, model_name: str) -> Callable:
        """Get model-specific prompt generator for this dataset"""
        return get_prompt_generator(
            model_name, self.name, self.default_prompt_generator
        )

    def get_model_specific_options(self, model_name: str) -> List[str]:
        """Get model-specific options for this dataset"""
        return get_options(model_name, self.name, self.default_options)


# Dataset registry - much simpler now!
DATASETS = {
    "hatexplain": DatasetConfig(
        name="hatexplain",
        loader=load_hatexplain_dataset,
    ),
    "movie": DatasetConfig(
        name="movie",
        loader=load_movies_dataset,
    ),
    "twitter": DatasetConfig(
        name="twitter",
        loader=load_twitter_dataset,
    ),
    "boolq": DatasetConfig(
        name="boolq",
        loader=load_boolq_dataset,
    ),
    "fever": DatasetConfig(
        name="fever",
        loader=load_fever_dataset,
    ),
    "scifact": DatasetConfig(
        name="scifact",
        loader=load_scifact_dataset,
    ),
}
