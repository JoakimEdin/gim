import pandas as pd
from transformers import AutoTokenizer
import numpy as np

from src.utils.dataset_factory import DATASETS
from src.models import MODELS

model_name = "llama3-1b-it"
model_path = MODELS[model_name]
tokenizer = AutoTokenizer.from_pretrained(
    model_path, clean_up_tokenization_spaces=False
)
statistics = []
for dataset_name in DATASETS.keys():
    dataset_config = DATASETS[dataset_name]
    dataset = dataset_config.loader()
    create_prompt = dataset_config.get_model_specific_prompt(model_name)
    if "query" in dataset.column_names:
        dataset = dataset.map(
            lambda x: create_prompt(context=x["context"], query=x["query"])
        )
    else:
        dataset = dataset.map(
            lambda x: create_prompt(context=x["context"]),
        )

    dataset = dataset.map(
        lambda x: tokenizer(x["prompt"], return_tensors="pt", truncation=True),
        remove_columns=["context"],
        num_proc=8,
        batch_size=1000,
    )

    dataset = dataset.map(
        lambda x: {"length": len(x["input_ids"][0])},
    )

    statistics.append(
        {
            "dataset_name": dataset_name,
            "num_samples": len(dataset),
            "num_tokens": np.mean(dataset["length"]),
        }
    )

statistics_df = pd.DataFrame(statistics).to_latex()
print(statistics_df)
