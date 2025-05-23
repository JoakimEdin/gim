Official repository for the paper GIM: Improved Interpretability for Large Language Models

## Setup
### Setup uv and pre-install

```bash
make setup
```

### Download the datasets

```bash
make download_data
```
You must download the twitter sentiment classification manually from `https://www.kaggle.com/competitions/tweet-sentiment-extraction/data`

### Compute classification accuracy of the large language models
```bash
CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_models.py
```
Change CUDA_VISIBLE_DEVICES if you want to use a different GPU.


## Running experiments
You can reproduce our three experiments using the following lines of code:

### Self-repair experiments
```bash
CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_self_repair.py
```

### Feature attribution experiments
```bash
CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_feature_attributions.py
```
This command will also compute the results needed for the ablation study. This will be take a lot of time. You can change the parameters in the code to only run a few models in the same run.

### Circuit identification experiments
```bash
CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_layers.py
```

## Figures and tables
The code for creating the figures and tables are in the `/results`folder.
