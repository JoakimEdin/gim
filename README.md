# Official repository for the paper GIM: Improved Interpretability for Large Language Models.

GIM (Gradient Interaction Modifications) is a state-of-the-art feature attribution method and circuit discovery method. It currently leads the leaderboard for the [Mechanistic Interpretability Benchmark](https://huggingface.co/spaces/mib-bench/leaderboard), while being as fast as gradients.

We have created [this PyPI package](https://pypi.org/project/gim-explain/) to make it effortless to use GIM on any Large Language Model. The code for the PyPI package is found in [this repository](https://github.com/corticph/gim).

The code in this repository is for reproducing the experiments in the paper. The code is less useful for other use cases.

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
