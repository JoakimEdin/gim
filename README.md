Official repository for the paper GIM: Improved Interpretability for Large Language Models

I've tested many approaches before ending up with GIM, and this is visible in the complexity of the code. GIM can be implemented in a simpler manner than in this repository, for example, by using Transformer Lens. I will release a better implementation after I've handed in my PhD thesis 1st of August.

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
