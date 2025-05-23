# Inspired by: https://blog.mathieu-leplatre.info/tips-for-your-makefile-with-python.html
# 			   https://www.thapaliya.com/en/writings/well-documented-makefiles/

.DEFAULT_GOAL := help

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: install
install:  ## Install the package for development along with pre-commit hooks.
	uv install --with dev

.PHONY: clean
clean:  ## Clean up the project directory removing __pycache__, .coverage, and the install stamp file.
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf coverage.xml test-output.xml test-results.xml htmlcov .pytest_cache .ruff_cache

setup:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv sync
	uv run pre-commit install
	export PYTHONHASHSEED=42

download_data:
	mkdir -p data/raw
	mkdir -p data/raw/hatexplain
	wget -O data/raw/boolq.tar.gz https://www.eraserbenchmark.com/zipped/boolq.tar.gz
	wget -O data/raw/fever.tar.gz https://www.eraserbenchmark.com/zipped/fever.tar.gz
	wget -O data/raw/movies.tar.gz https://www.eraserbenchmark.com/zipped/movies.tar.gz
	wget -O data/raw/scifact.tar.gz https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
	wget -O data/raw/hatexplain/hatexplain.json https://github.com/hate-alert/HateXplain/raw/refs/heads/master/Data/dataset.json
	wget -O data/raw/hatexplain/split.json https://github.com/hate-alert/HateXplain/raw/refs/heads/master/Data/post_id_divisions.json
	tar -xvf data/raw/boolq.tar.gz -C data/raw
	tar -xvf data/raw/fever.tar.gz -C data/raw
	tar -xvf data/raw/movies.tar.gz -C data/raw
	tar -xvf data/raw/scifact.tar.gz -C data/raw && mv data/raw/data data/raw/scifact
	rm data/raw/*.tar.gz

classification_eval:
	CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_models.py

self_repair_eval:
	CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_self_repair.py

feature_attribution_eval:
	CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_feature_attributions.py

circuits_eval:
	CUDA_VISIBLE_DEVICES="0" uv run python src/evaluation/evaluate_layers.py
