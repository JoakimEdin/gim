# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from fancy_einsum import einsum
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


from src.utils.analysis_tools import load_model_dataset
from src.utils.tensor import get_device
from src.gim.transpose_trace import (
    TransformerTrace,
    TransformerTraceConfig,
)
from src.gim.utils import softmax_derivation, softmax_tsg


def perturb_attention(attention_scores, top_2_indices, dl_ds, positions):
    nan_attention_scores = attention_scores.clone()
    nan_attention_scores[torch.isinf(nan_attention_scores)] = torch.nan
    attention_score_mean = nan_attention_scores.nanmean(dim=-1)
    for position in positions:
        attention_scores[
            torch.arange(top_2_indices.shape[0]), top_2_indices[:, position]
        ] = attention_score_mean
    attention_pattern = torch.softmax(attention_scores, dim=-1)
    new_output = (attention_pattern * dl_ds).sum(-1)
    return new_output


sns.set(style="whitegrid", context="paper", font_scale=1.5, palette="colorblind")

figure_dir = Path("figures/self_repair")

device = get_device()

model_names = [
    "gemma2-2b-it",
    "llama3-1b-it",
    "llama3-3b-it",
    "qwen2-1.5b-it",
    "qwen2-3b-it",
]

dataset_names = [
    "boolq",
    "fever",
    "movie",
    "hatexplain",
    "twitter",
    "scifact",
]

for model_name in model_names:
    for dataset_name in dataset_names:
        print(f"Running {model_name} on {dataset_name}")

        figure_path = figure_dir / dataset_name / model_name
        figure_path.mkdir(parents=True, exist_ok=True)

        model, dataset, tokenizer = load_model_dataset(
            model_name=model_name,
            dataset_name=dataset_name,
        )
        baseline_token_id = tokenizer(
            " ", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(device)
        start_token_id = tokenizer.cls_token_id
        end_token_id = tokenizer.sep_token_id

        # Create the transformer trace object
        trace = TransformerTrace(model)
        config = TransformerTraceConfig(
            layernorm_backward="grad",
            activation_backward="grad",
            query_value_key_divide_grad_strategy=False,
            abs_edge_strength=False,
            value_only=False,
            softmax_backward="grad",
            log_level=0,
            scale_mlp_gate=False,
        )

        important_attention = 0

        original_output_list = []
        ablate_largest_list = []
        ablate_second_largest_list = []
        ablate_both_list = []

        two_largest_grad_list = []
        two_largest_tsg_list = []
        two_largest_dl_ds_list = []
        two_largest_s_list = []
        two_largest_a_list = []
        attention_list = []

        for example_idx in tqdm(range(len(dataset))):
            example = dataset[example_idx]
            answer = example["answer"].to(device)
            wrong_answer = example["wrong_answer"].to(device)
            input_ids = example["input_ids"].to(device)
            start_token_id = tokenizer.cls_token_id
            end_token_id = tokenizer.sep_token_id
            target_vector = trace.answer_direction(answer)

            with torch.no_grad():
                _, clean_cache = model.run_with_cache(
                    input_ids,
                )

            gradient_manager = trace.backward(
                output_embeddings_grad=target_vector,
                clean_cache=clean_cache,
                config=config,
                return_prune_scores=False,
            )

            for layer_idx in range(model.cfg.n_layers):
                hook_v = gradient_manager.cache_manager.get_block_hidden_state(
                    layer_idx=layer_idx,
                    hidden_state_name="attn.hook_v",
                ).squeeze(0)
                attention_pattern = (
                    gradient_manager.cache_manager.get_block_hidden_state(
                        layer_idx=layer_idx,
                        hidden_state_name="attn.hook_pattern",
                    ).squeeze(0)
                )
                attention_scores = (
                    gradient_manager.cache_manager.get_block_hidden_state(
                        layer_idx=layer_idx,
                        hidden_state_name="attn.hook_attn_scores",
                    ).squeeze(0)
                )
                dl_ds = einsum(
                    "batch n_heads query_pos d_model, key_pos n_heads d_model -> n_heads query_pos key_pos",
                    gradient_manager.cache_manager.clean_attn[:, layer_idx],
                    hook_v,
                )

                dl_ds_s = attention_pattern * dl_ds
                indices = torch.sum(dl_ds_s > 1, dim=-1) > 1
                top_2_indices = torch.topk(dl_ds_s[indices], 2, dim=-1).indices

                grad = softmax_derivation(attention_pattern[indices], dl_ds[indices])
                tsg = softmax_tsg(
                    attention_scores[indices], dl_ds[indices], temperature=2
                )

                original_output = dl_ds_s[indices].sum(-1)

                ablate_largest = perturb_attention(
                    attention_scores.clone()[indices],
                    top_2_indices,
                    dl_ds[indices],
                    [0],
                )
                abalate_second_largest = perturb_attention(
                    attention_scores.clone()[indices],
                    top_2_indices,
                    dl_ds[indices],
                    [1],
                )

                abalate_both = perturb_attention(
                    attention_scores.clone()[indices],
                    top_2_indices,
                    dl_ds[indices],
                    [0, 1],
                )

                two_largest_grad_list.append(torch.gather(grad, 1, top_2_indices))
                two_largest_tsg_list.append(torch.gather(tsg, 1, top_2_indices))
                two_largest_dl_ds_list.append(
                    torch.gather(dl_ds[indices], 1, top_2_indices)
                )
                two_largest_s_list.append(
                    torch.gather(attention_pattern[indices], 1, top_2_indices)
                )
                two_largest_a_list.append(
                    torch.gather(attention_scores[indices], 1, top_2_indices)
                )
                original_output_list.append(original_output)
                ablate_largest_list.append(ablate_largest)
                ablate_second_largest_list.append(abalate_second_largest)
                ablate_both_list.append(abalate_both)
                attention_list.append(attention_scores[indices])

                important_attention += torch.sum(
                    torch.sum(dl_ds_s > 1, dim=-1) > 0
                ).item()

        two_largest_grad = torch.concat(two_largest_grad_list, dim=0)
        two_largest_tsg = torch.concat(two_largest_tsg_list, dim=0)
        two_largest_dl_ds = torch.concat(two_largest_dl_ds_list, dim=0)
        two_largest_s = torch.concat(two_largest_s_list, dim=0)
        two_largest_a = torch.concat(two_largest_a_list, dim=0)
        original_output = torch.concat(original_output_list, dim=0)
        ablate_largest = torch.concat(ablate_largest_list, dim=0)
        ablate_second_largest = torch.concat(ablate_second_largest_list, dim=0)
        ablate_both = torch.concat(ablate_both_list, dim=0)

        del two_largest_grad_list
        del two_largest_tsg_list
        del two_largest_dl_ds_list
        del two_largest_s_list
        del two_largest_a_list
        del original_output_list
        del ablate_largest_list
        del ablate_second_largest_list
        del ablate_both_list

        ablate_largest_change = original_output - ablate_largest
        ablate_second_largest_change = original_output - ablate_second_largest
        ablate_both_change = original_output - ablate_both
        interaction_effect = (
            ablate_both_change - ablate_second_largest_change - ablate_largest_change
        ) / ablate_both_change
        interaction_effect[ablate_both_change == 0] = 0

        interaction_effect = interaction_effect.cpu()
        interaction_effect_q95 = interaction_effect.quantile(0.95).cpu().item()

        max_value = ablate_both_change.max().cpu().item()
        min_value = ablate_both_change.min().cpu().item()
        x = (ablate_largest_change + ablate_second_largest_change).cpu().numpy()
        y = ablate_both_change.cpu().numpy()
        plt.scatter(
            x[interaction_effect < interaction_effect_q95],
            y[interaction_effect < interaction_effect_q95],
            s=1,
        )
        plt.scatter(
            x[interaction_effect > interaction_effect_q95],
            y[interaction_effect > interaction_effect_q95],
            label="95% Quantile",
            s=1,
        )
        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="black",
            linestyle="--",
        )
        plt.ylabel("Joint Ablation Effect")
        plt.xlabel("Sum of Individual Ablation Effects")
        plt.savefig(figure_path / "ablation_effect.png", bbox_inches="tight", dpi=300)

        plt.clf()

        min_value = two_largest_grad.sum(-1).min().cpu().item()
        max_value = two_largest_grad.sum(-1).max().cpu().item()

        plt.scatter(
            (two_largest_grad[interaction_effect < interaction_effect_q95].sum(-1))
            .cpu()
            .numpy(),
            two_largest_tsg[interaction_effect < interaction_effect_q95]
            .sum(-1)
            .cpu()
            .numpy(),
            s=1,
        )
        plt.scatter(
            (two_largest_grad[interaction_effect > interaction_effect_q95].sum(-1))
            .cpu()
            .numpy(),
            two_largest_tsg[interaction_effect > interaction_effect_q95]
            .sum(-1)
            .cpu()
            .numpy(),
            s=1,
            label="95% Quantile interaction effect",
        )
        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            color="black",
            linestyle="--",
            alpha=1,
        )
        plt.ylabel("TSG")
        plt.xlabel("Gradient")
        plt.savefig(figure_path / "grad_vs_tsg.png", bbox_inches="tight", dpi=300)
        plt.clf()
