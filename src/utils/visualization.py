# ruff: noqa: F722
from typing import Optional

import plotly.express as px
import torch
import transformer_lens
from IPython.display import HTML

# import transformer_lens
import transformer_lens.utils as utils
from jaxtyping import Float
from circuitsvis.attention import attention_heads


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


def show_attention_heads(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def visualize_attention_patterns(
    heads: tuple[list[int], int],
    local_cache,
    local_tokens: torch.Tensor,
    model: transformer_lens.HookedTransformer,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: list[str] = []
    patterns: list[Float[torch.Tensor, "query_pos key_pos"]] = []

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = model.to_str_tokens(local_tokens)

    # Combine the patterns into a single tensor
    patterns = torch.stack(patterns, dim=0)

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"


def visualize_most_important_attention_heads(
    per_head_logit_diffs: Float[torch.Tensor, "n_layers n_heads"], model, top_k: int = 5
) -> None:
    top_positive_logit_attr_heads = torch.topk(
        per_head_logit_diffs.flatten(), k=top_k
    ).indices

    positive_html = visualize_attention_patterns(
        top_positive_logit_attr_heads,
        model.cache,
        model.tokens[0],
        model,
        f"Top {top_k} Positive Logit Attribution Heads",
    )

    top_negative_logit_attr_heads = torch.topk(
        -per_head_logit_diffs.flatten(), k=top_k
    ).indices

    negative_html = visualize_attention_patterns(
        top_negative_logit_attr_heads,
        model.cache,
        model.tokens[0],
        model,
        title=f"Top {top_k} Negative Logit Attribution Heads",
    )

    return HTML(positive_html + negative_html)
