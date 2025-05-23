# ruff: noqa: F722
# mypy: disable-error-code="name-defined"

from typing import Literal
import logging

import torch
from fancy_einsum import einsum
from jaxtyping import Float
from transformer_lens import HookedTransformer

from src.gim.utils import CacheManager


def folded_layernorm_derivative(
    ln_in: Float[torch.Tensor, "... d_model"],
    ln_out_grad: Float[torch.Tensor, "... d_model"],
):
    """
    Efficiently computes the gradient of the loss with respect to the input
    when we know dL/dy (gradient of loss with respect to folded_layernorm output).

    This avoids explicitly constructing the full Jacobian matrix.

    Args:
        ln_in: Input tensor with shape (batch_size, hidden_size)
        ln_out_grad: Gradient of loss w.r.t. output, same shape as x

    Returns:
        Gradient of loss w.r.t. input x with shape (batch_size, hidden_size)
    """
    hidden_size = ln_in.shape[-1]

    variance = torch.mean(
        torch.square(ln_in), dim=-1, keepdim=True
    )  # mean(x²) - shape: (..., 1)
    inv_std = torch.rsqrt(variance)  # 1/sqrt(var) - shape: (..., 1)
    inv_std_cubed = inv_std**3  # shape: (batch_size, 1)

    # For our function o = x / sqrt(mean(x²)), the Jacobian-vector product can be computed as:
    # J·v = v/sqrt(var) - (x·v) * x / (n·var^(3/2))

    # First term: v/sqrt(var)
    first_term = ln_out_grad * inv_std

    # Second term: (x·v) * x / (n·var^(3/2))
    # Compute x·v (dot product) for each item in the batch

    x_v_dot = einsum(
        "batch ... d_model, batch ... d_model -> batch ...", ln_in, ln_out_grad
    ).unsqueeze(-1)
    # x_v_dot = torch.sum(
    #     ln_in * ln_out_grad, dim=-1, keepdim=True
    # )  # shape: (batch_size, 1)

    # Multiply by x and scale by 1/(n·var^(3/2))
    second_term = x_v_dot * ln_in * inv_std_cubed / hidden_size

    # Combine the terms
    return first_term - second_term


class LayerNormBackpropagator:
    """Handles backpropagation through layer normalization."""

    def __init__(self, model: HookedTransformer, logger: logging.Logger):
        """
        Initialize the layer norm backpropagator.

        Args:
            model: The transformer model
            logger: Logger for recording information
        """
        self.model = model
        self.logger = logger

    def backward(
        self,
        layer_idx: int,
        ln_out_grad: Float[torch.Tensor, "batch ... source_pos d_model"],
        layernorm_method: Literal["grad", "freeze", "skip"],
        component_name: Literal["attention", "mlp"],
        cache_manager: CacheManager,
    ) -> Float[torch.Tensor, " batch ... source_pos d_model"]:
        """
        Apply layer normalization gradient approximation to input vectors.

        Args:
            layer_idx: Index of the transformer block
            input_vectors: Input vectors to apply layernorm gradient approximation to
            gradient_method: Method for approximating layernorm gradient:
                            "grad" - Use direct gradient computation
                            "freeze" - Use scale-based approximation
            cache_manager: Manager for accessing activation caches

        Returns:
            Vectors with layernorm gradient approximation applied
        """
        if layernorm_method == "skip":
            # Skip layernorm gradient approximation
            return ln_out_grad
        batch_size = ln_out_grad.shape[0]
        source_pos = ln_out_grad.shape[-2]
        d_model = ln_out_grad.shape[-1]

        if component_name == "attention":
            ln_name = "ln1.hook_scale"
        elif component_name == "mlp":
            ln_name = "ln2.hook_scale"

        else:
            raise ValueError(
                f"Unknown component_name: {component_name}. Use 'attention' or 'mlp'."
            )

        if layernorm_method == "grad":
            # Use direct gradient computation through layernorm
            if component_name == "attention":
                ln_input: Float[torch.Tensor, "batch source_pos d_model"] = (
                    cache_manager.get_block_hidden_state(layer_idx, "hook_resid_pre")
                )
            elif component_name == "mlp":
                ln_input = cache_manager.get_block_hidden_state(
                    layer_idx, "hook_resid_mid"
                )

            # Create the new shape for ln_scale_factors
            # Keep batch dimension, add 1's for middle dimensions, keep source_pos and d_model
            new_shape = (
                (batch_size,) + (1,) * (ln_out_grad.dim() - 3) + (source_pos, d_model)
            )

            # Reshape ln_scale_factors to match input_vectors for broadcasting
            ln_input = ln_input.reshape(new_shape)
            torch.cuda.empty_cache()

            return folded_layernorm_derivative(ln_input, ln_out_grad)

        elif layernorm_method == "freeze":
            # Use scale-based approximation (divide by layernorm scale)

            ln_scale: Float[torch.Tensor, "batch source_pos d_model"] = (
                cache_manager.get_block_hidden_state(layer_idx, ln_name)
            )

            self.logger.debug(f"LayerNorm scale factors shape: {ln_scale.shape}")
            self.logger.debug(f"Input vectors shape: {ln_out_grad.shape}")

            # Create the new shape for ln_scale
            # Keep batch dimension, add 1's for middle dimensions, keep source_pos and d_model
            new_shape = (batch_size,) + (1,) * (ln_out_grad.dim() - 3) + (source_pos, 1)

            # Reshape ln_scale to match input_vectors for broadcasting
            ln_scale = ln_scale.reshape(new_shape)

            return ln_out_grad / ln_scale

        else:
            raise ValueError(
                f"Unknown gradient_method: {layernorm_method}. Use 'grad' or 'freeze'."
            )
