# ruff: noqa: F722
# mypy: disable-error-code="name-defined"
import logging
from typing import Literal
import math

import torch
from jaxtyping import Float
import einops
from fancy_einsum import einsum
from transformer_lens import HookedTransformer

from src.gim.utils import CacheManager, GradientManager
from src.gim.layernorm import LayerNormBackpropagator


def gelu_derivative(x):
    """
    Derivative of the GeLU activation function with respect to its input.

    d/dx[GeLU(x)] = Φ(x) + x * Φ'(x)
    where Φ(x) is the Gaussian CDF and Φ'(x) is the Gaussian PDF.

    Args:
        x: Input tensor

    Returns:
        Derivative of GeLU at input x
    """
    # Using the same approximation as in gelu
    cdf_approx = 0.5 * (
        1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )

    # Derivative of the tanh approximation
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    inner_derivative = math.sqrt(2 / math.pi) * (1 + 0.044715 * 3 * torch.pow(x, 2))
    pdf_approx = 0.5 * inner_derivative * (1 - torch.pow(torch.tanh(inner), 2))

    # Combine the parts: Φ(x) + x * Φ'(x)
    return cdf_approx + x * pdf_approx


def silu_derivative(x):
    """
    Derivative of the SiLU (Swish) activation function with respect to its input.

    d/dx[SiLU(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

    Args:
        x: Input tensor

    Returns:
        Derivative of SiLU at input x
    """
    # Compute sigmoid(x)
    sigmoid_x = torch.sigmoid(x)

    # Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)


class MLPBackpropagator:
    """Handles backpropagation through MLP layers."""

    def __init__(
        self,
        model: HookedTransformer,
        logger: logging.Logger,
        ln_backpropagator: LayerNormBackpropagator,
    ):
        """
        Initialize the MLP backpropagator.

        Args:
            model: The transformer model
            logger: Logger for recording information
        """
        self.model = model
        self.logger = logger
        self.n_heads = model.cfg.n_heads
        self.n_layers = model.cfg.n_layers
        self.ln_backpropagator = ln_backpropagator

    def backward_no_gate_grad(
        self,
        layer_idx: int,
        mlp_out_grad: Float[torch.Tensor, "batch seq_len d_model"],
        activation_grad_approx: Literal["grad", "freeze"],
        activation_function: Literal["gelu", "silu"],
        layernorm_method: Literal["grad", "freeze"],
        cache_manager: CacheManager,
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Backpropagate the gradients through the MLP at a specific layer for all positions in parallel.

        Args:
            layer_idx: Index of the layer whose MLP we're backpropagating through
            activation_grad_approx: Method for approximating activation gradients
            ln_grad_approx: Method for approximating layer norm gradients
            abs_edge_strengths: Whether to take the absolute value of edge strengths
            cache_manager: Manager for accessing activation caches
            gradient_manager: Manager for handling gradients
        """

        if activation_grad_approx == "grad":
            if activation_function == "gelu":
                activation_approx = gelu_derivative(
                    cache_manager.get_block_hidden_state(layer_idx, "mlp.hook_pre")
                )
            elif activation_function == "silu":
                activation_approx = silu_derivative(
                    cache_manager.get_block_hidden_state(layer_idx, "mlp.hook_pre")
                )

        elif activation_grad_approx == "freeze":
            activation_approx = cache_manager.get_block_hidden_state(
                layer_idx, "mlp.hook_post"
            ) / cache_manager.get_block_hidden_state(layer_idx, "mlp.hook_pre")

        W_in: Float[torch.Tensor, "d_model d_mlp"] = self.model.blocks[
            layer_idx
        ].mlp.W_in
        W_out: Float[torch.Tensor, "d_mlp d_model"] = einsum(
            "d_mlp d_model, batch seq_len d_mlp -> batch seq_len d_mlp d_model",
            self.model.blocks[layer_idx].mlp.W_out,
            activation_approx,
        )
        mlp_mid_grad = einsum(
            "batch seq_len d_model, batch seq_len d_mlp d_model  -> batch seq_len d_mlp",
            mlp_out_grad,
            W_out,
        )
        mlp_grad_post_ln = einsum(
            "batch seq_len d_mlp, d_model d_mlp -> batch seq_len d_model",
            mlp_mid_grad,
            W_in,
        )

        mlp_grad = self.ln_backpropagator.backward(
            layer_idx=layer_idx,
            ln_out_grad=mlp_grad_post_ln,
            layernorm_method=layernorm_method,
            cache_manager=cache_manager,
            component_name="mlp",
        )

        return mlp_grad

    def backward_gated_grad(
        self,
        layer_idx: int,
        activation_grad_approx: Literal["grad", "freeze"],
        activation_function: Literal["gelu", "silu"],
        layernorm_method: Literal["grad", "freeze"],
        cache_manager: CacheManager,
        mlp_out_grad: Float[torch.Tensor, "batch seq_len d_model"],
        scale_mlp_gate: bool = False,
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Backpropagate the gradients through the MLP at a specific layer for all positions in parallel.

        Args:
            layer_idx: Index of the layer whose MLP we're backpropagating through
            activation_grad_approx: Method for approximating activation gradients
            ln_grad_approx: Method for approximating layer norm gradients
            abs_edge_strengths: Whether to take the absolute value of edge strengths
            cache_manager: Manager for accessing activation caches
            gradient_manager: Manager for handling gradients
        """

        mlp_gate_output = cache_manager.get_block_hidden_state(
            layer_idx, "mlp.hook_pre"
        )
        mlp_in_output = cache_manager.get_block_hidden_state(
            layer_idx, "mlp.hook_pre_linear"
        )

        if activation_grad_approx == "grad":
            if activation_function == "gelu":
                activation_grad = gelu_derivative(mlp_gate_output)
                activation_out = torch.nn.functional.gelu(mlp_gate_output)

            elif activation_function == "silu":
                activation_grad = silu_derivative(mlp_gate_output)
                activation_out = torch.nn.functional.silu(mlp_gate_output)

        elif activation_grad_approx == "freeze":
            activation_grad = (
                cache_manager.get_block_hidden_state(layer_idx, "mlp.hook_post")
                / mlp_gate_output
            )

            activation_out = torch.nn.functional.gelu(mlp_gate_output)

        W_in: Float[torch.Tensor, "d_model d_mlp"] = self.model.blocks[
            layer_idx
        ].mlp.W_in

        W_gate: Float[torch.Tensor, "d_model d_mlp"] = self.model.blocks[
            layer_idx
        ].mlp.W_gate

        mlp_mid_grad = einsum(
            "d_mlp d_model, batch seq_len d_model -> batch seq_len d_mlp",
            self.model.blocks[layer_idx].mlp.W_out,
            mlp_out_grad,
        )

        in_grad = einsum(
            "batch seq_len d_mlp, batch seq_len d_mlp -> batch seq_len d_mlp",
            mlp_mid_grad,
            activation_out,
        )
        gate_grad = einsum(
            "batch seq_len d_mlp, batch seq_len d_mlp -> batch seq_len d_mlp",
            mlp_mid_grad,
            activation_grad,
        )
        del mlp_mid_grad

        in_grad = einsum(
            "batch seq_len d_mlp, d_model d_mlp-> batch seq_len d_model",
            in_grad,
            W_in,
        )
        gate_grad = einsum(
            "batch seq_len d_mlp, batch seq_len d_mlp -> batch seq_len d_mlp",
            gate_grad,
            mlp_in_output,
        )
        gate_grad = einsum(
            "batch seq_len d_mlp, d_model d_mlp -> batch seq_len d_model",
            gate_grad,
            W_gate,
        )

        mlp_grad = gate_grad + in_grad

        mlp_grad = self.ln_backpropagator.backward(
            layer_idx=layer_idx,
            ln_out_grad=mlp_grad,
            layernorm_method=layernorm_method,
            cache_manager=cache_manager,
            component_name="mlp",
        )

        if scale_mlp_gate:
            mlp_grad = mlp_grad / 2

        return mlp_grad

    def backward_grad(
        self,
        layer_idx: int,
        activation_grad_approx: Literal["grad", "freeze"],
        layernorm_method: Literal["grad", "freeze"],
        mlp_out_grad: Float[torch.Tensor, "batch seq_len d_model"],
        cache_manager: CacheManager,
        gradient_manager: GradientManager,
        scale_mlp_gate: bool,
    ) -> None:
        """Backpropagate the gradients through the MLP at a specific layer for all positions in parallel.

        Args:
            layer_idx: Index of the layer whose MLP we're backpropagating through
            activation_grad_approx: Method for approximating activation gradients
            ln_grad_approx: Method for approximating layer norm gradients
            abs_edge_strengths: Whether to take the absolute value of edge strengths
            cache_manager: Manager for accessing activation caches
            gradient_manager: Manager for handling gradients
        """
        # module_name = f"blocks.{layer_idx}.hook_mlp_in"
        activation_function = (
            "gelu" if "gelu" in self.model.cfg.act_fn else self.model.cfg.act_fn
        )
        gated = self.model.cfg.gated_mlp

        if gated:
            mlp_grad = self.backward_gated_grad(
                layer_idx=layer_idx,
                activation_grad_approx=activation_grad_approx,
                activation_function=activation_function,
                layernorm_method=layernorm_method,
                cache_manager=cache_manager,
                mlp_out_grad=mlp_out_grad,
                scale_mlp_gate=scale_mlp_gate,
            )
        else:
            mlp_grad = self.backward_no_gate_grad(
                layer_idx=layer_idx,
                mlp_out_grad=mlp_out_grad,
                activation_grad_approx=activation_grad_approx,
                activation_function=activation_function,
                layernorm_method=layernorm_method,
                cache_manager=cache_manager,
            )

        mlp_grad = einops.repeat(
            mlp_grad,
            "batch seq_len d_model -> batch 1 seq_len d_model",
        )

        gradient_manager.mlp_and_token_grad[:, : layer_idx + 1] += mlp_grad

        attn_grads = einsum(
            "batch n_layers seq_len d_model, n_layers n_heads d_head d_model -> batch n_layers n_heads seq_len d_head",
            mlp_grad,
            self.model.W_O[: layer_idx + 1],
        )
        gradient_manager.attention_head_grad[:, : layer_idx + 1] += attn_grads

        if gradient_manager.return_prune_scores:
            (
                mlp_and_token_hidden_states_diff,
                attn_head_hidden_states_diff,
            ) = cache_manager.get_activation_diffs_from_previous_layers(
                layer_idx=layer_idx, component_type="mlp"
            )

            gradient_manager.add_edges_to_prune_scores(
                module_name=f"blocks.{layer_idx}.hook_mlp_in",
                mlp_token_grad=mlp_grad,
                mlp_token_hidden_states=mlp_and_token_hidden_states_diff,
                attn_grad=attn_grads,
                attn_hidden_states=attn_head_hidden_states_diff,
            )

    def backward(
        self,
        layer_idx: int,
        activation_grad_approx: Literal["grad", "freeze"],
        layernorm_method: Literal["grad", "freeze"],
        cache_manager: CacheManager,
        gradient_manager: GradientManager,
        scale_mlp_gate: bool,
    ) -> None:
        """Backpropagate the gradients through the MLP at a specific layer for all positions in parallel.
        Args:
            layer_idx: Index of the layer whose MLP we're backpropagating through
            activation_grad_approx: Method for approximating activation gradients
            ln_grad_approx: Method for approximating layer norm gradients
            abs_edge_strengths: Whether to take the absolute value of edge strengths
            cache_manager: Manager for accessing activation caches
            gradient_manager: Manager for handling gradients
        """

        mlp_out_grad = gradient_manager.get_mlp_gradient(layer_idx=layer_idx)

        if self.model.cfg.use_normalization_before_and_after:
            ln_scale: Float[torch.Tensor, "batch seq_len d_model"] = (
                cache_manager.get_block_hidden_state(layer_idx, "ln2_post.hook_scale")
            )
            mlp_out_grad = (
                mlp_out_grad / ln_scale * self.model.blocks[layer_idx].ln2_post.w
            )

        self.backward_grad(
            layer_idx=layer_idx,
            activation_grad_approx="grad"
            if activation_grad_approx == "grad"
            else "freeze",
            layernorm_method=layernorm_method,
            mlp_out_grad=mlp_out_grad,
            cache_manager=cache_manager,
            gradient_manager=gradient_manager,
            scale_mlp_gate=scale_mlp_gate,
        )
