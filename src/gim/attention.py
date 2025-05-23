# ruff: noqa: F722
# mypy: disable-error-code="name-defined"
import logging
from typing import Literal

import torch
from jaxtyping import Float
import einops
from fancy_einsum import einsum
from transformer_lens import HookedTransformer

from src.gim.utils import (
    CacheManager,
    GradientManager,
    reverse_apply_rotary,
    softmax_derivation,
    softmax_derivation_atpstar,
    softmax_tsg,
)
from src.gim.layernorm import LayerNormBackpropagator


class AttentionBackpropagator:
    """Handles backpropagation through attention mechanisms."""

    def __init__(
        self,
        model: HookedTransformer,
        logger: logging.Logger,
        ln_backpropagator: LayerNormBackpropagator,
    ):
        """
        Initialize the attention backpropagator.

        Args:
            model: The transformer model
            logger: Logger for recording information
            ln_backpropagator: Helper for layer norm backpropagation
        """
        self.model = model
        self.logger = logger
        self.ln_backpropagator = ln_backpropagator
        self.n_heads = model.cfg.n_heads
        self.n_layers = model.cfg.n_layers
        self.attn_scale = model.cfg.attn_scale
        self.rotary = model.cfg.positional_embedding_type == "rotary"
        self.attn_scores_soft_cap = model.cfg.attn_scores_soft_cap

    def _get_rotary_params(self, layer_idx: int) -> tuple:
        """Get rotary embedding parameters for a specific layer."""
        rotary_sin = self.model.blocks[layer_idx].attn.rotary_sin
        rotary_cos = self.model.blocks[layer_idx].attn.rotary_cos
        rotary_dim = self.model.cfg.rotary_dim
        rotary_adjacent_pairs = self.model.cfg.rotary_adjacent_pairs
        return rotary_sin, rotary_cos, rotary_dim, rotary_adjacent_pairs

    def _calculate_value_contribution(
        self,
        layer_idx: int,
        grad: Float[torch.Tensor, "batch n_heads query_pos d_head"],
        cache_manager: CacheManager,
    ) -> Float[torch.Tensor, "batch n_heads query_pos key_pos"]:
        """
        Calculate how much each value vector contributes to the attention layer's output gradient.

        Args:
            layer_idx: Index of the transformer block
            grad: Gradient of the model output with respect to the attention layer's output
            cache_manager: Manager for accessing activation caches

        Returns:
            Contribution score of each value vector to the output gradient
        """
        # Get value vectors for all relevant positions
        value_vectors = cache_manager.get_block_hidden_state(layer_idx, "attn.hook_v")

        # Measure alignment with attention head output gradient vector
        return einsum(
            "batch key_pos n_heads d_head, batch n_heads query_pos d_head -> batch n_heads query_pos key_pos",
            value_vectors,
            grad,
        )

    def _grad_normalize(
        self,
        component: Literal["key", "query", "value"],
        grad: Float[torch.Tensor, "batch n_heads query_pos d_head"],
    ):
        """
        Normalize the gradient for a specific component (key, query, or value).

        Args:
            component: The component to normalize
            grad: The gradient tensor
            norm_strategy: Normalization strategy ('attnlrp' or 'three')

        Returns:
            Normalized gradient tensor
        """

        if component == "key":
            return grad / 4
        elif component == "query":
            return grad / 4
        elif component == "value":
            return grad / 2
        else:
            raise ValueError(
                f"Unknown component: {component}. Use 'key', 'query', or 'value'."
            )

    def backward_value(
        self,
        attn_out_grad: Float[torch.Tensor, "batch n_heads query_pos d_head"],
        layer_idx: int,
        layernorm_method: Literal["grad", "freeze"],
        cache_manager: CacheManager,
    ) -> Float[torch.Tensor, "batch n_heads key_pos d_model"]:
        """
        Compute gradient for the value component of attention.

        Args:
            attn_out_grad: Gradient of the model output with respect to attention output
            layer_idx: Index of the layer whose attention we're backpropagating through
            layernorm_method: Method for handling layernorm gradients ('grad' or 'freeze')
            cache_manager: Manager for accessing activation caches

        Returns:
            Gradient for the value component
        """
        # Get attention pattern
        attention_pattern = cache_manager.get_block_hidden_state(
            layer_idx, "attn.hook_pattern"
        )

        self.logger.debug(
            f"Attention pattern shape: {attention_pattern.shape}, Attention gradient shape: {attn_out_grad.shape}"
        )

        # Project gradient to input space through value matrix
        value_in_grad = einsum(
            "n_heads d_model d_head, batch n_heads query_pos d_head -> batch n_heads query_pos d_model",
            self.model.W_V[layer_idx],
            attn_out_grad,
        )

        # Apply attention pattern to distribute gradient across positions
        value_grad_post_ln = einsum(
            "batch n_heads query_pos d_model, batch n_heads query_pos key_pos -> batch n_heads key_pos d_model",
            value_in_grad,
            attention_pattern,
        )

        # Apply layer normalization gradient
        return self.ln_backpropagator.backward(
            layer_idx=layer_idx,
            ln_out_grad=value_grad_post_ln,
            layernorm_method=layernorm_method,
            cache_manager=cache_manager,
            component_name="attention",
        )

    def backward_key_grad(
        self,
        attn_out_grad: Float[torch.Tensor, "batch n_heads query_pos d_head"],
        layer_idx: int,
        layernorm_method: Literal["grad", "freeze"],
        softmax_method: Literal[
            "grad",
            "atpstar",
            "tsg",
        ],
        tsg_temperature: float,
        cache_manager: CacheManager,
    ) -> Float[torch.Tensor, "batch n_heads key_pos d_model"]:
        """
        Compute how changes in key vectors affect the model output using gradient-based approach.

        Args:
            attn_out_grad: Gradient with respect to attention output
            layer_idx: Index of the layer
            layernorm_method: Method for handling layernorm gradients
            cache_manager: Manager for accessing activation caches

        Returns:
            Gradient for the key component
        """
        # Get vectors needed for computation
        hook_name = "attn.hook_rot_q" if self.rotary else "attn.hook_q"
        query_vectors = cache_manager.get_block_hidden_state(layer_idx, hook_name)

        # Get value contribution and attention pattern
        value_contribution = self._calculate_value_contribution(
            layer_idx, attn_out_grad, cache_manager
        )
        attention_pattern = cache_manager.get_block_hidden_state(
            layer_idx, "attn.hook_pattern"
        )

        # Compute softmax gradient
        if softmax_method == "grad":
            softmax_grad = softmax_derivation(attention_pattern, value_contribution)
        elif softmax_method == "atpstar":
            corrupted_attention_pattern = cache_manager.get_block_corrupt_hidden_state(
                layer_idx, "attn.hook_pattern"
            )
            softmax_grad = softmax_derivation_atpstar(
                attention_pattern, corrupted_attention_pattern, value_contribution
            )
        elif softmax_method == "tsg":
            attention_scores = cache_manager.get_block_hidden_state(
                layer_idx, "attn.hook_attn_scores"
            )
            softmax_grad = softmax_tsg(
                attention_scores, value_contribution, temperature=tsg_temperature
            )
        else:
            raise ValueError(
                f"Unknown softmax method: {softmax_method}. Expected 'grad', 'direct', or 'importance_weight'."
            )

        # Apply gradient to compute key gradient
        key_grad = einsum(
            "batch query_pos n_heads d_head, batch n_heads query_pos key_pos -> batch key_pos n_heads d_head",
            query_vectors,
            softmax_grad,
        )

        # Apply rotary embedding if needed
        if self.rotary:
            (
                rotary_sin,
                rotary_cos,
                rotary_dim,
                rotary_adjacent_pairs,
            ) = self._get_rotary_params(layer_idx)
            key_grad = reverse_apply_rotary(
                key_grad, rotary_sin, rotary_cos, rotary_dim, rotary_adjacent_pairs
            )

        # Project through weight matrix
        key_grad = (
            einsum(
                "batch key_pos n_heads d_head, n_heads d_model d_head -> batch n_heads key_pos d_model",
                key_grad,
                self.model.W_K[layer_idx],
            )
            / self.attn_scale
        )

        # Apply layer normalization gradient
        return self.ln_backpropagator.backward(
            layer_idx=layer_idx,
            ln_out_grad=key_grad,
            layernorm_method=layernorm_method,
            cache_manager=cache_manager,
            component_name="attention",
        )

    def backward_query_grad(
        self,
        attn_out_grad: Float[torch.Tensor, "batch n_heads query_pos d_head"],
        layer_idx: int,
        layernorm_method: Literal["grad", "freeze"],
        softmax_method: Literal[
            "grad",
            "atpstar",
            "tsg",
        ],
        tsg_temperature: float,
        cache_manager: CacheManager,
    ) -> Float[torch.Tensor, "batch n_heads n_previous_components key_pos d_model"]:
        """
        Compute how changes in query vectors affect the model output using gradient-based approach.

        Args:
            attn_out_grad: Gradient with respect to attention output
            layer_idx: Index of the layer
            layernorm_method: Method for handling layernorm gradients
            cache_manager: Manager for accessing activation caches

        Returns:
            Gradient for the query component
        """
        # Get vectors needed for computation
        hook_name = "attn.hook_rot_k" if self.rotary else "attn.hook_k"
        key_vectors = cache_manager.get_block_hidden_state(layer_idx, hook_name)

        # Get value contribution and attention pattern
        value_contribution = self._calculate_value_contribution(
            layer_idx, attn_out_grad, cache_manager
        )
        attention_pattern = cache_manager.get_block_hidden_state(
            layer_idx, "attn.hook_pattern"
        )

        # Compute softmax gradient
        if softmax_method == "grad":
            softmax_grad = softmax_derivation(attention_pattern, value_contribution)
        elif softmax_method == "atpstar":
            corrupted_attention_pattern = cache_manager.get_block_corrupt_hidden_state(
                layer_idx, "attn.hook_pattern"
            )
            softmax_grad = softmax_derivation_atpstar(
                attention_pattern, corrupted_attention_pattern, value_contribution
            )
        elif softmax_method == "tsg":
            attention_scores = cache_manager.get_block_hidden_state(
                layer_idx, "attn.hook_attn_scores"
            )
            softmax_grad = softmax_tsg(
                attention_scores, value_contribution, temperature=tsg_temperature
            )
        else:
            raise ValueError(
                f"Unknown softmax method: {softmax_method}. Expected 'grad', 'direct', or 'importance_weight'."
            )

        # Apply gradient to compute query gradient
        query_grad = einsum(
            "batch key_pos n_heads d_head, batch n_heads query_pos key_pos -> batch query_pos n_heads d_head",
            key_vectors,
            softmax_grad,
        )

        # Apply rotary embedding if needed
        if self.rotary:
            (
                rotary_sin,
                rotary_cos,
                rotary_dim,
                rotary_adjacent_pairs,
            ) = self._get_rotary_params(layer_idx)
            query_grad = reverse_apply_rotary(
                query_grad, rotary_sin, rotary_cos, rotary_dim, rotary_adjacent_pairs
            )

        # Project through weight matrix
        query_grad = (
            einsum(
                "batch query_pos n_heads d_head, n_heads d_model d_head -> batch n_heads query_pos d_model",
                query_grad,
                self.model.W_Q[layer_idx],
            )
            / self.attn_scale
        )

        # Apply layer normalization gradient
        return self.ln_backpropagator.backward(
            layer_idx=layer_idx,
            ln_out_grad=query_grad,
            layernorm_method=layernorm_method,
            cache_manager=cache_manager,
            component_name="attention",
        )

    def add_grad_to_previous_layers(
        self,
        layer_idx: int,
        gradient_manager: GradientManager,
        cache_manager: CacheManager,
        grad: Float[torch.Tensor, "batch n_heads seq_len d_model"],
        module_name: str,
    ):
        """
        Add computed gradients to previous layers' gradients.

        Args:
            layer_idx: Current layer index
            gradient_manager: Manager for handling gradients
            grad: Computed gradient to add
        """
        torch.cuda.empty_cache()
        grad = einops.rearrange(
            grad,
            "batch n_heads_out seq_len d_model -> batch n_heads_out 1 seq_len d_model",
        )
        # Add to MLP and token gradients for all previous layers
        gradient_manager.mlp_and_token_grad[:, : layer_idx + 1] += grad.sum(1)

        if not gradient_manager.return_prune_scores:
            grad = grad.sum(1, keepdim=True)

        # Compute gradients for attention heads in previous layers
        attn_grads = einsum(
            "batch n_heads_out n_layers_in seq_len d_model, n_layers_in n_heads_in d_head d_model -> batch n_heads_out n_layers_in n_heads_in seq_len d_head",
            grad,
            self.model.W_O[:layer_idx],
        )

        # Add to attention head gradients for previous layers
        gradient_manager.attention_head_grad[:, :layer_idx] += attn_grads.sum(1)

        if gradient_manager.return_prune_scores:
            (
                mlp_and_token_hidden_states_diff,
                attn_head_hidden_states_diff,
            ) = cache_manager.get_activation_diffs_from_previous_layers(
                layer_idx=layer_idx, component_type="attention"
            )

            gradient_manager.add_edges_to_prune_scores(
                module_name=module_name,
                mlp_token_grad=grad,
                mlp_token_hidden_states=mlp_and_token_hidden_states_diff,
                attn_grad=attn_grads,
                attn_hidden_states=attn_head_hidden_states_diff,
            )

    def backward_kq_grad(
        self,
        attn_out_grad: Float[torch.Tensor, "batch n_heads query_pos d_head"],
        softmax_method: Literal[
            "grad",
            "atpstar",
            "tsg",
        ],
        layer_idx: int,
        layernorm_method: Literal["grad", "freeze"],
        query_value_key_divide_grad_strategy: bool,
        tsg_temperature: float,
        cache_manager: CacheManager,
        gradient_manager: GradientManager,
    ):
        """
        Compute and add query and key gradients using gradient-based approach.

        Args:
            attn_out_grad: Gradient with respect to attention output
            layer_idx: Layer index
            layernorm_method: Method for handling layernorm gradients
            abs_edge_strength: Whether to use absolute edge strengths
            query_value_key_divide_grad: Whether to divide gradients
            cache_manager: Manager for accessing activation caches
            gradient_manager: Manager for handling gradients
        """
        # Process query gradients
        query_grad = self.backward_query_grad(
            attn_out_grad=attn_out_grad,
            layer_idx=layer_idx,
            layernorm_method=layernorm_method,
            cache_manager=cache_manager,
            softmax_method=softmax_method,
            tsg_temperature=tsg_temperature,
        )

        torch.cuda.empty_cache()

        # Apply gradient scaling if needed
        if query_value_key_divide_grad_strategy:
            query_grad = self._grad_normalize("query", query_grad)

        # Add query gradients to previous layers
        self.add_grad_to_previous_layers(
            layer_idx=layer_idx,
            gradient_manager=gradient_manager,
            cache_manager=cache_manager,
            grad=query_grad,
            module_name=f"blocks.{layer_idx}.hook_q_input",
        )

        # Process key gradients
        key_grad = self.backward_key_grad(
            attn_out_grad=attn_out_grad,
            layer_idx=layer_idx,
            layernorm_method=layernorm_method,
            cache_manager=cache_manager,
            softmax_method=softmax_method,
            tsg_temperature=tsg_temperature,
        )

        torch.cuda.empty_cache()

        # Apply gradient scaling if needed
        if query_value_key_divide_grad_strategy:
            key_grad = self._grad_normalize("key", key_grad)

        # Add key gradients to previous layers
        self.add_grad_to_previous_layers(
            layer_idx=layer_idx,
            gradient_manager=gradient_manager,
            cache_manager=cache_manager,
            grad=key_grad,
            module_name=f"blocks.{layer_idx}.hook_k_input",
        )

    def backward(
        self,
        layer_idx: int,
        value_only: bool,
        softmax_method: Literal[
            "grad",
            "atpstar",
            "tsg",
        ],
        layernorm_method: Literal["grad", "freeze"],
        query_value_key_divide_grad_strategy: bool,
        tsg_temperature: float,
        cache_manager: CacheManager,
        gradient_manager: GradientManager,
    ):
        """
        Main backpropagation method for attention mechanism.

        Args:
            layer_idx: Index of the layer to backpropagate through
            value_only: If True, only compute value gradients
            softmax_method: Method for approximating softmax gradients
            layernorm_method: Method for handling layernorm gradients
            abs_edge_strength: Whether to use absolute value for edge strength
            query_value_key_divide_grad: Whether to divide gradients for each component
            cache_manager: Manager for accessing activation caches
            gradient_manager: Manager for handling gradients
            attn_comp_batch_size: Batch size for component-wise computation
        """
        # Get gradient for attention output
        attn_out_grad = gradient_manager.get_attn_gradient(layer_idx=layer_idx)

        if self.model.cfg.use_normalization_before_and_after:
            ln_scale: Float[torch.Tensor, "batch key_pos d_model"] = (
                cache_manager.get_block_hidden_state(layer_idx, "ln1_post.hook_scale")
            )

            attn_out_grad = einsum(
                "batch n_heads seq_len d_head,  n_head d_head d_model -> batch n_heads seq_len d_model",
                attn_out_grad,
                self.model.W_O[layer_idx],
            )
            attn_out_grad = (
                attn_out_grad / ln_scale * self.model.blocks[layer_idx].ln1_post.w
            )
            attn_out_grad = einsum(
                "batch n_heads seq_len d_model,  n_head d_head d_model -> batch n_heads seq_len d_head",
                attn_out_grad,
                self.model.W_O[layer_idx],
            )

        # Process value component
        value_grad = self.backward_value(
            attn_out_grad, layer_idx, layernorm_method, cache_manager
        )

        if query_value_key_divide_grad_strategy:
            value_grad = self._grad_normalize("value", value_grad)

        # Add value gradient to previous layers
        self.add_grad_to_previous_layers(
            layer_idx=layer_idx,
            gradient_manager=gradient_manager,
            cache_manager=cache_manager,
            grad=value_grad,
            module_name=f"blocks.{layer_idx}.hook_v_input",
        )

        torch.cuda.empty_cache()

        if value_only:
            return

        self.backward_kq_grad(
            attn_out_grad=attn_out_grad,
            layer_idx=layer_idx,
            softmax_method=softmax_method,
            layernorm_method=layernorm_method,
            query_value_key_divide_grad_strategy=query_value_key_divide_grad_strategy,
            tsg_temperature=tsg_temperature,
            cache_manager=cache_manager,
            gradient_manager=gradient_manager,
        )
