# ruff: noqa: F722
# mypy: disable-error-code="name-defined"
from typing import Optional, Literal
import logging

import torch
import einops
from fancy_einsum import einsum
from jaxtyping import Float

from transformer_lens import ActivationCache, HookedTransformer


def safe_division(tensor: torch.Tensor, min_value: float = 1e-11) -> torch.Tensor:
    """Safely handle division by ensuring no zeros in denominator."""
    tensor = tensor + torch.sign(tensor) * min_value
    tensor[tensor == 0] = min_value
    return tensor


def rotate_every_two(
    x: Float[torch.Tensor, "... rotary_dim"], rotary_adjacent_pairs: bool = False
) -> Float[torch.Tensor, "... rotary_dim"]:
    """
    Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

    The final axis of x must have even length.

    GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
    """
    rot_x = x.clone()
    if rotary_adjacent_pairs:
        rot_x[..., ::2] = -x[..., 1::2]
        rot_x[..., 1::2] = x[..., ::2]
    else:
        n = x.size(-1) // 2
        rot_x[..., :n] = -x[..., n:]
        rot_x[..., n:] = x[..., :n]

    return rot_x


def apply_rotary(
    x: Float[torch.Tensor, "batch pos head_index d_head"],
    rotary_sin: Float[torch.Tensor, "max_len rotary_dim"],
    rotary_cos: Float[torch.Tensor, "max_len rotary_dim"],
    rotary_dim: int,
    rotary_adjacent_pairs: bool = False,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)

    if x.device != rotary_sin.device:
        x = x.to(rotary_sin.device)

    x_pos = x.size(1)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_flip = rotate_every_two(x_rot, rotary_adjacent_pairs)

    rotary_cos = rotary_cos[None, :x_pos, None, :]
    rotary_sin = rotary_sin[None, :x_pos, None, :]
    x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
    return torch.cat([x_rotated, x_pass], dim=-1)


def reverse_apply_rotary(
    x_rotated: Float[torch.Tensor, "batch pos head_index d_head"],
    rotary_sin: Float[torch.Tensor, "max_len rotary_dim"],
    rotary_cos: Float[torch.Tensor, "max_len rotary_dim"],
    rotary_dim: int,
    rotary_adjacent_pairs: bool = False,
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """
    Reverse the rotary positional embedding that was applied using apply_rotary.

    Args:
        x_rotated: Tensor with rotary embedding applied
        rotary_sin: Sine component of rotary embedding
        rotary_cos: Cosine component of rotary embedding
        rotary_dim: Number of dimensions that had rotary applied
        rotary_adjacent_pairs: Whether rotary was applied to adjacent pairs

    Returns:
        Tensor with rotary embedding removed
    """
    if x_rotated.device != rotary_sin.device:
        x_rotated = x_rotated.to(rotary_sin.device)

    x_pos = x_rotated.size(1)
    x_rot_result = x_rotated[..., :rotary_dim]
    x_pass = x_rotated[..., rotary_dim:]
    rotary_cos = rotary_cos[None, :x_pos, None, :]
    rotary_sin = rotary_sin[None, :x_pos, None, :]

    x_flip_rotated = rotate_every_two(x_rot_result, rotary_adjacent_pairs)
    x_rot = x_rot_result * rotary_cos - x_flip_rotated * rotary_sin

    # Reassemble the tensor
    return torch.cat([x_rot, x_pass], dim=-1)


def softmax_derivation(
    softmax_output: Float[torch.Tensor, "batch n_heads query_pos key_pos"],
    grad_output: Float[torch.Tensor, "batch n_heads query_pos key_pos"],
) -> Float[torch.Tensor, "batch n_heads query_pos key_pos"]:
    """
    Compute gradient of softmax function efficiently without explicit Jacobian computation.

    Args:
        softmax_output: Output of softmax function
        grad_output: Gradient of loss with respect to softmax output

    Returns:
        Gradient of loss with respect to softmax input
    """
    # Sum of product for each example in batch
    sumdot = torch.sum(softmax_output * grad_output, dim=-1, keepdim=True)

    # Compute gradient in a vectorized way
    return softmax_output * (grad_output - sumdot)


def softmax_tsg(
    attention_scores: Float[torch.Tensor, "batch n_heads query_pos key_pos"],
    grad_output: Float[torch.Tensor, "batch n_heads query_pos key_pos"],
    temperature: float = 2,
) -> Float[torch.Tensor, "batch n_heads query_pos key_pos"]:
    # Compute Integrated Gradients for the softmax function
    attention_pattern = torch.softmax(attention_scores / temperature, dim=-1)
    return softmax_derivation(attention_pattern, grad_output)


def softmax_derivation_atpstar(
    attention_pattern: Float[torch.Tensor, "batch n_heads query_pos key_pos"],
    corrupt_attention_pattern: Float[torch.Tensor, "batch n_heads query_pos key_pos"],
    grad_output: Float[torch.Tensor, "batch n_heads query_pos key_pos"],
) -> Float[torch.Tensor, "batch n_heads query_pos key_pos"]:
    """
    Compute gradient of softmax function efficiently without explicit Jacobian computation.

    Args:
        softmax_output: Output of softmax function
        grad_output: Gradient of loss with respect to softmax output

    Returns:
        Gradient of loss with respect to softmax input
    """
    softmax_grad = attention_pattern - corrupt_attention_pattern
    return softmax_grad * grad_output


class CacheManager:
    """Manages activation caches for clean and corrupted states."""

    clean_cache: ActivationCache

    # Mapping to understand the component indices
    # Maps (type, layer, head) -> component_index and component_index -> (type, layer, head)
    component_to_idx: dict[tuple[str, int, Optional[int]], int]
    idx_to_component: dict[int, tuple[str, int, Optional[int]]]

    def __init__(
        self,
        model: HookedTransformer,
        clean_cache: ActivationCache,
        logger: logging.Logger,
        corrupted_cache: Optional[ActivationCache] = None,
    ):
        self.model = model
        self.logger = logger
        self.clean_cache: ActivationCache = clean_cache

        # Calculate the total number of components
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.n_components = 1 + self.n_layers * (
            self.n_heads + 1
        )  # token_embedding + (heads + MLP) per layer
        self.batch_size = next(iter(clean_cache.values())).shape[0]
        self.seq_len = next(iter(clean_cache.values())).shape[1]
        self.device = next(iter(clean_cache.values())).device
        self.dtype = next(iter(clean_cache.values())).dtype
        self.d_model = model.cfg.d_model
        self.d_head = model.cfg.d_head
        # Create component mapping for easier reference
        self.create_component_mapping()

        # Process clean cache
        self.set_clean_hidden_states(clean_cache)

        # Process corrupted cache (or set to zeros)
        if corrupted_cache is None:
            self.logger.info("Corrupted cache is None")
            # self.set_corrupted_hidden_states_to_zeros()
        else:
            self.set_corrupted_hidden_states(corrupted_cache)
            del corrupted_cache
            torch.cuda.empty_cache()

    def create_component_mapping(self) -> None:
        """Create a bidirectional mapping between component tuples and indices.

        This ensures that indices are assigned in a specific order:
        1. Token embedding
        2. All heads for layer 0
        3. MLP for layer 0
        4. All heads for layer 1
        5. MLP for layer 1
        6. And so on...

        This sequential assignment guarantees that all heads for a given layer
        have consecutive indices, which allows efficient slicing.
        """
        self.component_to_idx = {}
        self.idx_to_component = {}

        idx = 0

        # Token embedding
        self.component_to_idx[("token_embedding", 0, None)] = idx
        self.idx_to_component[idx] = ("token_embedding", 0, None)
        idx += 1

        # For each layer
        for layer in range(self.n_layers):
            # Attention heads (consecutive indices)
            for head in range(self.n_heads):
                self.component_to_idx[("attention", layer, head)] = idx
                self.idx_to_component[idx] = ("attention", layer, head)
                idx += 1

            # MLP
            self.component_to_idx[("mlp", layer, None)] = idx
            self.idx_to_component[idx] = ("mlp", layer, None)
            idx += 1

    def get_attention_head_indices(self, layer_idx: int) -> tuple[int, int]:
        """Get the start and end indices for all attention heads in a layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Tuple of (start_idx, end_idx) where end_idx is exclusive
        """
        start_idx = self.get_component_index("attention", layer_idx, 0)
        end_idx = start_idx + self.n_heads
        return start_idx, end_idx

    def _extract_hidden_states_from_activation_cache(
        self, cache: ActivationCache
    ) -> Float[torch.Tensor, "batch n_components seq_len d_model"]:
        """Extract hidden states from the activation cache into unified format.

        Args:
            cache: Activation cache from model forward pass

        Returns:
            Tensor containing all components in order [token_embed, L0H0, L0H1, ..., L0MLP, L1H0, ...]
        """
        mlp_and_token_hidden_states = torch.zeros(
            (self.batch_size, self.n_layers + 1, self.seq_len, self.d_model),
            device=self.device,
            dtype=self.dtype,
        )
        components = cache.decompose_resid(return_labels=False, mode="mlp")

        token_embeddings = components[: -self.n_layers]
        token_embeddings = einops.reduce(
            token_embeddings,
            "n_comp batch seq_len d_model -> batch seq_len d_model",
            "sum",
        )
        mlp_and_token_hidden_states[:, 0] = token_embeddings

        mlp_hidden_states = components[-self.n_layers :]
        mlp_hidden_states -= einops.repeat(
            self.model.b_out, "n_layers d_model -> n_layers 1 1 d_model"
        )
        mlp_hidden_states = einops.rearrange(
            mlp_hidden_states,
            "n_layers batch seq_len d_model -> batch n_layers seq_len d_model",
        )
        mlp_and_token_hidden_states[:, 1:] = mlp_hidden_states

        attention_head_hidden_states = cache.stack_activation("z")
        attention_head_hidden_states = einops.rearrange(
            attention_head_hidden_states,
            "n_layers batch seq_len n_heads d_head -> batch n_layers n_heads seq_len d_head",
        )
        return mlp_and_token_hidden_states, attention_head_hidden_states


    def set_clean_hidden_states(self, cache: ActivationCache) -> None:
        """Set the clean hidden states from activation cache."""
        self.logger.info("Setting clean hidden states")
        (
            self.clean_mlp_and_token,
            self.clean_attn,
        ) = self._extract_hidden_states_from_activation_cache(cache)
        self.clean_cache = cache

    def set_corrupted_hidden_states(self, cache: ActivationCache) -> None:
        """Set the corrupted hidden states from activation cache."""
        self.logger.info("Setting corrupted hidden states")
        (
            self.corrupted_mlp_and_token,
            self.corrupted_attn,
        ) = self._extract_hidden_states_from_activation_cache(cache)
        self.corrupt_cache = cache
        # self.corrupted_attention_scores = [cache[f"blocks.{layer_idx}.attn.hook_attn_scores"] for layer_idx in range(self.n_layers)]
        # self.corrupted_attention_pattern = [cache[f"blocks.{layer_idx}.attn.hook_pattern"] for layer_idx in range(self.n_layers)]

    def set_corrupted_hidden_states_to_zeros(self) -> None:
        """Set corrupted hidden states to zeros (for ablation studies)."""
        self.logger.info("Setting corrupted hidden states to zeros")
        self.corrupted_mlp_and_token = torch.zeros_like(self.clean_mlp_and_token)
        self.corrupted_attn = torch.zeros_like(self.clean_attn)
        self.corrupt_cache = None

    def get_block_hidden_state(
        self, layer_idx: int, hidden_state_name: str
    ) -> torch.Tensor:
        """Get a specific hidden state from the clean cache."""
        if self.clean_cache is None:
            raise ValueError("Clean cache is not set.")

        return self.clean_cache[f"blocks.{layer_idx}.{hidden_state_name}"]

    def get_block_corrupt_hidden_state(
        self, layer_idx: int, hidden_state_name: str
    ) -> torch.Tensor:
        """Get a specific hidden state from the clean cache."""
        if self.corrupt_cache is None:
            raise ValueError("Corrupt cache is not set.")

        return self.corrupt_cache[f"blocks.{layer_idx}.{hidden_state_name}"]


    def get_activations_from_previous_layers(
        self, layer_idx: int, component_type: Literal["attention", "mlp"]
    ) -> Float[torch.Tensor, "batch n_previous_components seq_len d_model"]:
        """Get activation values from all layers preceding the specified layer.

        Args:
            layer_idx: Index of the current layer
            component_type: Whether we're processing attention or MLP

        Returns:
            Tensor of all activations from previous layers/components
        """
        # Calculate the index in the component dimension up to which we need to include
        if component_type == "attention":
            # Include everything up to the end of the previous layer
            return self.clean_mlp_and_token[:, : layer_idx + 1], self.clean_attn[
                :, :layer_idx
            ]

        return self.clean_mlp_and_token[:, : layer_idx + 1], self.clean_attn[
            :, : layer_idx + 1
        ]

    def get_activation_diffs_from_previous_layers(
        self, layer_idx: int, component_type: Literal["attention", "mlp"]
    ) -> Float[torch.Tensor, "batch n_previous_components seq_len d_model"]:
        """Get activation values from all layers preceding the specified layer.

        Args:
            layer_idx: Index of the current layer
            component_type: Whether we're processing attention or MLP

        Returns:
            Tensor of all activations from previous layers/components
        """
        # Calculate the index in the component dimension up to which we need to include
        if component_type == "attention":
            # Include everything up to the end of the previous layer
            return self.clean_mlp_and_token[
                :, : layer_idx + 1
            ] - self.corrupted_mlp_and_token[:, : layer_idx + 1], self.clean_attn[
                :, :layer_idx
            ] - self.corrupted_attn[:, :layer_idx]

        return self.clean_mlp_and_token[
            :, : layer_idx + 1
        ] - self.corrupted_mlp_and_token[:, : layer_idx + 1], self.clean_attn[
            :, : layer_idx + 1
        ] - self.corrupted_attn[:, : layer_idx + 1]

    def get_corrupted_activations_from_previous_layers(
        self, layer_idx: int, component_type: Literal["attention", "mlp"]
    ) -> Float[torch.Tensor, "batch n_previous_components seq_len d_model"]:
        """Get activation values from all layers preceding the specified layer.

        Args:
            layer_idx: Index of the current layer
            component_type: Whether we're processing attention or MLP

        Returns:
            Tensor of all activations from previous layers/components
        """
        # Calculate the index in the component dimension up to which we need to include
        if component_type == "attention":
            # Include everything up to the end of the previous layer
            return self.corrupted_mlp_and_token[
                :, : layer_idx + 1
            ], self.corrupted_attn[:, :layer_idx]

        return self.corrupted_mlp_and_token[:, : layer_idx + 1], self.corrupted_attn[
            :, : layer_idx + 1
        ]

    def get_component_index(
        self, component_type: str, layer_idx: int, head_idx: Optional[int] = None
    ) -> int:
        """Get the index in the component dimension for a specific component.

        Args:
            component_type: One of "token_embedding", "attention", "mlp"
            layer_idx: Index of the layer
            head_idx: Index of the head (only for attention)

        Returns:
            The index in the component dimension
        """
        key = (component_type, layer_idx, head_idx)
        if key not in self.component_to_idx:
            raise ValueError(
                f"Component not found: {component_type}, layer {layer_idx}, head {head_idx}"
            )
        return self.component_to_idx[key]


class GradientManager:
    """Manages gradients during backpropagation."""

    def __init__(
        self,
        model: HookedTransformer,
        cache_manager: CacheManager,
        logger: logging.Logger,
        return_prune_scores: bool = False,
        abs_edge_strength: bool = False,
    ):
        self.model = model
        self.logger = logger
        self.cache_manager = cache_manager
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_model = model.cfg.d_model
        self.d_head = model.cfg.d_head
        self.n_components = cache_manager.n_components
        self.batch_size = cache_manager.batch_size
        self.seq_len = cache_manager.seq_len
        self.device = cache_manager.device
        self.dtype = cache_manager.dtype
        self.abs_edge_strength = abs_edge_strength

        self.attention_head_grad = torch.zeros(
            (self.batch_size, self.n_layers, self.n_heads, self.seq_len, self.d_head),
            device=self.device,
            dtype=self.dtype,
        )

        self.mlp_and_token_grad = torch.zeros(
            (self.batch_size, self.n_layers + 1, self.seq_len, self.d_model),
            device=self.device,
            dtype=self.dtype,
        )

        # Prune scores for circuit analysis
        self.prune_scores: dict[str, Float[torch.Tensor]] = {}
        self.return_prune_scores = return_prune_scores

        if return_prune_scores:
            self.logger.info("Prune scores will be returned")
        else:
            self.logger.info("Prune scores will not be returned")

    def calculate_feature_attributions(self) -> Float[torch.Tensor, "batch seq_len"]:
        """Calculate feature attributions based on gradients and activation differences."""
        # Extract token embedding component (index 0)
        token_embedding_grad = self.mlp_and_token_grad[:, 0]

        if hasattr(self.cache_manager, "corrupted_mlp_and_token"):
            token_embedding_diff = (
                self.cache_manager.clean_mlp_and_token[:, 0]
                - self.cache_manager.corrupted_mlp_and_token[:, 0]
            )
        else:
            token_embedding_diff = self.cache_manager.clean_mlp_and_token[:, 0]

        return einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            token_embedding_grad,
            token_embedding_diff,
        )

    def get_attn_gradient(
        self,
        layer_idx: int,
    ) -> Float[torch.Tensor, "batch n_heads seq_len d_head"]:
        """Get gradient attention heads

        Args:
            layer_idx: Index of the layer

        """
        return self.attention_head_grad[:, layer_idx]

    def get_mlp_gradient(
        self,
        layer_idx: int,
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """Get mlp gradient

        Args:
            layer_idx: Index of the layer

        """
        return self.mlp_and_token_grad[:, layer_idx + 1]

    def combine_attention_and_mlp_edge_strengths(
        self,
        attn_edge_strengths: Float[
            torch.Tensor, "n_previous_layers n_heads batch seq_len ..."
        ],
        mlp_edge_strengths: Float[torch.Tensor, "n_previous_layers batch seq_len ..."],
    ) -> Float[torch.Tensor, "batch seq_len ... n_previous_components"]:
        """Combine attention and MLP edge strengths into a single tensor that it."""

        # We add the first layer from mlp, then the first layer from attn, then the second layer from mlp, etc.
        # This is done to ensure that the attention and mlp edge strengths are combined in a way that respects the layer structure.
        # the mlp layer can have one more layer that the attn layer
        num_attn_layers = attn_edge_strengths.shape[0]
        num_mlp_layers = mlp_edge_strengths.shape[0]
        edge_strengths = []
        for idx in range(self.n_layers + 1):
            if idx < num_mlp_layers:
                edge_strengths.append(mlp_edge_strengths[idx].unsqueeze(0))
            if idx < num_attn_layers:
                edge_strengths.append(attn_edge_strengths[idx])
        edge_strengths = torch.cat(edge_strengths, dim=0)
        edge_strengths = einops.rearrange(
            edge_strengths,
            "n_previous_components batch seq_len ... -> batch seq_len ... n_previous_components",
        )
        return edge_strengths

    def add_edges_to_prune_scores(
        self,
        module_name: str,
        mlp_token_grad: Float[
            torch.Tensor, "batch ... n_previous_layers seq_len d_model"
        ],
        mlp_token_hidden_states: Float[
            torch.Tensor, "batch n_previous_components seq_len d_model"
        ],
        attn_grad: Optional[
            Float[torch.Tensor, "batch ... n_previous_layers n_heads seq_len d_head"]
        ] = None,
        attn_hidden_states: Optional[
            Float[torch.Tensor, "batch n_previous_layers n_heads seq_len d_head"]
        ] = None,
    ) -> None:
        """Add edges to the prune scores for circuit analysis."""

        if not self.return_prune_scores:
            return

        mlp_and_token_edge_strengths = einsum(
            " batch ... n_previous_layers seq_len d_model, batch n_previous_layers seq_len d_model -> n_previous_layers batch seq_len ... ",
            mlp_token_grad,
            mlp_token_hidden_states,
        )
        if attn_grad is not None and attn_hidden_states is not None:
            attn_edge_strengths = einsum(
                "batch ... n_previous_layers n_heads seq_len d_head, batch n_previous_layers n_heads seq_len d_head -> n_previous_layers n_heads batch seq_len ...",
                attn_grad,
                attn_hidden_states,
            )
            edge_strengths = self.combine_attention_and_mlp_edge_strengths(
                attn_edge_strengths, mlp_and_token_edge_strengths
            )
        else:
            edge_strengths = einops.rearrange(
                mlp_and_token_edge_strengths,
                "n_previous_layers batch seq_len ... -> batch seq_len ... n_previous_layers",
            )

        if self.abs_edge_strength:
            edge_strengths = edge_strengths.abs()

        self.prune_scores[module_name] = einops.reduce(
            edge_strengths,
            "batch ... -> ...",
            "mean",
        )
