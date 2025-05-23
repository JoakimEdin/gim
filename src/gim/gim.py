# ruff: noqa: F722
# mypy: disable-error-code="name-defined"
import logging

import torch
from jaxtyping import Float
import einops
from typing import Optional
from fancy_einsum import einsum
from transformer_lens import ActivationCache, HookedTransformer

from src.gim.utils import CacheManager, GradientManager
from src.gim.layernorm import LayerNormBackpropagator
from src.gim.mlp import MLPBackpropagator
from src.gim.attention import AttentionBackpropagator
from src.gim.config import TransformerTraceConfig


class GIM:
    def __init__(self, model: HookedTransformer):
        """This class implements backpropagation through the transformer model used for interpretability. It uses special rules for non-linear operations
        to improve the causal interpretation of the model. Make sure that the model uses layernorm folding.

        Args:
            model (HookedTransformer): Model from the transformer_lens module. The model should have the following configuration:
                - fold_ln = True,
                - center_writing_weights = True,
                - center_unembed = False,
                - fold_value_biases = True,
        """
        # Set up logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Add a handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(
            f"Initializing {self.__class__.__name__} with model: {model.cfg.model_name}"
        )

        # # Validate model configuration
        # assert model.cfg.use_attn_result is True
        # assert model.cfg.use_attn_in is False
        # assert model.cfg.use_split_qkv_input is False
        # assert model.cfg.use_hook_mlp_in is True

        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.n_components = 1 + self.n_layers * (self.n_heads + 1)
        self.d_model = model.cfg.d_model
        self.device = next(model.parameters()).device

        self.logger.debug(
            f"Model config: layers={self.n_layers}, heads={self.n_heads}, d_model={self.d_model}, device={self.device}"
        )

        # Initialize backpropagator helpers
        self.ln_backpropagator = LayerNormBackpropagator(model, self.logger)
        self.mlp_backpropagator = MLPBackpropagator(
            model, self.logger, self.ln_backpropagator
        )
        self.attn_backpropagator = AttentionBackpropagator(
            model, self.logger, self.ln_backpropagator
        )

        # Store model biases for later use
        self.attn_biases: Float[torch.Tensor, "n_layers d_model"] = self.model.b_O

        self.logger.info(f"Initialized {self.__class__.__name__}")

    def backward_layer(
        self,
        layer: int,
        config: TransformerTraceConfig,
        cache_manager: CacheManager,
        gradient_manager: GradientManager,
    ) -> None:
        """Backpropagate the gradients through a layer.

        Args:
            layer: Layer index
            config: Configuration for the backpropagation
            cache_manager: Manager for accessing activation caches
            gradient_manager: Manager for handling gradients
        """
        # Backpropagate through MLP
        self.mlp_backpropagator.backward(
            layer,
            activation_grad_approx=config.activation_backward,
            layernorm_method=config.layernorm_backward,
            scale_mlp_gate=config.scale_mlp_gate,
            cache_manager=cache_manager,
            gradient_manager=gradient_manager,
        )

        self.attn_backpropagator.backward(
            layer,
            value_only=config.value_only,
            softmax_method=config.softmax_backward,
            layernorm_method=config.layernorm_backward,
            query_value_key_divide_grad_strategy=config.query_value_key_divide_grad_strategy,
            tsg_temperature=config.tsg_temperature,
            cache_manager=cache_manager,
            gradient_manager=gradient_manager,
        )

    def answer_diff_direction(
        self,
        correct_answers: Float[torch.Tensor, "batch n_correct_answers"]
        | list[Float[torch.Tensor, " n_correct_answers"]],
        incorrect_answers: Float[torch.Tensor, "batch n_incorrect_answers"]
        | list[Float[torch.Tensor, " n_incorrect_answers"]],
    ) -> Float[torch.Tensor, " d_model"]:
        """Calculate the direction in embedding space that points from incorrect to correct answers.

        Args:
            correct_answers: Tensor or list of tensors representing correct answer tokens
            incorrect_answers: Tensor or list of tensors representing incorrect answer tokens

        Returns:
            Direction vector pointing from incorrect to correct answers
        """
        if isinstance(correct_answers, torch.Tensor):
            correct_answer_direction = self.model.tokens_to_residual_directions(
                correct_answers
            )
            incorrect_answer_direction = self.model.tokens_to_residual_directions(
                incorrect_answers
            )
            if incorrect_answer_direction.dim() == 3:
                incorrect_answer_direction = incorrect_answer_direction.mean(dim=1)
                correct_answer_direction = correct_answer_direction.mean(dim=1)

            return correct_answer_direction - incorrect_answer_direction

        if isinstance(correct_answers, list) and isinstance(incorrect_answers, list):
            # The answers will be a list when the answers are not of the same length
            answer_diff_direction_list = []
            for correct_answer, incorrect_answer in zip(
                correct_answers, incorrect_answers
            ):
                correct_answer_direction = self.model.tokens_to_residual_directions(
                    correct_answer
                )
                incorrect_answer_direction = self.model.tokens_to_residual_directions(
                    incorrect_answer
                )
                answer_diff_direction_list.append(
                    correct_answer_direction.mean(0)
                    - incorrect_answer_direction.mean(0)
                )

            return torch.vstack(answer_diff_direction_list)

        raise ValueError(
            "The correct_answers and incorrect_answers must be either a list or a tensor"
        )

    def answer_direction(
        self,
        correct_answers: Float[torch.Tensor, "batch n_correct_answers"]
        | list[Float[torch.Tensor, " n_correct_answers"]],
    ) -> Float[torch.Tensor, " d_model"]:
        """Calculate the direction in embedding space that points from incorrect to correct answers.

        Args:
            correct_answers: Tensor or list of tensors representing correct answer tokens
            incorrect_answers: Tensor or list of tensors representing incorrect answer tokens

        Returns:
            Direction vector pointing from incorrect to correct answers
        """
        if isinstance(correct_answers, torch.Tensor):
            correct_answer_direction = self.model.tokens_to_residual_directions(
                correct_answers
            )
            if correct_answer_direction.dim() == 3:
                correct_answer_direction = correct_answer_direction.mean(dim=1)

            return correct_answer_direction

        if isinstance(correct_answers, list):
            # The answers will be a list when the answers are not of the same length
            answer_direction_list = []
            for correct_answer in correct_answers:
                correct_answer_direction = self.model.tokens_to_residual_directions(
                    correct_answer
                )

                answer_direction_list.append(correct_answer_direction.mean(0))

            return torch.vstack(answer_direction_list)

        raise ValueError(
            "The correct_answers and incorrect_answers must be either a list or a tensor"
        )

    def add_classification_grad(
        self,
        output_embeddings_grad: Float[torch.Tensor, "batch d_model"],
        cache_manager: CacheManager,
        gradient_manager: GradientManager,
    ) -> Float[torch.Tensor, "batch d_model"]:
        ln_scale = cache_manager.clean_cache["ln_final.hook_scale"][:, -1]
        output_embeddings_grad = output_embeddings_grad / ln_scale
        output_embeddings_grad = einops.repeat(
            output_embeddings_grad,
            "batch d_model -> batch 1 d_model",
        )
        gradient_manager.mlp_and_token_grad[:, :, -1] += output_embeddings_grad
        attn_grads = einsum(
            "batch n_layers d_model, n_layers n_heads d_head d_model -> batch n_layers n_heads d_head",
            output_embeddings_grad,
            self.model.W_O,
        )
        gradient_manager.attention_head_grad[:, :, :, -1] += attn_grads

        if gradient_manager.return_prune_scores:
            (
                mlp_and_token_hidden_states,
                attn_head_hidden_states,
            ) = cache_manager.get_activations_from_previous_layers(
                layer_idx=self.n_layers, component_type="attention"
            )

            mlp_and_token_hidden_states = mlp_and_token_hidden_states.clone()
            attn_head_hidden_states = attn_head_hidden_states.clone()
            mlp_and_token_hidden_states[:, :, :-1] = 0.0
            attn_head_hidden_states[:, :, :, :-1] = 0.0

            mlp_token_grad = einops.repeat(
                output_embeddings_grad,
                "batch n_layers d_model -> batch n_layers 1 d_model",
            )
            attn_grads = einops.repeat(
                attn_grads,
                "batch n_layers n_heads d_head -> batch n_layers n_heads 1 d_head",
            )
            gradient_manager.add_edges_to_prune_scores(
                module_name=f"blocks.{self.n_layers - 1}.hook_resid_post",
                mlp_token_grad=mlp_token_grad,
                mlp_token_hidden_states=mlp_and_token_hidden_states,
                attn_grad=attn_grads,
                attn_hidden_states=attn_head_hidden_states,
            )

    @torch.no_grad()
    def backward(
        self,
        output_embeddings_grad: Float[torch.Tensor, "batch d_model"],
        config: TransformerTraceConfig,
        clean_cache: ActivationCache,
        corrupted_cache: Optional[ActivationCache] = None,
        return_prune_scores: bool = True,
    ) -> GradientManager:
        """Backpropagate the gradients through the transformer model.

        Args:
            output_embeddings_grad: Gradients of the output embeddings
            clean_cache: Cache from a clean forward pass
            corrupted_cache: Optional cache from a corrupted forward pass
            config: Configuration for transformer tracing behavior

        Returns:
            Dictionary of prune scores for circuit analysis
        """

        if output_embeddings_grad.dtype != torch.float32:
            raise ValueError(
                f"Output embeddings gradient must be of type float32, but got {output_embeddings_grad.dtype}"
            )

        # Set logger level from config
        self.logger.setLevel(config.log_level)

        self.logger.info("Starting full backward pass")
        self.logger.debug(
            f"Output embeddings gradient shape: {output_embeddings_grad.shape}"
        )
        self.logger.info(config)

        # Initialize managers for this backward pass
        cache_manager = CacheManager(
            self.model, clean_cache, self.logger, corrupted_cache
        )
        gradient_manager = GradientManager(
            self.model,
            cache_manager,
            self.logger,
            return_prune_scores,
            abs_edge_strength=config.abs_edge_strength,
        )

        # Add the gradients of the output embeddings
        self.logger.debug("Adding output embedding gradients to final token position")

        ln_scale = cache_manager.clean_cache["ln_final.hook_scale"][:, -1]
        output_embeddings_grad = output_embeddings_grad / ln_scale

        self.add_classification_grad(
            output_embeddings_grad,
            cache_manager,
            gradient_manager,
        )

        for layer in range(self.n_layers - 1, -1, -1):
            self.logger.info(f"Processing layer {layer}")
            self.backward_layer(layer, config, cache_manager, gradient_manager)
            torch.cuda.empty_cache()

        # Return the prune scores for circuit analysis
        return gradient_manager

    def compute_feature_attributions(
        self,
        output_embeddings_grad: Float[torch.Tensor, "batch d_model"],
        clean_cache: ActivationCache,
        corrupted_cache: Optional[ActivationCache] = None,
        config: Optional[TransformerTraceConfig] = None,
    ) -> Float[torch.Tensor, "batch seq_len"]:
        """Calculate importance of each token position based on gradients and activation differences.

        Args:
            clean_cache: Cache from a clean forward pass
            corrupted_cache: Optional cache from a corrupted forward pass
            output_embeddings_grad: Optional gradients of the output embeddings
            config: Optional configuration for transformer tracing

        Returns:
            Token importance scores for each position
        """
        # Run backward pass if needed
        gradient_manager = self.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            config=config,
            return_prune_scores=False,
        )
        return gradient_manager.calculate_feature_attributions()
