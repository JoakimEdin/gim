from typing import Callable, Optional

import captum
import torch
from transformer_lens import HookedTransformer
from lxt.efficient import monkey_patch
from fancy_einsum import einsum

from src.gim.transpose_trace import (
    TransformerTrace,
    TransformerTraceConfig,
)

Explainer = Callable[[torch.Tensor, torch.Tensor, str | torch.device], torch.Tensor]


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids).logits[:, -1]


def create_baseline_input(
    input_ids: torch.Tensor,
    baseline_token_id: int = 50_000,
    cls_token_id: Optional[int] = 0,
    eos_token_id: Optional[int] = 2,
    n_tokens_before_context: Optional[int] = None,
    n_tokens_after_context: Optional[int] = None,
) -> torch.Tensor:
    """Create baseline input for a given input

    Args:
        input_ids (torch.Tensor): Input ids to create baseline input for
        baseline_token_id (int, optional): Baseline token id. Defaults to 50_000.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        torch.Tensor: Baseline input
    """

    if n_tokens_before_context is not None and n_tokens_after_context is not None:
        baseline = input_ids.clone()
        baseline[:, n_tokens_before_context:-n_tokens_after_context] = baseline_token_id
        return baseline

    baseline = torch.ones_like(input_ids) * baseline_token_id
    if cls_token_id is not None:
        baseline[:, 0] = cls_token_id
    if eos_token_id is not None:
        baseline[:, -1] = eos_token_id
    return baseline


def embedding_attributions_to_token_attributions(
    attributions: torch.Tensor,
) -> torch.Tensor:
    """Convert embedding attributions to token attributions.

    Args:
        attributions (torch.Tensor): Embedding Attributions,

    Returns:
        torch.Tensor: Token attributions
    """

    return attributions.sum(-1)


def get_attnlrp_callable(
    model: HookedTransformer,
    baseline_token_id: int = 50_000,
    cls_token_id: Optional[int] = 0,
    eos_token_id: Optional[int] = 2,
    **kwargs,
) -> Explainer:
    if "llama" in model.cfg.model_name.lower():
        print("Loading LLAMA model")
        from transformers.models.llama import modeling_llama

        monkey_patch(modeling_llama)
        model = modeling_llama.LlamaForCausalLM.from_pretrained(
            model.cfg.tokenizer_name
        ).to(model.cfg.device)

    elif "qwen" in model.cfg.model_name.lower():
        print("Loading Qwen model")
        from transformers.models.qwen2 import modeling_qwen2

        monkey_patch(modeling_qwen2)
        model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(
            model.cfg.tokenizer_name
        ).to(model.cfg.device)

    else:
        raise ValueError(
            f"AttnLRP only works with Llama and Qwen models, not {model.cfg.model_name}"
        )

    # Deactivate gradients on parameters
    for param in model.parameters():
        param.requires_grad = False

    # Optionally enable gradient checkpointing (2x forward pass)
    model.train()
    model.gradient_checkpointing_enable()

    def attnlrp_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        answer = answer.to(device).squeeze()

        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.requires_grad_(True)

        # Inference
        output_logits = model(inputs_embeds=input_embeds, use_cache=False).logits

        # Take the maximum logit at last token position. You can also explain any other token, or several tokens together!
        if wrong_answer is None:
            target_logit = output_logits[0, -1, answer]
        else:
            target_logit = (
                output_logits[0, -1, answer] - output_logits[0, -1, wrong_answer]
            )

        # Backward pass (the relevance is initialized with the value of max_logits)
        target_logit.backward()

        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.get_input_embeddings()(corrupt)

        # Obtain relevance. (Works at any layer in the model!)
        relevance = (
            (input_embeds.grad * (input_embeds - corrupt_embedding))
            .float()
            .sum(-1)
            .detach()
            .cpu()
        )  # Cast to float32 for higher precision

        return relevance

    return attnlrp_callable


@torch.no_grad()
def get_atpstar_callable(
    model: HookedTransformer,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
    config = TransformerTraceConfig(
        layernorm_backward="grad",
        activation_backward="grad",
        query_value_key_divide_grad_strategy=False,
        abs_edge_strength=False,
        value_only=False,
        softmax_backward="atpstar",
        log_level=0,
        scale_mlp_gate=False,
    )

    @torch.no_grad()
    def atpstar_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        clean = input_ids.to(device)
        answer = answer.to(device)

        if wrong_answer is not None:
            wrong_answer = wrong_answer.to(device)
            # Get the gradient of the wrong answer
            output_embeddings_grad = trace.answer_diff_direction(
                answer,
                wrong_answer,
            )
        else:
            output_embeddings_grad = trace.answer_direction(answer)

        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.embed(corrupt)
        clean_embedding = model.embed(clean)

        _, clean_cache = model.run_with_cache(clean)
        gradient_manager = trace.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
            return_prune_scores=False,
        )
        gradient = gradient_manager.mlp_and_token_grad[:, 0]

        feature_attributions = einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            (clean_embedding - corrupt_embedding),
            gradient,
        )

        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return atpstar_callable


@torch.no_grad()
def get_transformerlrp_callable(
    model: HookedTransformer,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
    config = TransformerTraceConfig(
        layernorm_backward="freeze",
        activation_backward="grad",
        query_value_key_divide_grad_strategy=False,
        abs_edge_strength=False,
        value_only=True,
        softmax_backward="grad",
        log_level=0,
        scale_mlp_gate=False,
    )

    @torch.no_grad()
    def transformerlrp_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        answer = answer.to(device)
        model.cfg.ungroup_grouped_query_attention = True
        _, clean_cache = model.run_with_cache(
            input_ids,
        )
        output_embeddings_grad = trace.answer_direction(answer)
        # Run the backward pass with the new API
        feature_attributions = trace.compute_feature_attributions(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
        )
        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return transformerlrp_callable


@torch.no_grad()
def get_gim_callable(
    model: HookedTransformer,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
    config = TransformerTraceConfig(
        layernorm_backward="freeze",
        activation_backward="grad",
        query_value_key_divide_grad_strategy=True,
        abs_edge_strength=False,
        value_only=False,
        softmax_backward="tsg",
        log_level=0,
        scale_mlp_gate=True,
    )

    @torch.no_grad()
    def gim_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        clean = input_ids.to(device)
        answer = answer.to(device)
        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.embed(corrupt)
        clean_embedding = model.embed(clean)

        model.cfg.ungroup_grouped_query_attention = True

        _, clean_cache = model.run_with_cache(
            clean,
        )

        if wrong_answer is not None:
            wrong_answer = wrong_answer.to(device)
            # Get the gradient of the wrong answer
            output_embeddings_grad = trace.answer_diff_direction(
                answer,
                wrong_answer,
            )
        else:
            output_embeddings_grad = trace.answer_direction(answer)

        # Run the backward pass with the new API
        gradient_manager = trace.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
            return_prune_scores=False,
        )
        grad = gradient_manager.mlp_and_token_grad[:, 0]

        feature_attributions = einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            (clean_embedding - corrupt_embedding),
            grad,
        )
        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return gim_callable


@torch.no_grad()
def get_grad_tsg_callable(
    model: HookedTransformer,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
    config = TransformerTraceConfig(
        layernorm_backward="grad",
        activation_backward="grad",
        query_value_key_divide_grad_strategy=False,
        abs_edge_strength=False,
        value_only=False,
        softmax_backward="tsg",
        log_level=0,
        scale_mlp_gate=False,
    )

    @torch.no_grad()
    def grad_tsg_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        clean = input_ids.to(device)
        answer = answer.to(device)
        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.embed(corrupt)
        clean_embedding = model.embed(clean)

        model.cfg.ungroup_grouped_query_attention = True

        _, clean_cache = model.run_with_cache(
            clean,
        )

        if wrong_answer is not None:
            wrong_answer = wrong_answer.to(device)
            # Get the gradient of the wrong answer
            output_embeddings_grad = trace.answer_diff_direction(
                answer,
                wrong_answer,
            )
        else:
            output_embeddings_grad = trace.answer_direction(answer)

        # Run the backward pass with the new API
        gradient_manager = trace.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
            return_prune_scores=False,
        )
        grad = gradient_manager.mlp_and_token_grad[:, 0]

        feature_attributions = einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            (clean_embedding - corrupt_embedding),
            grad,
        )
        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return grad_tsg_callable


@torch.no_grad()
def get_grad_freeze_norm_callable(
    model: HookedTransformer,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
    config = TransformerTraceConfig(
        layernorm_backward="freeze",
        activation_backward="grad",
        query_value_key_divide_grad_strategy=True,
        abs_edge_strength=False,
        value_only=False,
        softmax_backward="grad",
        log_level=0,
        scale_mlp_gate=True,
    )

    @torch.no_grad()
    def grad_freeze_norm_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        model.cfg.ungroup_grouped_query_attention = True
        clean = input_ids.to(device)
        answer = answer.to(device)

        if wrong_answer is not None:
            wrong_answer = wrong_answer.to(device)
            # Get the gradient of the wrong answer
            output_embeddings_grad = trace.answer_diff_direction(
                answer,
                wrong_answer,
            )
        else:
            output_embeddings_grad = trace.answer_direction(answer)

        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.embed(corrupt)
        clean_embedding = model.embed(clean)

        _, clean_cache = model.run_with_cache(clean)
        gradient_manager = trace.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
            return_prune_scores=False,
        )
        gradient = gradient_manager.mlp_and_token_grad[:, 0]

        feature_attributions = einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            (clean_embedding - corrupt_embedding),
            gradient,
        )

        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return grad_freeze_norm_callable


@torch.no_grad()
def get_grad_freeze_callable(
    model: HookedTransformer,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
    config = TransformerTraceConfig(
        layernorm_backward="freeze",
        activation_backward="grad",
        query_value_key_divide_grad_strategy=False,
        abs_edge_strength=False,
        value_only=False,
        softmax_backward="grad",
        log_level=0,
        scale_mlp_gate=False,
    )

    @torch.no_grad()
    def grad_freeze_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        model.cfg.ungroup_grouped_query_attention = True
        clean = input_ids.to(device)
        answer = answer.to(device)

        if wrong_answer is not None:
            wrong_answer = wrong_answer.to(device)
            # Get the gradient of the wrong answer
            output_embeddings_grad = trace.answer_diff_direction(
                answer,
                wrong_answer,
            )
        else:
            output_embeddings_grad = trace.answer_direction(answer)

        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.embed(corrupt)
        clean_embedding = model.embed(clean)

        _, clean_cache = model.run_with_cache(clean)
        gradient_manager = trace.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
            return_prune_scores=False,
        )
        gradient = gradient_manager.mlp_and_token_grad[:, 0]

        feature_attributions = einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            (clean_embedding - corrupt_embedding),
            gradient,
        )

        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return grad_freeze_callable


@torch.no_grad()
def get_grad_norm_callable(
    model: HookedTransformer,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
    config = TransformerTraceConfig(
        layernorm_backward="grad",
        activation_backward="grad",
        query_value_key_divide_grad_strategy=True,
        abs_edge_strength=False,
        value_only=False,
        softmax_backward="grad",
        log_level=0,
        scale_mlp_gate=True,
    )

    @torch.no_grad()
    def grad_norm_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        model.cfg.ungroup_grouped_query_attention = True
        clean = input_ids.to(device)
        answer = answer.to(device)

        if wrong_answer is not None:
            wrong_answer = wrong_answer.to(device)
            # Get the gradient of the wrong answer
            output_embeddings_grad = trace.answer_diff_direction(
                answer,
                wrong_answer,
            )
        else:
            output_embeddings_grad = trace.answer_direction(answer)

        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.embed(corrupt)
        clean_embedding = model.embed(clean)

        _, clean_cache = model.run_with_cache(clean)
        gradient_manager = trace.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
            return_prune_scores=False,
        )
        gradient = gradient_manager.mlp_and_token_grad[:, 0]

        feature_attributions = einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            (clean_embedding - corrupt_embedding),
            gradient,
        )

        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return grad_norm_callable


@torch.no_grad()
def get_grad_baseline_callable(
    model: HookedTransformer,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    model.eval()

    trace = TransformerTrace(model)

    # Create the configuration
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

    @torch.no_grad()
    def grad_baseline_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        model.cfg.ungroup_grouped_query_attention = True
        clean = input_ids.to(device)
        answer = answer.to(device)

        if wrong_answer is not None:
            wrong_answer = wrong_answer.to(device)
            # Get the gradient of the wrong answer
            output_embeddings_grad = trace.answer_diff_direction(
                answer,
                wrong_answer,
            )
        else:
            output_embeddings_grad = trace.answer_direction(answer)

        corrupt = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        corrupt_embedding = model.embed(corrupt)
        clean_embedding = model.embed(clean)

        _, clean_cache = model.run_with_cache(clean)
        gradient_manager = trace.backward(
            output_embeddings_grad=output_embeddings_grad,
            clean_cache=clean_cache,
            config=config,
            return_prune_scores=False,
        )
        gradient = gradient_manager.mlp_and_token_grad[:, 0]

        feature_attributions = einsum(
            "batch seq_len d_model, batch seq_len d_model -> batch seq_len",
            (clean_embedding - corrupt_embedding),
            gradient,
        )

        model.cfg.ungroup_grouped_query_attention = False
        torch.cuda.empty_cache()
        return feature_attributions.squeeze(0).cpu()

    return grad_baseline_callable


@torch.no_grad()
def get_occlusion_1_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    **kwargs,
) -> Explainer:
    def occlusion_1_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        num_classes = answer.shape[0]
        attributions = torch.zeros(sequence_length, num_classes)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            y_pred = (
                model(input_ids)[:, -1, answer].detach().cpu().squeeze(0)
            )  # [num_classes]

        for idx in range(sequence_length):
            permuted_input_ids = input_ids.clone()
            permuted_input_ids[:, idx] = baseline_token_id
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                y_permute = (
                    model(permuted_input_ids)[:, -1, answer].detach().cpu().squeeze(0)
                )  # [num_classes]
            attributions[idx] = y_pred - y_permute

        return attributions

    return occlusion_1_callable


def get_deeplift_callable(
    model_in: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    **kwargs,
) -> Explainer:
    """Get a callable DeepLift explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 50_000.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable DeepLift explainer
    """
    model = ModelWrapper(model_in)
    explainer = captum.attr.LayerDeepLift(model, model.embed, multiply_by_inputs=True)

    def deeplift_callable(
        input_ids: torch.Tensor,
        device: str | torch.device,
        answer: torch.Tensor,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        """Calculate DeepLift attributions for each class in answer.

        Args:
            inputs (torch.Tensor): Input token ids [batch_size, sequence_length]
            device (str | torch.device): Device to use
            answer (torch.Tensor): Target token ids [num_classes]

        Returns:
            torch.Tensor: Attributions [sequence_length, num_classes]
        """
        sequence_length = input_ids.shape[1]
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )

        num_classes = answer.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            for idx, target in enumerate(answer):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    target=target,
                )
                attributions = embedding_attributions_to_token_attributions(
                    attributions
                )
                class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions

    return deeplift_callable


def get_gradient_x_input_callable(
    model: torch.nn.Module,
    **kwargs,
) -> Explainer:
    """Get a callable Gradient x Input explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Gradient x Input explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            attention_mask=attention_mask,
        )

        return output[:, -1]

    explainer = captum.attr.LayerGradientXActivation(
        predict, model.embed, multiply_by_inputs=True
    )

    def gradients_x_input_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.tensor:
        input_ids = input_ids.to(device)
        sequence_length = input_ids.shape[1]
        num_classes = answer.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(answer):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                attributions = explainer.attribute(
                    input_ids,
                    target=answer,
                )
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions

    return gradients_x_input_callable


def get_integrated_gradient_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    batch_size: int = 1,
    **kwargs,
) -> Explainer:
    """Get a callable Integrated Gradients explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): EOS token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            attention_mask=attention_mask,
        )

        return output[:, -1]

    explainer = captum.attr.LayerIntegratedGradients(
        predict, model.embed, multiply_by_inputs=True
    )

    def integrated_gradients_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
        n_steps: int = 50,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)

        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        sequence_length = input_ids.shape[1]
        num_classes = answer.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        for idx, target in enumerate(answer):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    target=target,
                    internal_batch_size=batch_size,
                    n_steps=n_steps,
                )
            attributions = embedding_attributions_to_token_attributions(attributions)
            class_attributions[:, idx] = attributions.detach().cpu()
        return class_attributions

    return integrated_gradients_callable


def get_kernelshap_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: Optional[int] = 0,
    eos_token_id: Optional[int] = 2,
    sample_ratio: int = 3,
    **kwargs,
) -> Explainer:
    """Get a callable Kernel Shap explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            attention_mask=attention_mask,
        )

        return output[:, -1]

    explainer = captum.attr.KernelShap(predict)

    @torch.no_grad()
    def kernelshap_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        feature_mask: Optional[torch.Tensor] = None,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )

        sequence_length = input_ids.shape[1]
        num_classes = answer.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for idx, target in enumerate(answer):
                attributions = explainer.attribute(
                    input_ids,
                    baselines=baseline,
                    n_samples=sample_ratio * sequence_length,
                    target=target,
                    feature_mask=feature_mask,
                )
                class_attributions[:, idx] = attributions.squeeze().detach().cpu()
        return class_attributions

    return kernelshap_callable


@torch.no_grad()
def get_lime_callable(
    model: torch.nn.Module,
    baseline_token_id: int = 1,
    cls_token_id: Optional[int] = 0,
    eos_token_id: Optional[int] = 2,
    sample_ratio: int = 3,
    **kwargs,
) -> Explainer:
    """Get a callable Kernel Shap explainer

    Args:
        model (torch.nn.Module): Model to explain
        baseline_token_id (int, optional): Baseline token id. Defaults to 0.
        cls_token_id (int, optional): CLS token id. Defaults to 0.
        eos_token_id (int, optional): SEP token id. Defaults to 2.

    Returns:
        Explainer: Callable Integrated Gradients explainer
    """

    def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(
            inputs,
            attention_mask=attention_mask,
        )

        return output[:, -1]

    explainer = captum.attr.Lime(predict)

    @torch.no_grad()
    def lime_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device)
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            n_tokens_before_context=n_tokens_before_context,
            n_tokens_after_context=n_tokens_after_context,
        )
        sequence_length = input_ids.shape[1]
        num_classes = answer.shape[0]
        class_attributions = torch.zeros(sequence_length, num_classes)
        # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        for idx, target in enumerate(answer):
            attributions = explainer.attribute(
                input_ids,
                baselines=baseline,
                n_samples=100,
                target=target,
                perturbations_per_eval=1,
            )
            print(answer)
            class_attributions[:, idx] = attributions.squeeze().detach().cpu()
        return class_attributions

    return lime_callable


def get_random_baseline_callable(model, **kwargs) -> Explainer:
    def random_baseline_callable(
        input_ids: torch.Tensor,
        answer: torch.Tensor,
        device: str | torch.device,
        wrong_answer: Optional[torch.Tensor] = None,
        n_tokens_before_context: Optional[int] = None,
        n_tokens_after_context: Optional[int] = None,
    ) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        num_classes = answer.shape[0]
        attributions = torch.abs(torch.randn((sequence_length, num_classes)))
        return attributions / attributions.sum(0, keepdim=True)

    return random_baseline_callable
