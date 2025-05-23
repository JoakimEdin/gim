# ruff: noqa: F722
# mypy: disable-error-code="name-defined"
from typing import Literal
import logging


from pydantic import BaseModel


class TransformerTraceConfig(BaseModel):
    """Configuration for transformer tracing behavior."""

    layernorm_backward: Literal["grad", "freeze"] = "grad"
    activation_backward: Literal["grad", "freeze"] = "grad"
    query_value_key_divide_grad_strategy: bool = False
    abs_edge_strength: bool = False
    value_only: bool = False
    softmax_backward: Literal[
        "grad",
        "tsg",
        "atpstar",
    ] = "grad"
    tsg_temperature: float = 2.0
    log_level: int = logging.DEBUG
    scale_mlp_gate: bool = False
