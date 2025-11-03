import contextlib
import torch
import torch.nn.functional as F


class _ScaleGrad(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, g): 
        return g * ctx.scale, None

def scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _ScaleGrad.apply(x, float(scale))

@contextlib.contextmanager
def _patch_sdpa_qkv_scales(q_scale: float, k_scale: float, v_scale: float):
    if not hasattr(F, "scaled_dot_product_attention"):
        # If no attention is available, raise error
        raise RuntimeError("Attention scale was requested and Pytorch model detected, but torch.nn.functional.scaled_dot_product_attention not found.")
    orig = F.scaled_dot_product_attention

    def sdpa(q, k, v, *args, **kw):
        q = scale_grad(q, q_scale)
        k = scale_grad(k, k_scale)
        v = scale_grad(v, v_scale)
        return orig(q, k, v, *args, **kw)
    
    F.scaled_dot_product_attention = sdpa
    try: 
        yield
    finally: 
        F.scaled_dot_product_attention = orig

def _tlens_qkv_scales(model, q_scale: float, k_scale: float, v_scale: float):
    def hq(q, hook): 
        return scale_grad(q, q_scale)
    def hk(k, hook): 
        return scale_grad(k, k_scale)
    def hv(v, hook): 
        return scale_grad(v, v_scale)
    fwd_hooks = [
        (lambda n: n.endswith(".attn.hook_q"), hq),
        (lambda n: n.endswith(".attn.hook_k"), hk),
        (lambda n: n.endswith(".attn.hook_v"), hv),
    ]
    if len(fwd_hooks) == 0:
        raise RuntimeError("Attention scale was requested and TransformerLens model detected, but no Q/K/V hooks were found in the model.")
    return model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True)