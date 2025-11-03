import contextlib
import torch
import torch.nn.functional as F

class _SoftmaxBackwardTOnly(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, T: float):
        ctx.dim = int(dim) 
        ctx.T = float(T)
        ctx.save_for_backward(x)
        return torch.softmax(x, dim=ctx.dim)
    
    @staticmethod
    def backward(ctx, gout: torch.Tensor):
        (x,) = ctx.saved_tensors 
        dim, T = ctx.dim, ctx.T
        sT = torch.softmax(x / T, dim=dim)
        dot = (gout * sT).sum(dim=dim, keepdim=True)
        gin = sT * (gout - dot)
        return gin, None, None, None

def _softmax_bwT(x: torch.Tensor, *, dim=None, dtype=None, T=1.0):
    if dim is None: dim = x.dim() - 1
    return _SoftmaxBackwardTOnly.apply(x, dim, float(T), dtype)

@contextlib.contextmanager
def _patch_softmax_backward_T_only(T: float):
    orig_F, orig_torch = F.softmax, torch.softmax
    orig_tensor_method = torch.Tensor.softmax

    def F_patched(input, dim=None, _stacklevel=3, dtype=None):
        return _softmax_bwT(input, dim=dim, dtype=dtype, T=T)
    def torch_patched(input, dim, dtype=None):
        return _softmax_bwT(input, dim=dim, dtype=dtype, T=T)
    def tensor_method(self, dim, dtype=None):
        return F_patched(self, dim=dim, dtype=dtype)

    F.softmax = F_patched
    torch.softmax = torch_patched
    torch.Tensor.softmax = tensor_method
    try:
        yield
    finally:
        F.softmax = orig_F
        torch.softmax = orig_torch
        torch.Tensor.softmax = orig_tensor_method