from src.gim.context.gim import GIM

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True):
        B, T, D = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]
        k = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]
        v = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, Dh]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)      # [B, H, T, Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D)                      # [B, T, D]
        return self.Wo(y)

class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinyAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, max_len: int = 256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.LongTensor):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)   # [B, T, V]
        return logits

device = "cuda" if torch.cuda.is_available() else "cpu"
V = 1000
B, T = 4, 32

model = TinyLM(vocab_size=V, d_model=128, n_layers=2, n_heads=4).to(device)
batch_tokens = torch.randint(0, V, (B, T), device=device)
batch_targets = torch.randint(0, V, (B,), device=device)


with GIM(model, freeze_norm=True, softmax_temperature=2.0,
         q_scale=0.25, k_scale=0.25, v_scale=0.5):
    logits = model(batch_tokens)
    loss = F.cross_entropy(logits[:, -1, :], batch_targets)
    loss.backward()

print("loss:", float(loss))
print("Grad ||Wq||:", model.blocks[0].attn.Wq.weight.grad.norm().item())
print("Grad ||Wk||:", model.blocks[0].attn.Wk.weight.grad.norm().item())
print("Grad ||Wv||:", model.blocks[0].attn.Wv.weight.grad.norm().item())
