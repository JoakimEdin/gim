import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# Adjust this import to where you saved the context manager
from src.gim.context.gim import GIM

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load a small TLens model
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.zero_grad(set_to_none=True)
model.eval()  # avoid dropout for a clean demo

# 2) Make a tiny synthetic batch and next-token targets
texts = [
    "hello world",
    "the cat sat on the mat",
    "transformers are fun",
    "gradient-based methods rock",
]
tokens = model.to_tokens(texts, prepend_bos=True).to(device)   # [B, T+1]
x = tokens[:, :-1]                                             # inputs  [B, T]
y = tokens[:, 1:]                                              # targets [B, T]

# 3) Run with GIM (freeze LN/RMSNorm stats, softmax T=2, Q/K÷4, V÷2)
with GIM(model,
         freeze_norm=True,
         softmax_temperature=2.0,
         q_scale=0.25, k_scale=0.25, v_scale=0.5):
    logits = model(x)  # [B, T, V]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [B*T, V]
        y.reshape(-1)                         # [B*T]
    )
    loss.backward()

print("loss:", float(loss))
# Peek at some grads (should be nonzero)
print("||dW_Q||:", model.blocks[0].attn.W_Q.grad.norm().item())
print("||dW_K||:", model.blocks[0].attn.W_K.grad.norm().item())
print("||dW_V||:", model.blocks[0].attn.W_V.grad.norm().item())
