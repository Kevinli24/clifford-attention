import torch
import torch.nn as nn
import time
from clifford_layer import CliffordAttention

def benchmark(fn, label, n=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{label:<30} {ms:.3f} ms/iter")

device = 'cuda'
B, T, D, H = 4, 64, 256, 4   # batch, seq_len, d_model, heads
# D=256, H=4 → head_dim=64 → n_mv=8 multivectors per head ✓

x = torch.randn(B, T, D, device=device)

std  = nn.MultiheadAttention(D, H, batch_first=True).to(device)
clf  = CliffordAttention(D, H).to(device)

print(f"\nConfig: B={B}, T={T}, d_model={D}, heads={H}")
print("-" * 50)
benchmark(lambda: std(x, x, x), "Standard Attention (MHA)")
benchmark(lambda: clf(x),        "Clifford Attention")
print("-" * 50)
print("\nNote: Clifford is expected to be slower — no native CUDA kernel.")
print("The value is richer geometric expressiveness, not raw speed.")