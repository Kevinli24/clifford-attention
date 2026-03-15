import torch
import torch.nn as nn
from clifford_layer import CliffordAttention


class StandardTransformerModel(nn.Module):
    """
    Each input vector becomes its own token.
    Attention must aggregate across tokens → dot product is forced to matter.
    """
    def __init__(self, token_dim: int, n_tokens: int, d_model: int,
                 num_heads: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, d_model)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads,
                                  dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        # Pool over tokens then predict
        self.output_head = nn.Sequential(
            nn.Linear(d_model * n_tokens, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )
        self.n_tokens = n_tokens

    def forward(self, x):
        # x: (B, n_tokens, token_dim)
        x = self.token_proj(x) # (B, n_tokens, d_model)
        for attn, norm in zip(self.layers, self.norms):
            out, _ = attn(x, x, x)
            x = norm(x + out)
        x = x.reshape(x.size(0), -1) # (B, n_tokens * d_model)
        return self.output_head(x)


class CliffordTransformerModel(nn.Module):
    def __init__(self, token_dim: int, n_tokens: int, d_model: int,
                 num_heads: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % (num_heads * 8) == 0, \
            f"d_model must be divisible by num_heads*8={num_heads*8}"
        self.token_proj = nn.Linear(token_dim, d_model)
        self.layers = nn.ModuleList([
            CliffordAttention(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.output_head = nn.Sequential(
            nn.Linear(d_model * n_tokens, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim)
        )
        self.n_tokens = n_tokens

    def forward(self, x):
        x = self.token_proj(x)
        for attn, norm in zip(self.layers, self.norms):
            out = attn(x)
            x = norm(x + out)
        x = x.reshape(x.size(0), -1)
        return self.output_head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)