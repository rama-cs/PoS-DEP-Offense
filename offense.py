import math
import torch
import torch.nn as nn

class SyntaxBiasedAttention(nn.Module):
    """Multi-head attention with additive bias from POS/DEP embeddings."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj_pos = nn.Linear(d_model, d_model)
        self.proj_dep = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, h: torch.Tensor, pos_e: torch.Tensor, dep_e: torch.Tensor, key_padding_mask=None):
        # h: (B, L, D); pos_e/dep_e: (B, L, D)
        bias = self.proj_pos(pos_e) + self.proj_dep(dep_e)  # (B, L, D)
        h_biased = h + bias
        out, attn_w = self.attn(h_biased, h_biased, h_biased, key_padding_mask=~key_padding_mask.bool() if key_padding_mask is not None else None)
        return out, attn_w

class ContrastiveProjection(nn.Module):
    def __init__(self, d_model: int, d_proj: int = 256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_proj), nn.Tanh(), nn.Linear(d_proj, d_proj))
    def forward(self, x):
        z = self.proj(x)  # (B, L, P)
        z = nn.functional.normalize(z, dim=-1)
        return z
