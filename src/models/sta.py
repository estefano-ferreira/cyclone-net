"""Spatio-Temporal Attention blocks (paper-aligned).

Design goal:
  Provide an explicit spatio-temporal attention mechanism over (T,H,W) features,
  while staying lightweight and reproducible.

Implementation:
  - Temporal self-attention at each spatial location (H*W sequences of length T).
  - Optional spatial gating per timestep (learned attention map over HxW).

Input/Output:
  x: (B, C, T, H, W)
  returns: (B, C, T, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        x0 = self.norm(x_seq)
        attn_out, _ = self.mha(x0, x0, x0, need_weights=False)
        return x_seq + self.proj_drop(attn_out)


class SpatialGating(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        x_ = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        w = self.conv(x_)
        w = torch.relu(w) + 1e-6
        w = w / (w.sum(dim=(-2, -1), keepdim=True) + 1e-6)
        x_weighted = x_ * w
        x_weighted = x_weighted.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x_weighted


class SpatioTemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0, spatial_gating: bool = True):
        super().__init__()
        self.temporal = TemporalSelfAttention(
            dim, num_heads, attn_dropout, proj_dropout)
        self.spatial_gating = SpatialGating(dim) if spatial_gating else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_gating is not None:
            x = self.spatial_gating(x)
        B, C, T, H, W = x.shape
        x_seq = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        x_seq = self.temporal(x_seq)
        x_out = x_seq.reshape(B, H, W, T, C).permute(
            0, 4, 3, 1, 2).contiguous()
        return x_out
