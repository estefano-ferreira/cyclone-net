from __future__ import annotations

"""CycloneNet — experimental spatio-temporal attention modules.

Important scientific note
-------------------------
This module is intentionally marked as experimental. The published results in the
current CycloneNet release correspond to the attention-inactive baseline.
If this module is enabled, the resulting model must be reported separately with
controlled ablations under the same splits, seeds, and threshold-selection policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGating(nn.Module):
    """Apply lightweight spatial gating independently at each time step."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,C,H,W), got {tuple(x.shape)}")
        gate = torch.sigmoid(self.proj(x))
        return x * gate


class TemporalSelfAttention(nn.Module):
    """Per-pixel temporal self-attention over the sequence dimension.

    Input shape
    -----------
    (B, T, C, H, W)
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)
        self.scale = float(channels) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x with shape (B,T,C,H,W), got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        xp = x.permute(0, 3, 4, 1, 2).contiguous()  # (B,H,W,T,C)
        x_norm = self.norm(xp)

        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,W,T,T)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)
        y = self.out(y)
        y = y + xp  # residual connection
        return y.permute(0, 3, 4, 1, 2).contiguous()  # (B,T,C,H,W)


class SpatioTemporalAttention(nn.Module):
    """Experimental attention block for ablation studies only."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.spatial = SpatialGating(channels)
        self.temporal = TemporalSelfAttention(channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x with shape (B,C,T,H,W), got {tuple(x.shape)}")

        b, c, t, h, w = x.shape
        spatial_frames = [self.spatial(x[:, :, i, :, :]) for i in range(t)]
        xs = torch.stack(spatial_frames, dim=2)  # (B,C,T,H,W)
        xt = self.temporal(xs.permute(0, 2, 1, 3, 4))
        return xt.permute(0, 2, 1, 3, 4).contiguous()