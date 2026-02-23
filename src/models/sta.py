from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGating(nn.Module):
    """Spatial attention/gating over (H,W) for each time step."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        a = self.proj(x)  # (B,1,H,W)
        a = torch.sigmoid(a)
        return x * a

class TemporalSelfAttention(nn.Module):
    """Per-pixel temporal self-attention over T."""
    def __init__(self, channels: int):
        super().__init__()
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C,H,W) -> for each (H,W), attend over T
        B, T, C, H, W = x.shape
        xp = x.permute(0, 3, 4, 1, 2).contiguous()  # (B,H,W,T,C)
        q = self.q(xp)  # (B,H,W,T,C)
        k = self.k(xp)
        v = self.v(xp)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,W,T,T)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,W,T,C)
        out = out.permute(0, 3, 4, 1, 2).contiguous()  # (B,T,C,H,W)
        return out

class SpatioTemporalAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = SpatialGating(channels)
        self.temporal = TemporalSelfAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,H,W)
        B, C, T, H, W = x.shape
        # spatial gating per time step
        xs = []
        for t in range(T):
            xs.append(self.spatial(x[:, :, t, :, :]))
        x2 = torch.stack(xs, dim=2)  # (B,C,T,H,W)
        # temporal self-attn expects (B,T,C,H,W)
        x3 = self.temporal(x2.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return x3
