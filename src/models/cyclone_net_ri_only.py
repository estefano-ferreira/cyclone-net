"""CycloneNet RI-only model with explicit Spatio-Temporal Attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import cfg_get
from src.models.sta import SpatioTemporalAttention


class Conv3DBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CycloneNetRIOnly(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        in_ch = int(cfg_get(config, "model.input_channels", 4))
        dropout = float(cfg_get(config, "model.dropout", 0.3))
        base = int(cfg_get(config, "model.backbone.base_filters", 32))

        self.stem = Conv3DBlock(in_ch, base, dropout=dropout * 0.5)
        self.down1 = nn.Sequential(
            Conv3DBlock(base, base * 2, dropout=dropout * 0.5),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.down2 = nn.Sequential(
            Conv3DBlock(base * 2, base * 4, dropout=dropout * 0.5),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.bottleneck = Conv3DBlock(
            base * 4, base * 4, dropout=dropout * 0.5)

        sta_cfg = cfg_get(config, "model.sta", {})
        sta_enabled = bool(sta_cfg.get("enabled", True))
        num_heads = int(sta_cfg.get("num_heads", 4))
        attn_dropout = float(sta_cfg.get("attn_dropout", 0.0))
        proj_dropout = float(sta_cfg.get("proj_dropout", 0.0))
        spatial_gating = bool(sta_cfg.get("spatial_gating", True))

        self.sta = SpatioTemporalAttention(
            dim=base * 4,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            spatial_gating=spatial_gating,
        ) if sta_enabled else nn.Identity()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.bottleneck(x)
        x = self.sta(x)
        logits = self.head(x)
        return logits
