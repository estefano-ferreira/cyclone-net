from __future__ import annotations

"""CycloneNet — baseline/ablation model for RI classification and intensity deltas.

Scientific role
---------------
This file provides a clean baseline-compatible architecture for ablation studies.
It can optionally enable an experimental attention block, but the default and
recommended configuration for the current release is attention-inactive.

Important release constraint
----------------------------
Heat-flux fields are NOT used as inputs or as forward-physics drivers in the
reported release. Therefore, this model does not consume total heat flux and
does not implement a forward heat-flux prediction path.
"""

import torch
import torch.nn as nn

from src.models.sta import SpatioTemporalAttention


class FuelMapHead(nn.Module):
    """Small spatial head that aggregates temporal evidence into a 2D FuelMap."""

    def __init__(self, in_channels: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 5:
            raise ValueError(f"Expected feat with shape (B,C,T,H,W), got {tuple(feat.shape)}")
        logits_t = self.net(feat)  # (B,1,T,H,W)
        return logits_t.mean(dim=2)  # (B,1,H,W)


class CycloneNetRiOnly(nn.Module):
    """Release-aligned baseline model.

    Outputs
    -------
    - ri_logit: (B,)
    - dv12: (B,)
    - dv24: (B,)
    - fuelmap_logits: (B,1,H,W)

    Notes
    -----
    The FuelMap is produced for interpretability and weakly supervised physics-
    guided training, but this class does not inject external heat-flux products
    into the learning path.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.1,
        use_sta: bool = False,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.sta = SpatioTemporalAttention(hidden_channels, dropout=dropout) if use_sta else nn.Identity()
        self.fuelmap_head = FuelMapHead(hidden_channels, hidden=16)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head_ri = nn.Linear(hidden_channels, 1)
        self.head_dv12 = nn.Linear(hidden_channels, 1)
        self.head_dv24 = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor, prior_map_t0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected x with shape (B,C,T,H,W), got {tuple(x.shape)}")

        feat = self.stem(x)
        feat = self.sta(feat)

        fuelmap_logits = self.fuelmap_head(feat)
        emb = self.fc_shared(self.pool(feat))

        ri_logit = self.head_ri(emb).squeeze(-1)
        dv12 = self.head_dv12(emb).squeeze(-1)
        dv24 = self.head_dv24(emb).squeeze(-1)

        return {
            "ri_logit": ri_logit,
            "dv12": dv12,
            "dv24": dv24,
            "fuelmap_logits": fuelmap_logits,
        }