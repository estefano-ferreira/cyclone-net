from __future__ import annotations

"""CycloneNet — physics-guided model (FuelMap + forward constraint).

Outputs:
- ri_logit: (B,)
- dv12: (B,)
- dv24: (B,)
- fuelmap_logits: (B,1,H,W)  (unnormalized spatial logits)
- dv24_forward_hat: (B,)     (optional, requires prior_map_t0)

Input:
- x: (B,C,T,H,W) where C is the profile channel count (4, 9, or 12)

Scientific intent:
- FuelMap is supervised against a physical prior map P (saved during preprocessing).
- Forward head links localized energy score to dv24 as an additional physical constraint.
"""

import torch
import torch.nn as nn


class FuelMapHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # (B,C,T,H,W) -> (B,1,H,W)
        logits_t = self.net(feat)
        return logits_t.mean(dim=2)


class ForwardIntensityHead(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, energy_score: torch.Tensor) -> torch.Tensor:
        return self.mlp(energy_score.unsqueeze(-1)).squeeze(-1)


class CycloneNetPhysicsGuided(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.1, forward_hidden: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fuelmap_head = FuelMapHead(hidden_channels, hidden=32)

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head_ri = nn.Linear(hidden_channels, 1)
        self.head_dv12 = nn.Linear(hidden_channels, 1)
        self.head_dv24 = nn.Linear(hidden_channels, 1)

        self.forward_head = ForwardIntensityHead(hidden_size=forward_hidden)

    def forward(self, x: torch.Tensor, prior_map_t0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feat = self.stem(x)  # (B,hidden,T,H,W)

        fuelmap_logits = self.fuelmap_head(feat)  # (B,1,H,W)

        emb = self.fc(self.pool(feat))
        ri_logit = self.head_ri(emb).squeeze(-1)
        dv12 = self.head_dv12(emb).squeeze(-1)
        dv24 = self.head_dv24(emb).squeeze(-1)

        out = {"ri_logit": ri_logit, "dv12": dv12, "dv24": dv24, "fuelmap_logits": fuelmap_logits}

        if prior_map_t0 is not None:
            Fw = torch.softmax(fuelmap_logits.view(fuelmap_logits.size(0), -1), dim=-1)
            Pw = torch.softmax(prior_map_t0.view(prior_map_t0.size(0), -1), dim=-1)
            energy_score = (Fw * Pw).sum(dim=-1)
            out["dv24_forward_hat"] = self.forward_head(energy_score)

        return out
