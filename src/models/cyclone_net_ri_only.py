"""
CycloneNet: Physics-guided model architecture with FuelMap and optional forward physics model.
Now includes energy‑balance forward model using FuelMap‑weighted heat flux.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sta import SpatioTemporalAttention


class PhysicsGuidedFuelMap(nn.Module):
    """FuelMap with optional physical prior and initialization."""

    def __init__(self, in_channels: int, hidden: int = 16, init_from_physics: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )
        self.init_from_physics = init_from_physics
        self.physical_prior = None  # to be set externally

    def set_physical_prior(self, prior_map: torch.Tensor):
        """Set a physical prior map (e.g., normalized heat flux) for initialization."""
        self.physical_prior = prior_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,H,W)
        m = self.net(x)            # (B,1,T,H,W)
        m = torch.sigmoid(m)
        m = m.mean(dim=2)          # (B,1,H,W)
        # If physical prior is set and we want to initialize, we could blend.
        # For simplicity, we ignore blending here; the prior can be used in loss.
        return m


class SimplifiedIntensificationModel(nn.Module):
    """
    Simplified forward model that estimates dv24 from physical fields at t0.
    Now enhanced to use FuelMap‑weighted heat flux.
    """

    def __init__(self, input_channels: int, hidden_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) – physical fields at t0
        h = self.conv(x).squeeze(-1).squeeze(-1)  # (B, hidden)
        return self.fc(h).squeeze(-1)


class CycloneNetPhysicsGuided(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.1,
                 use_sta: bool = True, fuelmap_enabled: bool = True,
                 physics_guided_config: dict = None):
        super().__init__()
        self.fuelmap_enabled = fuelmap_enabled
        self.use_sta = use_sta
        self.config = physics_guided_config or {}

        if fuelmap_enabled:
            self.fuelmap = PhysicsGuidedFuelMap(
                in_channels,
                hidden=16,
                init_from_physics=self.config.get(
                    'guided_attention', {}).get('init_from_physics', False)
            )
        else:
            self.fuelmap = None

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv3d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.sta = SpatioTemporalAttention(
            hidden_channels) if use_sta else nn.Identity()
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

        # Optional physics-based forward model
        if self.config.get('forward_model', {}).get('enabled', False):
            # Input to forward model: we will compute a single weighted heat flux,
            # so input_channels = 1 (or we could add more features). For flexibility,
            # we keep the conv structure but will pass a (B,1,H,W) tensor.
            self.forward_phys = SimplifiedIntensificationModel(
                input_channels=1,   # weighted heat flux map (single channel)
                hidden_size=self.config['forward_model']['hidden_size']
            )
        else:
            self.forward_phys = None

    def set_physical_prior(self, prior_map: torch.Tensor):
        """Pass physical prior to FuelMap."""
        if self.fuelmap is not None:
            self.fuelmap.set_physical_prior(prior_map)

    def forward(self, x: torch.Tensor, physical_fields: dict = None) -> dict[str, torch.Tensor]:
        """
        x: (B, C, T, H, W)
        physical_fields: optional dict containing fields like 'total_heat_flux' (B,1,H,W) for t0.
        """
        B, C, T, H, W = x.shape

        if self.fuelmap_enabled and self.fuelmap is not None:
            m = self.fuelmap(x)          # (B,1,H,W)
            x = x * m.unsqueeze(2)       # gate across time
        else:
            m = torch.zeros((B, 1, H, W), device=x.device)

        z = self.stem(x)
        z = self.sta(z)
        z = self.pool(z)
        h = self.fc_shared(z)

        ri_logit = self.head_ri(h).squeeze(-1)
        dv12 = self.head_dv12(h).squeeze(-1)
        dv24 = self.head_dv24(h).squeeze(-1)

        output = {
            "ri_logit": ri_logit,
            "dv12": dv12,
            "dv24": dv24,
            "fuelmap": m,
            "embedding": h
        }

        # If forward physics model is enabled and physical_fields provided, add its prediction
        if self.forward_phys is not None and physical_fields is not None:
            total_heat = physical_fields.get('total_heat_flux')  # (B,1,H,W)
            if total_heat is not None and self.fuelmap is not None:
                # Weighted average of total heat flux using FuelMap
                # FuelMap m is (B,1,H,W) in [0,1]
                numerator = (total_heat * m).sum(dim=(2, 3))      # (B,1)
                denominator = m.sum(dim=(2, 3)) + 1e-8           # (B,1)
                weighted_heat = numerator / denominator          # (B,1)
                # Use weighted_heat as the input to forward_phys (expand to (B,1,H,W) with constant spatial map)
                # We create a constant map filled with the scalar weighted_heat for each sample
                heat_map = weighted_heat.view(
                    B, 1, 1, 1).expand(B, 1, H, W)  # (B,1,H,W)
                dv24_phys = self.forward_phys(heat_map)          # (B,)
                output['dv24_phys'] = dv24_phys
            else:
                # Fallback: if no heat flux, use zeros (or could use standard forward model)
                output['dv24_phys'] = torch.zeros_like(dv24)

        return output
