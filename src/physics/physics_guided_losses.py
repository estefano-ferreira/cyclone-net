# src/physics/physics_guided_losses.py
from __future__ import annotations

"""Physics-guided losses for CycloneNet.

This module enables the core claim:
  "The model is physics-guided via supervised physical fuel potential maps and
   equation-consistency constraints."

Implemented losses:
- KL alignment between FuelMap logits and physical prior map P
- Equation consistency: vort/div derived from u/v must match diagnostic channels
- Regularizers: TV (smoothness) and L1 (sparsity) for FuelMap
"""

import torch
import torch.nn.functional as F


def fuelmap_kl_alignment_loss(fuelmap_logits: torch.Tensor, prior_map: torch.Tensor) -> torch.Tensor:
    """KL(P || FuelMap) where both are converted to spatial distributions by softmax.

    fuelmap_logits: (B,1,H,W) unnormalized logits
    prior_map:      (B,1,H,W) non-negative map (will be normalized via softmax)
    """
    p = prior_map.view(prior_map.size(0), -1)
    q = fuelmap_logits.view(fuelmap_logits.size(0), -1)
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    kl = (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(dim=-1)
    return kl.mean()


def tv_loss_2d(x: torch.Tensor) -> torch.Tensor:
    """Total variation regularization for (B,1,H,W)."""
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw


def l1_loss(x: torch.Tensor) -> torch.Tensor:
    """L1 sparsity regularization."""
    return torch.abs(x).mean()


def vort_div_from_uv(u: torch.Tensor, v: torch.Tensor, dx: float, dy: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute discrete vorticity and divergence from u/v using central differences.

    u, v: (B,1,H,W) in m/s
    dx, dy: meters (scalar)
    Returns:
      vort, div: (B,1,H,W) in s^-1
    """
    # d/dx kernel: [-1, 0, 1] / (2*dx); d/dy kernel: transpose equivalent
    kx = torch.tensor([[-1.0, 0.0, 1.0]], device=u.device,
                      dtype=u.dtype).view(1, 1, 1, 3) / (2.0 * dx)
    ky = torch.tensor([[-1.0, 0.0, 1.0]], device=u.device,
                      dtype=u.dtype).view(1, 1, 3, 1) / (2.0 * dy)

    du_dx = F.conv2d(u, kx, padding=(0, 1))
    du_dy = F.conv2d(u, ky, padding=(1, 0))
    dv_dx = F.conv2d(v, kx, padding=(0, 1))
    dv_dy = F.conv2d(v, ky, padding=(1, 0))

    vort = dv_dx - du_dy
    div = du_dx + dv_dy
    return vort, div


def equation_consistency_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    vort_channel: torch.Tensor,
    div_channel: torch.Tensor,
    dx: float,
    dy: float,
    lambda_vort: float = 1.0,
    lambda_div: float = 1.0,
) -> torch.Tensor:
    """
    Enforce vort/div channels to match derivatives derived from u/v.

    Note: The accuracy of finite differences depends on the grid spacing (dx, dy).
    The current implementation uses central differences with padding, which is
    acceptable for the 0.25° ERA5 grid. For other resolutions, the loss scale may
    need adjustment.
    """
    vort_uv, div_uv = vort_div_from_uv(u, v, dx=dx, dy=dy)
    loss_vort = F.mse_loss(vort_channel, vort_uv)
    loss_div = F.mse_loss(div_channel, div_uv)
    return lambda_vort * loss_vort + lambda_div * loss_div
