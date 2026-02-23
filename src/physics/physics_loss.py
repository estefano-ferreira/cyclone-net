"""
Physics-guided loss functions for CycloneNet.
Enforces thermodynamic consistency and alignment with heat fluxes.
"""

import torch
import torch.nn.functional as F


def heat_flux_alignment_loss(
    fuelmap: torch.Tensor,           # (B,1,H,W)
    total_heat_flux: torch.Tensor,   # (B,1,H,W) normalized (e.g., to [0,1])
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Loss that forces the FuelMap to focus on regions of high total heat flux.
    Uses KL divergence between the FuelMap distribution and the normalized heat flux distribution.
    """
    # Flatten and convert to probability distributions
    fm = fuelmap.view(fuelmap.size(0), -1)
    hf = total_heat_flux.view(total_heat_flux.size(0), -1)
    # Normalize heat flux to sum to 1 (probability distribution)
    hf = torch.softmax(hf, dim=-1)
    # FuelMap values are in [0,1] but may not sum to 1; we also apply softmax to make it a distribution
    fm = torch.softmax(fm, dim=-1)

    # KL divergence: sum(p * log(p/q))
    kl = (hf * (torch.log(hf + 1e-8) - torch.log(fm + 1e-8))).sum(dim=-1)
    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    else:
        return kl


def physics_consistency_loss(
    dv24_pred: torch.Tensor,
    total_heat_flux: torch.Tensor,
    fuelmap: torch.Tensor,
    sst: torch.Tensor,
    msl: torch.Tensor,
    dx: float,
    dy: float,
    lambda_heat: float = 1.0,
    lambda_grad: float = 0.1
) -> torch.Tensor:
    """
    Composite loss linking future wind change (dv24) to total heat flux and pressure gradients.
    - Heat term: dv24 should be proportional to the heat flux averaged over the FuelMap region.
    - Gradient term: FuelMap should align with high MSLP gradients.
    """
    # 1. Weighted average of heat flux by FuelMap
    fm = fuelmap  # (B,1,H,W)
    B, _, H, W = fm.shape
    heat_masked = (fm * total_heat_flux).view(B, -1).sum(dim=-1)  # (B,)
    # Normalize both to similar scales (optional, but helps)
    loss_heat = F.mse_loss(dv24_pred, heat_masked)

    # 2. Pressure gradient alignment
    # Compute MSLP gradients using finite differences
    # MSLP shape: (B,1,H,W)
    dp_dx = (msl[:, :, :, 1:] - msl[:, :, :, :-1]) / dx   # (B,1,H,W-1)
    dp_dy = (msl[:, :, 1:, :] - msl[:, :, :-1, :]) / dy   # (B,1,H-1,W)
    # Magnitude (approximate at interior)
    grad_mag = torch.sqrt(dp_dx[:, :, :, :-1]**2 +
                          dp_dy[:, :, :-1, :]**2)  # (B,1,H-1,W-1)
    # Pad to original size
    grad_mag = F.pad(grad_mag, (0, 1, 0, 1))  # (B,1,H,W)
    # Maximize overlap (negative sign to minimize loss)
    loss_grad = - (fm * grad_mag).mean()

    return lambda_heat * loss_heat + lambda_grad * loss_grad
