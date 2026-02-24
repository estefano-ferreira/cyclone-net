import torch
import torch.nn.functional as F

def energy_balance_loss(
    dv24_pred: torch.Tensor,          # (B,)
    fuelmap: torch.Tensor,            # (B,1,H,W) probabilidades (após softmax)
    latent_heat: torch.Tensor,        # (B,1,H,W) W/m²
    sensible_heat: torch.Tensor,      # (B,1,H,W) W/m²
    dx: float,                        # grid spacing in meters
    dy: float,
    efficiency: float = 0.7,          # eficiência termodinâmica (ajustável)
    rho_air: float = 1.2,             # kg/m³
    cp: float = 1005.0,                # J/kg/K
    scale_factor: float = 1.0          # fator de escala para compatibilizar unidades
) -> torch.Tensor:
    """
    Enforces that the predicted intensity change (dv24) is proportional to the
    FuelMap-weighted average of total heat flux (latent + sensible).
    
    The physical idea: The energy input to the storm is the integral of surface heat fluxes
    over the region where the storm extracts energy. The FuelMap acts as a weighting function
    representing where the storm is actively drawing energy.
    
    Returns MSE loss between scaled heat integral and dv24.
    """
    total_heat = latent_heat + sensible_heat  # (B,1,H,W)
    # Weighted average: sum( fuelmap * total_heat * area ) / sum( fuelmap )
    area_per_pixel = dx * dy  # m² per grid cell
    weighted_sum = (fuelmap * total_heat * area_per_pixel).sum(dim=(2,3))  # (B,1)
    total_weight = fuelmap.sum(dim=(2,3))  # (B,1)
    # Avoid division by zero
    weighted_avg = weighted_sum / (total_weight + 1e-8)  # (B,1) in W/m² * m² = W? Wait, check units.
    # Actually weighted_sum is in W (since heat flux in W/m² * area m² = W). So weighted_avg is in W/m²? No, we divided by total_weight (dimensionless sum of probabilities), so weighted_avg is in W.
    # But dv24 is in knots per 24h. We need a conversion factor.
    # Simple approach: let the model learn the scaling, but we can also provide a physical estimate.
    # For now, we'll use a learnable scalar per batch or a fixed conversion.
    # Alternatively, we can treat this as a consistency term: the two should be correlated.
    
    # Let's compute correlation loss: we want dv24_pred to be proportional to weighted_avg.
    # Use cosine similarity or simply MSE after normalizing both to zero mean and unit variance?
    # But that might be too loose. Better: assume linear relationship and minimize MSE after scaling.
    # We'll let the network learn the scale through another head, or we can add a small MLP that maps weighted_avg to dv24.
    # For simplicity now, we'll just return MSE between dv24_pred and weighted_avg.squeeze(1) after scaling by a learnable parameter.
    # But we don't have a learnable parameter here. We'll pass a scale factor from config.
    
    return F.mse_loss(dv24_pred, weighted_avg.squeeze(1) * scale_factor)