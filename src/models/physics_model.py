"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import torch
import torch.nn as nn
import numpy as np
from src.utils.config import PARAMS

T, H, W, C = PARAMS['T'], PARAMS['H'], PARAMS['W'], PARAMS['C']


class PhysicsGuidedCycloneNet(nn.Module):
    """
    CNN 3D guided physics for cyclone enhancement.
    """

    def __init__(self):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(C, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.energy_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        features = self.conv3d(x)
        intensity = self.energy_head(features)
        return intensity, features


# Global instance of the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysicsGuidedCycloneNet().to(device)
model.eval()


# =========================
# Public functions of the model
# =========================

def predict_numpy(x_np: np.ndarray) -> np.ndarray:
    """
    Recebe numpy (B,T,H,W,C) → retorna intensidade (B,1)
    """
    with torch.no_grad():
        x = torch.from_numpy(x_np).float().permute(0, 4, 1, 2, 3).to(device)
        intensity, _ = model(x)
        return intensity.cpu().numpy()


def hotspot_numpy(x_np: np.ndarray) -> np.ndarray:
    """
    Calculates physical sensitivity map.
    """
    x = torch.from_numpy(x_np).float().permute(0, 4, 1, 2, 3).to(device)
    x.requires_grad_(True)

    intensity, features = model(x)
    intensity.sum().backward()

    grad = x.grad.detach().cpu().numpy()  # (B,C,T,H,W)

    # energy integrated in time and channels
    hotspot = np.linalg.norm(grad, axis=(1, 2))

    # normalization
    hotspot /= hotspot.max(axis=(1, 2), keepdims=True) + 1e-8
    return hotspot
