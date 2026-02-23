"""Label Smoothing Binary Cross-Entropy Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )
        return loss
