"""
Tests for the counterfactual causal-ablation harness.

We validate the harness itself on synthetic models with KNOWN causal structure:
  - a model whose RI depends on the FuelMap region  -> must be flagged significant
  - a model whose FuelMap is unrelated to its RI driver -> must NOT be flagged
This guards against both false negatives and false positives in the causal test.
"""
import torch

from src.evaluation.causal_ablation import (
    ablation_step,
    summarize_ablation,
    topk_bottomk_masks,
)

B, C, T, H, W = 40, 4, 2, 12, 12


def _x():
    torch.manual_seed(0)
    return torch.rand(B, C, T, H, W)


class _DependsOnFuelRegion(torch.nn.Module):
    """RI is driven by the highest-SST pixels, and the FuelMap points at SST."""

    def forward(self, x, prior_map_t0=None):
        sst0 = x[:, 0, 0]                      # (B,H,W)
        fuel = sst0.unsqueeze(1)               # FuelMap == SST field
        ri = 10.0 * torch.topk(sst0.flatten(1), 10, dim=1).values.mean(1)
        return {"ri_logit": ri, "dv24": ri, "fuelmap_logits": fuel}


class _FuelUnrelatedToDriver(torch.nn.Module):
    """RI is driven by SST (ch0) but the FuelMap points at an UNRELATED field (ch1)."""

    def forward(self, x, prior_map_t0=None):
        sst0 = x[:, 0, 0]
        unrelated = x[:, 1, 0].unsqueeze(1)    # FuelMap == ch1, independent of ch0
        ri = 10.0 * torch.topk(sst0.flatten(1), 10, dim=1).values.mean(1)
        return {"ri_logit": ri, "dv24": ri, "fuelmap_logits": unrelated}


def test_masks_are_equal_size_and_disjoint():
    fuel = torch.rand(5, 1, H, W)
    top, bot = topk_bottomk_masks(fuel, k=0.1)
    assert top.sum().item() == bot.sum().item()        # equal pixel counts
    assert (top * bot).sum().item() == 0.0             # disjoint


def test_causal_test_detects_true_dependence():
    res = ablation_step(_DependsOnFuelRegion(), _x(), None, ch_indices=[0], k=0.1, factor=0.9)
    report = summarize_ablation(res["d_fuel"].tolist(), res["d_ctrl"].tolist())
    assert report["ri_probability"]["mean_difference_fuel_minus_control"] > 0
    assert report["causal_evidence"]["significant"] is True


def test_causal_test_rejects_unrelated_fuelmap():
    res = ablation_step(_FuelUnrelatedToDriver(), _x(), None, ch_indices=[0], k=0.1, factor=0.9)
    report = summarize_ablation(res["d_fuel"].tolist(), res["d_ctrl"].tolist())
    # FuelMap is unrelated to the RI driver -> no significant causal dependence.
    assert report["causal_evidence"]["significant"] is False
