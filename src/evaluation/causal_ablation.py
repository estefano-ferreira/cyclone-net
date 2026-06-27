from __future__ import annotations

"""
CycloneNet — counterfactual (ablation) causal test for the FuelMap claim.

Question
--------
Does the model's predicted rapid-intensification (RI) probability *causally* depend
on the region the FuelMap identifies as the energy source?

Method (counterfactual with a control)
--------------------------------------
For each event:
  1. Baseline inference -> p0 = sigmoid(ri_logit), and the FuelMap.
  2. FUEL mask  = top-k% FuelMap pixels (the identified energy source).
     CONTROL mask = bottom-k% FuelMap pixels (same pixel count, the region the model
     itself says is LEAST fuel). The control has identical size and undergoes the
     identical operation, so it isolates the effect of *location*.
  3. Ablate the chosen physical channels (e.g. SST anomaly, wind) inside each mask:
        x' = x * (1 - factor * mask)   (drives the local anomaly toward the mean).
  4. Re-run inference -> p_fuel, p_ctrl. Record the drops d_fuel = p0 - p_fuel and
     d_ctrl = p0 - p_ctrl.

Causal evidence holds only if d_fuel is significantly greater than d_ctrl across
events (paired test): ablating the identified source must hurt the predicted RI
more than ablating an equally-sized region the model considers irrelevant.

This is a model-internal counterfactual: it shows what the *trained model* relies
on. It is strong, reviewer-friendly evidence, not a claim about the real atmosphere.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def topk_bottomk_masks(fuelmap: torch.Tensor, k: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build per-sample top-k% and bottom-k% spatial masks from FuelMap logits.

    fuelmap: (B,1,H,W). Returns two (B,1,H,W) float masks with equal pixel counts.
    """
    b, _, h, w = fuelmap.shape
    flat = fuelmap.view(b, -1)
    n = flat.shape[1]
    n_sel = max(1, int(round(k * n)))

    top_idx = torch.topk(flat, n_sel, dim=1, largest=True).indices
    bot_idx = torch.topk(flat, n_sel, dim=1, largest=False).indices

    top = torch.zeros_like(flat)
    bot = torch.zeros_like(flat)
    top.scatter_(1, top_idx, 1.0)
    bot.scatter_(1, bot_idx, 1.0)
    return top.view(b, 1, h, w), bot.view(b, 1, h, w)


def _ablate(x: torch.Tensor, mask_b1hw: torch.Tensor, ch_indices: List[int], factor: float) -> torch.Tensor:
    """
    x: (B,C,T,H,W); mask: (B,1,H,W). Scale selected channels toward 0 inside the
    mask (across all timesteps). On z-scored inputs, 0 == the channel mean, so this
    removes the local anomaly without inventing out-of-distribution values.
    """
    x2 = x.clone()
    mask_t = mask_b1hw.unsqueeze(2)  # (B,1,1,H,W) -> broadcasts over T
    for ci in ch_indices:
        x2[:, ci:ci + 1] = x2[:, ci:ci + 1] * (1.0 - factor * mask_t)
    return x2


def _forward_scores(model: torch.nn.Module, x: torch.Tensor, prior: Optional[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    out = model(x, prior_map_t0=prior) if isinstance(prior, torch.Tensor) else model(x)
    pri = _sigmoid(out["ri_logit"].detach().cpu().numpy().reshape(-1))
    dv24 = out["dv24"].detach().cpu().numpy().reshape(-1) if "dv24" in out else np.full_like(pri, np.nan)
    fuel = out.get("fuelmap_logits", None)
    return pri, dv24, fuel


def ablation_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    prior: Optional[torch.Tensor],
    ch_indices: List[int],
    k: float,
    factor: float,
) -> Dict[str, np.ndarray]:
    """
    Run the baseline + fuel-ablated + control-ablated forward passes for one batch.

    Returns per-sample arrays: p0, p_fuel, p_ctrl, d_fuel, d_ctrl (RI probability),
    and the dv24 analogues.
    """
    model.eval()
    with torch.no_grad():
        p0, dv0, fuel = _forward_scores(model, x, prior)
        if fuel is None:
            raise ValueError("Model produced no 'fuelmap_logits'; cannot run FuelMap ablation.")

        fuel_mask, ctrl_mask = topk_bottomk_masks(fuel, k)

        x_fuel = _ablate(x, fuel_mask, ch_indices, factor)
        x_ctrl = _ablate(x, ctrl_mask, ch_indices, factor)

        p_fuel, dv_fuel, _ = _forward_scores(model, x_fuel, prior)
        p_ctrl, dv_ctrl, _ = _forward_scores(model, x_ctrl, prior)

    return {
        "p0": p0, "p_fuel": p_fuel, "p_ctrl": p_ctrl,
        "d_fuel": p0 - p_fuel, "d_ctrl": p0 - p_ctrl,
        "dv0": dv0, "dv_fuel": dv_fuel, "dv_ctrl": dv_ctrl,
        "dvd_fuel": dv0 - dv_fuel, "dvd_ctrl": dv0 - dv_ctrl,
    }


def _resolve_channel_indices(input_channel_names: List[str], wanted: List[str]) -> List[int]:
    idx = []
    for name in wanted:
        if name not in input_channel_names:
            raise KeyError(
                f"Ablation channel '{name}' not in model inputs {input_channel_names}."
            )
        idx.append(input_channel_names.index(name))
    return idx


def run_causal_ablation(
    cfg: Dict[str, Any],
    split: str = "test",
    k: float = 0.05,
    factor: float = 0.5,
    channels: Optional[List[str]] = None,
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full counterfactual causal test over a split, using the released checkpoint.
    Writes nothing; returns an auditable report dict.
    """
    from src.data.dataset import PhysicsDataset
    from src.evaluation.evaluate import _load_checkpoint, _select_checkpoint_path
    from src.training.trainer import _build_model
    from src.utils.config import cfg_get
    from torch.utils.data import DataLoader

    channels = channels or ["sst_anom_K", "wind_mps"]

    device_cfg = str(cfg_get(cfg, "training.device", "auto")).lower()
    device = torch.device("cuda" if (device_cfg == "cuda" or (device_cfg == "auto" and torch.cuda.is_available())) else "cpu")

    ds = PhysicsDataset(cfg=cfg, split=split)
    ch_indices = _resolve_channel_indices(list(ds.input_channels_names), channels)

    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = _build_model(cfg).to(device)
    _load_checkpoint(model, _select_checkpoint_path(cfg), device)

    d_fuel_all: List[float] = []
    d_ctrl_all: List[float] = []
    dvd_fuel_all: List[float] = []
    dvd_ctrl_all: List[float] = []
    n_seen = 0

    for batch in loader:
        x = batch["x"].to(device)
        prior = batch.get("prior_map_t0", None)
        if isinstance(prior, torch.Tensor):
            prior = prior.to(device)

        res = ablation_step(model, x, prior, ch_indices, k=k, factor=factor)
        d_fuel_all.extend(res["d_fuel"].tolist())
        d_ctrl_all.extend(res["d_ctrl"].tolist())
        dvd_fuel_all.extend(res["dvd_fuel"].tolist())
        dvd_ctrl_all.extend(res["dvd_ctrl"].tolist())

        n_seen += x.shape[0]
        if max_events is not None and n_seen >= max_events:
            break

    return summarize_ablation(
        d_fuel_all, d_ctrl_all, dvd_fuel_all, dvd_ctrl_all,
        meta={"split": split, "k": k, "factor": factor, "channels": channels,
              "n_events": n_seen, "device": str(device)},
    )


def summarize_ablation(
    d_fuel: List[float],
    d_ctrl: List[float],
    dvd_fuel: Optional[List[float]] = None,
    dvd_ctrl: Optional[List[float]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate paired ablation drops into an auditable causal report."""
    df = np.asarray(d_fuel, dtype=float)
    dc = np.asarray(d_ctrl, dtype=float)
    report: Dict[str, Any] = {"meta": meta or {}}

    if df.size == 0:
        report["status"] = "no_events"
        return report

    diff = df - dc
    paired_t = paired_p = None
    try:
        from scipy.stats import ttest_rel
        if df.size > 1 and np.std(diff) > 0:
            t_res = ttest_rel(df, dc)
            paired_t = float(t_res.statistic)
            paired_p = float(t_res.pvalue)
    except Exception as exc:  # scipy missing or degenerate
        logger.warning("paired t-test skipped: %s", exc)

    significant = bool(paired_p is not None and paired_p < 0.05 and float(np.mean(diff)) > 0.0)

    report.update({
        "status": "ok",
        "ri_probability": {
            "mean_drop_fuel": float(np.mean(df)),
            "mean_drop_control": float(np.mean(dc)),
            "mean_difference_fuel_minus_control": float(np.mean(diff)),
            "fraction_fuel_stronger": float(np.mean(df > dc)),
            "paired_t_statistic": paired_t,
            "paired_p_value": paired_p,
        },
        "causal_evidence": {
            "significant": significant,
            "interpretation": (
                "Ablating the FuelMap-identified region reduces predicted RI "
                "significantly more than ablating an equally-sized low-fuel control region."
                if significant else
                "No significant causal dependence detected: ablating the identified "
                "region does NOT hurt predicted RI more than the control. The localization "
                "may not be what drives the model's RI prediction."
            ),
        },
    })

    if dvd_fuel is not None and dvd_ctrl is not None and len(dvd_fuel) == df.size:
        vf = np.asarray(dvd_fuel, dtype=float)
        vc = np.asarray(dvd_ctrl, dtype=float)
        report["dv24"] = {
            "mean_drop_fuel": float(np.nanmean(vf)),
            "mean_drop_control": float(np.nanmean(vc)),
            "mean_difference_fuel_minus_control": float(np.nanmean(vf - vc)),
        }

    return report


def run_and_save(cfg: Dict[str, Any], split: str, k: float, factor: float,
                 channels: Optional[List[str]], out_path: Path,
                 max_events: Optional[int] = None) -> Dict[str, Any]:
    report = run_causal_ablation(cfg, split=split, k=k, factor=factor,
                                 channels=channels, max_events=max_events)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Causal ablation report saved to %s", out_path)
    return report
