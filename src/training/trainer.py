from __future__ import annotations

"""
CycloneNet — Trainer (config-driven, scientifically auditable).

This module is intentionally minimal and reproducible:
- Public entrypoint: train(cfg)  -> used by run.py
- No NetCDF access (only reads dataset outputs from data/interim)
- No "directional" physical filtering; physics-guided terms are optional and data-driven.

Core tasks:
- RI classification: BCEWithLogits
- dv12 / dv24 regression: SmoothL1 with missing-target masks

Optional physics-guided terms (only if present in batch):
- Prior alignment: KL divergence between model FuelMap distribution and a physical prior map (fuel potential)
- Equation consistency: encourages consistency between u/v and (vort/div) at t0 (lightweight, optional)
- FuelMap regularizers: TV + L1 (optional)

This trainer assumes the dataset returns keys consistent with PhysicsDataset:
  x, y, dv12, dv24, dv12_mask, dv24_mask
  prior_map_t0, prior_mask
  u10_t0, v10_t0, vort_t0, div_t0, dx_m, dy_m, eq_mask
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.utils.config import cfg_get

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Small losses (deterministic, standard)
# -----------------------------------------------------------------------------
def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SmoothL1 with mask (mask=1 for valid targets). Shapes: (B,)"""
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / denom


def tv_loss_2d(logits: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss on logits, encourages spatial smoothness.
    logits: (B,1,H,W)
    """
    dx = torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]).mean()
    dy = torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]).mean()
    return dx + dy


def l1_loss(logits: torch.Tensor) -> torch.Tensor:
    """L1 sparsity on logits (not probabilities)."""
    return torch.abs(logits).mean()


def kl_alignment_loss(fuelmap_logits: torch.Tensor, prior_prob: torch.Tensor) -> torch.Tensor:
    """
    KL(prior || fuelmap) using spatial distributions.
    fuelmap_logits: (B,1,H,W)
    prior_prob: (B,1,H,W) expected to be non-negative; will be normalized here.
    """
    B = fuelmap_logits.shape[0]
    fm = fuelmap_logits.view(B, -1)
    pr = prior_prob.view(B, -1)

    pr = torch.clamp(pr, min=0.0)
    pr = pr / torch.clamp(pr.sum(dim=-1, keepdim=True), min=1e-12)

    fm = torch.softmax(fm, dim=-1)
    kl = (pr * (torch.log(pr + 1e-12) - torch.log(fm + 1e-12))).sum(dim=-1)
    return kl.mean()


def equation_consistency_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    vort: torch.Tensor,
    div: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """
    Enforce that computed finite-diff vort/div from u/v are consistent with provided vort/div.
    u,v,vort,div: (B,1,H,W)
    """
    # Finite differences in torch (central-ish)
    # du/dx
    du_dx = (u[:, :, :, 2:] - u[:, :, :, :-2]) / (2.0 * dx)
    du_dx = F.pad(du_dx, (1, 1, 0, 0), mode="replicate")
    # dv/dy
    dv_dy = (v[:, :, 2:, :] - v[:, :, :-2, :]) / (2.0 * dy)
    dv_dy = F.pad(dv_dy, (0, 0, 1, 1), mode="replicate")

    # dv/dx
    dv_dx = (v[:, :, :, 2:] - v[:, :, :, :-2]) / (2.0 * dx)
    dv_dx = F.pad(dv_dx, (1, 1, 0, 0), mode="replicate")
    # du/dy
    du_dy = (u[:, :, 2:, :] - u[:, :, :-2, :]) / (2.0 * dy)
    du_dy = F.pad(du_dy, (0, 0, 1, 1), mode="replicate")

    vort_hat = dv_dx - du_dy
    div_hat = du_dx + dv_dy

    return F.mse_loss(vort_hat, vort) + F.mse_loss(div_hat, div)


# -----------------------------------------------------------------------------
# Config bundles
# -----------------------------------------------------------------------------
@dataclass
class LossWeights:
    lambda_ri: float = 1.0
    lambda_dv12: float = 0.4
    lambda_dv24: float = 0.6
    lambda_prior: float = 0.2
    lambda_eq: float = 0.1
    lambda_tv: float = 0.001
    lambda_l1: float = 0.0008


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    dev = str(cfg_get(cfg, "training.device", "auto")).lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Build the model in a robust, signature-safe way.

    Why this exists:
    - Different project versions may expose different class names.
    - __init__ signatures may differ (e.g., 'use_sta' not accepted).
    - We must not guess or pass unsupported kwargs.

    Strategy:
    1) Try physics-guided module(s), scan for a nn.Module class with "Guided" in name.
    2) Fallback to RI-only module, scan for a nn.Module class with "RI" in name.
    3) Instantiate using only kwargs accepted by the class signature.
    """
    import importlib
    import inspect
    import torch.nn as nn

    in_ch = int(cfg_get(cfg, "model.input_channels", len(
        cfg_get(cfg, "model.input_channels_names", []))))
    hidden = int(cfg_get(cfg, "model.hidden_channels", 32))
    dropout = float(cfg_get(cfg, "model.dropout", 0.10))
    use_sta = bool(cfg_get(cfg, "model.use_sta", True))

    def _pick_model_class(mod, prefer_keywords: list[str]) -> type[nn.Module]:
        """
        Pick the best candidate class from a module by scanning nn.Module subclasses.
        prefer_keywords: ordered keywords to prefer in the class name.
        """
        candidates: list[type[nn.Module]] = []
        for name in dir(mod):
            obj = getattr(mod, name, None)
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                candidates.append(obj)

        if not candidates:
            raise ImportError(
                f"No nn.Module subclasses found in module: {mod.__name__}")

        def score(cls: type[nn.Module]) -> int:
            n = cls.__name__.lower()
            s = 0
            for i, kw in enumerate(prefer_keywords):
                if kw.lower() in n:
                    s += (len(prefer_keywords) - i) * 10
            if "cyclonenet" in n:
                s += 5
            return s

        candidates.sort(key=score, reverse=True)
        return candidates[0]

    def _instantiate(cls: type[nn.Module], kwargs: Dict[str, Any]) -> nn.Module:
        """
        Instantiate cls using only kwargs accepted by its __init__ signature.
        """
        sig = inspect.signature(cls.__init__)
        accepted = set(sig.parameters.keys())
        # remove 'self'
        accepted.discard("self")
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return cls(**filtered)

    # Common kwargs we would like to pass if supported
    common_kwargs = {
        "in_channels": in_ch,
        "input_channels": in_ch,   # some implementations use input_channels
        "hidden_channels": hidden,
        "dropout": dropout,
        "use_sta": use_sta,
        "cfg": cfg,
        "config": cfg,
    }

    # 1) Try physics-guided models
    guided_modules = [
        "src.models.cyclone_net_physics_guided",
        "src.models.cyclone_net_physics_guided_true",
        "src.models.cyclonenet_physics_guided",
    ]
    for mname in guided_modules:
        try:
            mod = importlib.import_module(mname)
            cls = _pick_model_class(mod, prefer_keywords=[
                                    "guided", "physics", "cyclonenet"])
            logger.info(f"Using model class: {mname}.{cls.__name__}")
            return _instantiate(cls, common_kwargs)
        except Exception:
            continue

    # 2) Fallback: RI-only
    ri_modules = [
        "src.models.cyclone_net_ri_only",
        "src.models.cyclonenet_ri_only",
    ]
    last_err: Optional[Exception] = None
    for mname in ri_modules:
        try:
            mod = importlib.import_module(mname)
            cls = _pick_model_class(mod, prefer_keywords=["ri", "cyclonenet"])
            logger.info(f"Using model class: {mname}.{cls.__name__}")
            return _instantiate(cls, common_kwargs)
        except Exception as e:
            last_err = e
            continue

    raise ImportError(f"Could not build a model. Last error: {last_err}")


def _build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val loaders using PhysicsDataset(cfg, split=...).
    """
    from src.data.dataset import PhysicsDataset  # lazy import

    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    num_workers = int(cfg_get(cfg, "repro.num_workers", 4))

    ds_train = PhysicsDataset(cfg, split="train")
    ds_val = PhysicsDataset(cfg, split="val")

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def _build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    lr = float(cfg_get(cfg, "training.lr", 7e-4))
    wd = float(cfg_get(cfg, "training.weight_decay", 1e-4))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _load_loss_weights(cfg: Dict[str, Any]) -> LossWeights:
    return LossWeights(
        lambda_ri=float(cfg_get(cfg, "training.lambda_ri", 1.0)),
        lambda_dv12=float(cfg_get(cfg, "training.lambda_dv12", 0.4)),
        lambda_dv24=float(cfg_get(cfg, "training.lambda_dv24", 0.6)),
        lambda_prior=float(cfg_get(cfg, "training.lambda_heat", 0.2)),
        lambda_eq=float(cfg_get(cfg, "training.lambda_phys_consistency", 0.1)),
        lambda_tv=float(cfg_get(cfg, "training.lambda_fuelmap_tv", 0.001)),
        lambda_l1=float(cfg_get(cfg, "training.lambda_fuelmap_l1", 0.0008)),
    )


# -----------------------------------------------------------------------------
# AUC computation on validation set
# -----------------------------------------------------------------------------
def compute_val_auc(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Compute ROC-AUC on validation set using raw logits."""
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            out = model(x)
            scores = torch.sigmoid(out["ri_logit"]).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(y)
    return roc_auc_score(all_labels, all_scores)


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------
@torch.no_grad()
def _eval_one_epoch(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    w: LossWeights,
) -> Dict[str, float]:
    model.eval()

    totals = {"loss": 0.0, "loss_ri": 0.0, "loss_dv12": 0.0,
              "loss_dv24": 0.0, "loss_prior": 0.0, "loss_eq": 0.0}
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        dv12 = batch["dv12"].to(device)
        dv24 = batch["dv24"].to(device)
        dv12_mask = batch["dv12_mask"].to(device)
        dv24_mask = batch["dv24_mask"].to(device)

        out = model(x)

        loss_ri = F.binary_cross_entropy_with_logits(out["ri_logit"], y)
        loss_dv12 = masked_smooth_l1(out["dv12"], dv12, dv12_mask)
        loss_dv24 = masked_smooth_l1(out["dv24"], dv24, dv24_mask)

        loss = w.lambda_ri * loss_ri + w.lambda_dv12 * \
            loss_dv12 + w.lambda_dv24 * loss_dv24

        loss_prior = torch.tensor(0.0, device=device)
        loss_eq = torch.tensor(0.0, device=device)

        # Optional prior alignment if model provides fuelmap logits and dataset provides prior
        if "fuelmap_logits" in out and "prior_map_t0" in batch:
            prior = batch["prior_map_t0"].to(device)
            prior_mask = batch.get("prior_mask", torch.zeros(
                (x.size(0),), device=device)).to(device)
            if prior_mask.sum() > 0:
                m = prior_mask > 0.5
                loss_prior = kl_alignment_loss(
                    out["fuelmap_logits"][m], prior[m])
                loss = loss + w.lambda_prior * loss_prior
                loss = loss + w.lambda_tv * \
                    tv_loss_2d(out["fuelmap_logits"]) + \
                    w.lambda_l1 * l1_loss(out["fuelmap_logits"])

        # Optional equation consistency at t0
        if "fuelmap_logits" in out and batch.get("eq_mask", None) is not None:
            eq_mask = batch["eq_mask"].to(device)
            if eq_mask.sum() > 0:
                m = eq_mask > 0.5
                u = batch["u10_t0"].to(device)[m]
                v = batch["v10_t0"].to(device)[m]
                vort = batch["vort_t0"].to(device)[m]
                div = batch["div_t0"].to(device)[m]
                dx = float(batch["dx_m"].to(device)[m].mean().item())
                dy = float(batch["dy_m"].to(device)[m].mean().item())
                loss_eq = equation_consistency_loss(
                    u, v, vort, div, dx=dx, dy=dy)
                loss = loss + w.lambda_eq * loss_eq

        totals["loss"] += float(loss.item())
        totals["loss_ri"] += float(loss_ri.item())
        totals["loss_dv12"] += float(loss_dv12.item())
        totals["loss_dv24"] += float(loss_dv24.item())
        totals["loss_prior"] += float(loss_prior.item())
        totals["loss_eq"] += float(loss_eq.item())
        n += 1

    for k in totals:
        totals[k] /= max(1, n)
    return totals


def _train_one_epoch(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    w: LossWeights,
    epoch: int,
    epochs: int,
) -> Dict[str, float]:
    model.train()

    totals = {"loss": 0.0, "loss_ri": 0.0, "loss_dv12": 0.0,
              "loss_dv24": 0.0, "loss_prior": 0.0, "loss_eq": 0.0}
    n = 0

    iterable = loader
    if tqdm is not None:
        iterable = tqdm(loader, desc=f"train {epoch}/{epochs}", unit="batch")

    for batch in iterable:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        dv12 = batch["dv12"].to(device)
        dv24 = batch["dv24"].to(device)
        dv12_mask = batch["dv12_mask"].to(device)
        dv24_mask = batch["dv24_mask"].to(device)

        optim.zero_grad(set_to_none=True)

        out = model(x)

        loss_ri = F.binary_cross_entropy_with_logits(out["ri_logit"], y)
        loss_dv12 = masked_smooth_l1(out["dv12"], dv12, dv12_mask)
        loss_dv24 = masked_smooth_l1(out["dv24"], dv24, dv24_mask)

        loss = w.lambda_ri * loss_ri + w.lambda_dv12 * \
            loss_dv12 + w.lambda_dv24 * loss_dv24

        loss_prior = torch.tensor(0.0, device=device)
        loss_eq = torch.tensor(0.0, device=device)

        # Optional prior alignment
        if "fuelmap_logits" in out and "prior_map_t0" in batch:
            prior = batch["prior_map_t0"].to(device)
            prior_mask = batch.get("prior_mask", torch.zeros(
                (x.size(0),), device=device)).to(device)
            if prior_mask.sum() > 0:
                m = prior_mask > 0.5
                loss_prior = kl_alignment_loss(
                    out["fuelmap_logits"][m], prior[m])
                loss = loss + w.lambda_prior * loss_prior
                loss = loss + w.lambda_tv * \
                    tv_loss_2d(out["fuelmap_logits"]) + \
                    w.lambda_l1 * l1_loss(out["fuelmap_logits"])

        # Optional equation consistency
        if "fuelmap_logits" in out and batch.get("eq_mask", None) is not None:
            eq_mask = batch["eq_mask"].to(device)
            if eq_mask.sum() > 0:
                m = eq_mask > 0.5
                u = batch["u10_t0"].to(device)[m]
                v = batch["v10_t0"].to(device)[m]
                vort = batch["vort_t0"].to(device)[m]
                div = batch["div_t0"].to(device)[m]
                dx = float(batch["dx_m"].to(device)[m].mean().item())
                dy = float(batch["dy_m"].to(device)[m].mean().item())
                loss_eq = equation_consistency_loss(
                    u, v, vort, div, dx=dx, dy=dy)
                loss = loss + w.lambda_eq * loss_eq

        loss.backward()
        optim.step()

        totals["loss"] += float(loss.item())
        totals["loss_ri"] += float(loss_ri.item())
        totals["loss_dv12"] += float(loss_dv12.item())
        totals["loss_dv24"] += float(loss_dv24.item())
        totals["loss_prior"] += float(loss_prior.item())
        totals["loss_eq"] += float(loss_eq.item())
        n += 1

    for k in totals:
        totals[k] /= max(1, n)
    return totals


# -----------------------------------------------------------------------------
# Public entrypoints expected by run.py
# -----------------------------------------------------------------------------
def train(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Public config-driven entrypoint expected by run.py.
    """
    device = _resolve_device(cfg)
    train_loader, val_loader = _build_loaders(cfg)
    model = _build_model(cfg).to(device)
    optim = _build_optimizer(cfg, model)
    w = _load_loss_weights(cfg)

    epochs = int(cfg_get(cfg, "training.epochs", 1))

    out_dir = Path(cfg_get(cfg, "paths.results_dir",
                   "./outputs/results")).resolve()
    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir",
                    "./models/checkpoints")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_auc = -1.0
    best_loss_path = ckpt_dir / "best_model.pt"
    best_auc_path = ckpt_dir / "best_auc_model.pt"
    history = []

    logger.info(f"Device: {device}")
    logger.info(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    logger.info(f"Loss weights: {w}")

    # Log model architecture to verify STA
    logger.info(f"Model class: {model.__class__.__name__}")
    if hasattr(model, 'use_sta'):
        logger.info(f"use_sta = {model.use_sta}")

    for ep in range(1, epochs + 1):
        tr = _train_one_epoch(cfg, model, train_loader,
                              optim, device, w, epoch=ep, epochs=epochs)
        va = _eval_one_epoch(cfg, model, val_loader, device, w)

        # Compute validation AUC
        val_auc = compute_val_auc(model, val_loader, device)

        row = {"epoch": ep, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}, "val_auc": val_auc}
        history.append(row)

        logger.info(f"[epoch {ep}/{epochs}] train_loss={tr['loss']:.6f} val_loss={va['loss']:.6f} "
                    f"val_ri={va['loss_ri']:.6f} val_dv24={va['loss_dv24']:.6f} val_auc={val_auc:.4f}")

        # Save best model by validation loss (optional, can keep)
        if va["loss"] < best_val_loss:
            best_val_loss = va["loss"]
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                    "cfg": cfg,
                    "best_val_loss": best_val_loss,
                },
                best_loss_path,
            )
            logger.debug(
                f"Best loss model updated at epoch {ep} (loss={best_val_loss:.4f})")

        # Save best model by validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_auc_path)
            logger.info(
                f"New best model by AUC (AUC={val_auc:.4f}) saved to {best_auc_path}")

    # Save training history
    hist_path = out_dir / "train_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "best_val_loss": float(best_val_loss),
        "best_val_auc": float(best_val_auc),
        "checkpoint_loss": str(best_loss_path),
        "checkpoint_auc": str(best_auc_path),
        "history": str(hist_path),
    }


def main() -> None:
    """CLI entrypoint for debugging."""
    from src.utils.config import load_config
    cfg = load_config("config.yaml")
    train(cfg)


if __name__ == "__main__":
    main()
