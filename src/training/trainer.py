from __future__ import annotations

"""
CycloneNet — config-driven trainer aligned with the released scientific scope.

Scientific intent
-----------------
This trainer is designed to remain consistent with the CycloneNet release:

- The primary task is RI classification.
- Auxiliary tasks dv12 / dv24 are allowed, but must not dominate the training
  objective unless explicitly configured.
- Threshold selection is performed on validation only and then reused unchanged
  during evaluation.
- Checkpoints and reports strictly follow the configured project paths.

Important design principle
--------------------------
This module must obey the project architecture declared in config.yaml.
No training artifact path is hardcoded outside the configuration.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.utils.config import cfg_get
from src.utils.paths import rel_to_root

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(cfg: Dict[str, Any]) -> torch.device:
    """
    Resolve device from config while preserving explicit project control.
    """
    device_cfg = str(cfg_get(cfg, "training.device", "auto")).lower()

    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("training.device='cuda' but CUDA is not available.")
        return torch.device("cuda")
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise ValueError(f"Unsupported training.device value: {device_cfg}")


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked SmoothL1 loss for auxiliary regression targets.
    """
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return loss.sum() / denom


def tv_loss_2d(logits: torch.Tensor) -> torch.Tensor:
    """
    Total variation penalty for spatial smoothness of FuelMap logits.
    """
    dx = torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]).mean()
    dy = torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]).mean()
    return dx + dy


def l1_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    L1 sparsity penalty for FuelMap logits.
    """
    return torch.abs(logits).mean()


def kl_alignment_loss(fuelmap_logits: torch.Tensor, prior_prob: torch.Tensor) -> torch.Tensor:
    """
    KL alignment between predicted FuelMap distribution and prior map distribution.

    This is a weak physics-guided alignment term. It is not ground truth.
    """
    b, _, h, w = fuelmap_logits.shape
    p = torch.softmax(fuelmap_logits.view(b, -1), dim=-1)

    q = torch.clamp(prior_prob.view(b, -1), min=0.0)
    q = q / torch.clamp(q.sum(dim=-1, keepdim=True), min=1e-8)

    return torch.sum(
        q * (torch.log(torch.clamp(q, min=1e-8)) - torch.log(torch.clamp(p, min=1e-8))),
        dim=-1,
    ).mean()


def select_threshold_by_f1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Select threshold on validation by maximizing F1.

    This avoids degenerate thresholds such as 0.0 that may appear when
    optimizing recall alone under severe class imbalance.
    """
    best_f1 = -1.0
    best_threshold = 0.5

    unique_scores = np.unique(y_scores)
    if unique_scores.size == 0:
        return 0.5

    candidates = np.concatenate(
        [
            np.array([0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]),
            unique_scores,
        ]
    )
    candidates = np.unique(np.clip(candidates, 0.0, 1.0))

    for threshold in candidates:
        pred = (y_scores >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return float(best_threshold)


@dataclass
class EpochResult:
    """
    Container for per-epoch training or validation metrics.
    """
    loss: float
    cls_loss: float
    dv12_loss: float
    dv24_loss: float
    phys_loss: float
    roc_auc: Optional[float]


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move tensor entries of a batch to the target device.
    """
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    """
    Compute ROC-AUC safely when both classes are present.
    """
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def _extract_ri_logit(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Extract RI classification logits from model output dictionary.
    """
    if "ri_logit" not in outputs:
        raise KeyError("Model outputs must contain 'ri_logit'.")
    return outputs["ri_logit"].float().view(-1)


def _extract_regression(outputs: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    """
    Extract a scalar regression head from model outputs.
    """
    if key not in outputs:
        raise KeyError(f"Model outputs must contain '{key}'.")
    return outputs[key].float().view(-1)


def _build_loaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """
    Build train/validation loaders from the configured dataset.
    """
    from src.data.dataset import PhysicsDataset

    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    num_workers = int(cfg_get(cfg, "repro.num_workers", 0))
    pin_memory = torch.cuda.is_available()

    train_ds = PhysicsDataset(cfg=cfg, split="train")
    val_ds = PhysicsDataset(cfg=cfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def _build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Build the released physics-guided model.

    The model class is imported explicitly so the trained architecture is
    deterministic and auditable — no dynamic discovery, no silent fallback.
    """
    from src.models.cyclone_net_physics_guided import CycloneNetPhysicsGuided

    in_ch = int(
        cfg_get(
            cfg,
            "model.input_channels",
            len(cfg_get(cfg, "model.input_channels_names", [])),
        )
    )
    # A mismatch between the explicit count and the names list would only
    # surface as an opaque conv-shape error at the first batch — fail here
    # with a message that says what to fix instead.
    _names = cfg_get(cfg, "model.input_channels_names", [])
    if _names and in_ch != len(_names):
        raise ValueError(
            f"model.input_channels ({in_ch}) != len(model.input_channels_names) "
            f"({len(_names)}). Fix the config: the two must agree (or drop "
            f"model.input_channels to derive it from the names list)."
        )
    # The dataset appends one extra ADT ocean channel when enabled; the model's first
    # conv must account for it. Single source of truth: the config flag.
    if bool(cfg_get(cfg, "model.use_adt_input", False)):
        in_ch += 1
    hidden = int(cfg_get(cfg, "model.hidden_channels", 32))
    dropout = float(cfg_get(cfg, "model.dropout", 0.10))

    logger.info("Using model class: src.models.cyclone_net_physics_guided.CycloneNetPhysicsGuided")
    return CycloneNetPhysicsGuided(in_channels=in_ch, hidden_channels=hidden, dropout=dropout)


def _build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build the optimizer from configuration.
    """
    lr = float(cfg_get(cfg, "training.lr", 1e-4))
    wd = float(cfg_get(cfg, "training.weight_decay", 1e-4))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _physics_loss(cfg: Dict[str, Any], batch: Dict[str, Any], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Compute the physics-guided loss that makes CycloneNet physics-guided in practice.

    Terms (each gated by a weight in config training.physics.*; 0.0 disables it):

      1. lambda_prior_align — KL alignment pulling the learned FuelMap distribution
         toward the physical prior map P (SST-anomaly x wind x convergence, or total
         heat flux). This is a weak supervision toward a physically motivated
         energy-source map, not ground truth.

      2. lambda_forward — forward physical constraint. The model derives an energy
         score from the overlap of FuelMap and the prior map and maps it to a
         predicted dv24 (`dv24_forward_hat`). Supervising it against the real dv24
         forces "localized surface energy -> intensification" to be learned.

      3/4. lambda_tv / lambda_l1 — spatial smoothness / sparsity regularizers on the
         FuelMap so the localization is physically plausible (compact, contiguous).

      5. lambda_consistency — OPTIONAL, OFF by default. Equation consistency between
         vort/div recomputed from u/v and the stored vort/div channels. Both sides
         derive from the same input wind field, so this is near-degenerate and is
         documented as a weak representational regularizer, not a physical claim.

    When all weights are 0.0 this returns 0 and the model is a plain 3D-CNN.
    """
    device = batch["x"].device
    total = torch.zeros((), device=device)

    lam_prior = float(cfg_get(cfg, "training.physics.lambda_prior_align", 0.0))
    lam_fwd = float(cfg_get(cfg, "training.physics.lambda_forward", 0.0))
    lam_tv = float(cfg_get(cfg, "training.physics.lambda_tv", 0.0))
    lam_l1 = float(cfg_get(cfg, "training.physics.lambda_l1", 0.0))
    lam_cons = float(cfg_get(cfg, "training.physics.lambda_consistency", 0.0))

    fuelmap = outputs.get("fuelmap_logits", None)

    # 1. Prior alignment: FuelMap <- physical prior map.
    if fuelmap is not None and lam_prior > 0.0 and isinstance(batch.get("prior_map_t0"), torch.Tensor):
        total = total + lam_prior * kl_alignment_loss(fuelmap, batch["prior_map_t0"])

    # 2. Forward physical constraint: localized energy -> dv24.
    if lam_fwd > 0.0 and isinstance(outputs.get("dv24_forward_hat"), torch.Tensor):
        fwd_loss = masked_smooth_l1(
            outputs["dv24_forward_hat"].float().view(-1),
            batch["dv24"].float().view(-1),
            batch["dv24_mask"].float().view(-1),
        )
        total = total + lam_fwd * fwd_loss

    # 3/4. FuelMap regularizers.
    if fuelmap is not None and lam_tv > 0.0:
        total = total + lam_tv * tv_loss_2d(fuelmap)
    if fuelmap is not None and lam_l1 > 0.0:
        total = total + lam_l1 * l1_loss(fuelmap)

    # 5. Equation consistency (weak; off by default). Computed only over samples whose
    #    equation fields are present (eq_mask=1), using their physical grid spacing.
    if lam_cons > 0.0 and isinstance(batch.get("eq_mask"), torch.Tensor):
        from src.physics.physics_guided_losses import equation_consistency_loss

        eq_mask = batch["eq_mask"].float().view(-1)
        valid = eq_mask > 0.0
        if bool(valid.any()):
            dx = float(batch["dx_m"].float().view(-1)[valid].mean().item())
            dy = float(batch["dy_m"].float().view(-1)[valid].mean().item())
            cons = equation_consistency_loss(
                batch["u10_t0"][valid],
                batch["v10_t0"][valid],
                batch["vort_t0"][valid],
                batch["div_t0"][valid],
                dx=dx,
                dy=dy,
            )
            total = total + lam_cons * cons

    return total


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer],
) -> EpochResult:
    """
    Run one train or validation epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_cls = 0.0
    total_dv12 = 0.0
    total_dv24 = 0.0
    total_phys = 0.0
    n_batches = 0

    y_true_all = []
    y_prob_all = []

    for batch in loader:
        batch = _to_device(batch, device)
        y = batch["y"].float().view(-1)
        dv12 = batch["dv12"].float().view(-1)
        dv24 = batch["dv24"].float().view(-1)
        dv12_mask = batch["dv12_mask"].float().view(-1)
        dv24_mask = batch["dv24_mask"].float().view(-1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            prior = batch.get("prior_map_t0", None)
            outputs = model(batch["x"], prior_map_t0=prior) if isinstance(prior, torch.Tensor) else model(batch["x"])

            ri_logit = _extract_ri_logit(outputs)
            dv12_pred = _extract_regression(outputs, "dv12")
            dv24_pred = _extract_regression(outputs, "dv24")

            cls_loss = F.binary_cross_entropy_with_logits(ri_logit, y)
            dv12_loss = masked_smooth_l1(dv12_pred, dv12, dv12_mask)
            dv24_loss = masked_smooth_l1(dv24_pred, dv24, dv24_mask)
            phys_loss = _physics_loss(cfg, batch, outputs)

            w_cls = float(cfg_get(cfg, "training.lambda_ri", 1.0))
            w_dv12 = float(cfg_get(cfg, "training.lambda_dv12", 1.0))
            w_dv24 = float(cfg_get(cfg, "training.lambda_dv24", 1.0))

            loss = w_cls * cls_loss + w_dv12 * dv12_loss + w_dv24 * dv24_loss + phys_loss

            if is_train:
                loss.backward()
                max_grad_norm = float(cfg_get(cfg, "training.max_grad_norm", 0.0))
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_cls += float(cls_loss.detach().cpu())
        total_dv12 += float(dv12_loss.detach().cpu())
        total_dv24 += float(dv24_loss.detach().cpu())
        total_phys += float(phys_loss.detach().cpu())
        n_batches += 1

        y_true_all.append(y.detach().cpu().numpy())
        y_prob_all.append(torch.sigmoid(ri_logit).detach().cpu().numpy())

    y_true_np = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.float32)
    y_prob_np = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=np.float32)
    auc = _safe_auc(y_true_np, y_prob_np) if len(y_true_np) else None

    denom = max(n_batches, 1)
    return EpochResult(
        loss=total_loss / denom,
        cls_loss=total_cls / denom,
        dv12_loss=total_dv12 / denom,
        dv24_loss=total_dv24 / denom,
        phys_loss=total_phys / denom,
        roc_auc=auc,
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_name: str,
    metric_value: float,
) -> None:
    """
    Save a training checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metric_name": metric_name,
            "metric_value": metric_value,
        },
        path,
    )


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main training entrypoint.

    This function obeys the configured architecture:
    - checkpoints -> paths.checkpoints_dir
    - reports/history -> paths.results_dir
    """
    seed = int(cfg_get(cfg, "repro.seed", cfg_get(cfg, "training.seed", 42)))
    set_global_seed(seed)

    device = _resolve_device(cfg)

    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir", "./models/checkpoints")).resolve()
    results_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = _build_loaders(cfg)
    model = _build_model(cfg).to(device)
    optimizer = _build_optimizer(cfg, model)

    epochs = int(cfg_get(cfg, "training.epochs", 20))
    best_val_loss = float("inf")
    best_val_auc = float("-inf")

    history = []

    best_auc_path = ckpt_dir / "best_auc_model.pt"
    best_loss_path = ckpt_dir / "best_model.pt"
    threshold_path = ckpt_dir / "best_threshold.json"
    history_path = results_dir / "training_history.json"
    summary_path = results_dir / "train_summary.json"

    for epoch in range(1, epochs + 1):
        train_res = run_epoch(model, train_loader, device, cfg, optimizer)
        val_res = run_epoch(model, val_loader, device, cfg, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_res.loss,
            "train_cls_loss": train_res.cls_loss,
            "train_dv12_loss": train_res.dv12_loss,
            "train_dv24_loss": train_res.dv24_loss,
            "train_phys_loss": train_res.phys_loss,
            "val_loss": val_res.loss,
            "val_cls_loss": val_res.cls_loss,
            "val_dv12_loss": val_res.dv12_loss,
            "val_dv24_loss": val_res.dv24_loss,
            "val_phys_loss": val_res.phys_loss,
            "val_auc": val_res.roc_auc,
        }
        history.append(row)

        logger.info(
            "Epoch %d/%d | train_loss=%.5f | val_loss=%.5f | val_auc=%s",
            epoch,
            epochs,
            train_res.loss,
            val_res.loss,
            f"{val_res.roc_auc:.5f}" if val_res.roc_auc is not None else "NA",
        )

        if val_res.loss < best_val_loss:
            best_val_loss = val_res.loss
            _save_checkpoint(best_loss_path, model, optimizer, epoch, "val_loss", best_val_loss)

        if val_res.roc_auc is not None and val_res.roc_auc > best_val_auc:
            best_val_auc = val_res.roc_auc
            _save_checkpoint(best_auc_path, model, optimizer, epoch, "val_auc", best_val_auc)

    if best_auc_path.exists():
        checkpoint = torch.load(best_auc_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"], strict=False)

    y_true = []
    y_prob = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, device)
            prior = batch.get("prior_map_t0", None)
            outputs = model(batch["x"], prior_map_t0=prior) if isinstance(prior, torch.Tensor) else model(batch["x"])
            ri_logit = _extract_ri_logit(outputs)
            y_true.append(batch["y"].float().view(-1).cpu().numpy())
            y_prob.append(torch.sigmoid(ri_logit).cpu().numpy())

    y_true_np = np.concatenate(y_true) if y_true else np.array([], dtype=np.float32)
    y_prob_np = np.concatenate(y_prob) if y_prob else np.array([], dtype=np.float32)

    # Threshold selection (validation only). Default policy honours the project's
    # forensic high-recall mandate: pick the highest-precision threshold that still
    # reaches training.eval_target_recall. Configurable via training.threshold_method.
    from src.utils.thresholding import ThresholdConfig, select_threshold

    thr_cfg = ThresholdConfig(
        method=str(cfg_get(cfg, "training.threshold_method", "precision_at_recall")),
        min_recall=float(cfg_get(cfg, "training.eval_target_recall", 0.90)),
        fallback_threshold=0.5,
    )
    if len(y_true_np):
        selected_threshold, thr_metrics = select_threshold(y_true_np, y_prob_np, thr_cfg)
    else:
        selected_threshold, thr_metrics = 0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    with threshold_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(selected_threshold),
                "method": thr_cfg.method,
                "min_recall": thr_cfg.min_recall,
                "val_metrics_at_threshold": thr_metrics,
            },
            f,
            indent=2,
        )

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "best_val_auc": None if best_val_auc == float("-inf") else best_val_auc,
        "selected_threshold": float(selected_threshold),
        "checkpoint_dir": rel_to_root(ckpt_dir),
        "results_dir": rel_to_root(results_dir),
        "best_auc_checkpoint": rel_to_root(best_auc_path),
        "best_loss_checkpoint": rel_to_root(best_loss_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary