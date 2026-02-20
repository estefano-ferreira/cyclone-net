"""
CycloneNet Trainer (RI-only).

International-standard, config-driven trainer for imbalanced binary
classification (Rapid Intensification / RI).

Key guarantees:
- No hard-coded hyperparameters: all values come from config.yaml (with safe defaults).
- Validation-only threshold selection (frozen for test).
- Optional Platt scaling (validation-only fit; applied to val/test for calibrated decisions).
- Supports BCE-with-logits (with optional label smoothing + pos_weight) OR focal loss.
- Reproducible: seeds + deterministic settings.

Author: Estefano Senhor Ferreira
License: CC BY-NC 4.0
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from src.models.cyclone_net_ri_only import CycloneNetRIOnly
from src.losses.label_smoothing import LabelSmoothingBCEWithLogitsLoss
from src.data.dataset import PhysicsDataset
from src.utils.calibration import fit_platt_scaler
from src.utils.thresholding import ThresholdConfig, select_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get nested config values via dot path, with default fallback."""
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def set_seed(seed: int) -> None:
    """Set deterministic seeds (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism: note that some CUDA ops may still be nondeterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLossWithLogits(nn.Module):
    """Binary focal loss applied to logits.

    Args:
        alpha: Positive-class weight in [0,1]. Common value: 0.25.
        gamma: Focusing parameter. Common value: 2.0.
        label_smoothing: Optional target smoothing in [0, 0.5).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + 0.5 * eps

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class CycloneTrainer:
    """RI-only trainer.

    This class is intentionally single-task (RI classification) to keep
    the scientific pipeline auditable.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
    ) -> None:
        self.config = config

        # Device
        device_setting = str(
            cfg_get(config, "training.device", "auto")).lower()
        if device_setting == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_setting)
        logger.info("Device: %s", self.device)

        # Seed
        set_seed(int(cfg_get(config, "training.seed", 42)))

        # Model
        self.model = CycloneNetRIOnly(config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model parameters: %s", f"{n_params:,}")

        # Loss (config-driven)
        self.loss_name = str(cfg_get(config, "training.loss", "bce")).lower()
        self.label_smoothing = float(
            cfg_get(config, "training.label_smoothing", 0.0))
        self.lambda_ri = float(cfg_get(config, "training.lambda_ri", 1.0))

        self.ri_pos_weight = float(
            cfg_get(config, "training.ri_pos_weight", 1.0))

        if self.loss_name == "focal":
            focal_gamma = float(cfg_get(config, "training.focal_gamma", cfg_get(
                config, "training.localizer.focal_gamma", 2.0)))
            focal_alpha = float(cfg_get(config, "training.focal_alpha", cfg_get(
                config, "training.localizer.focal_alpha", 0.25)))
            self.ri_loss_fn = FocalLossWithLogits(
                alpha=focal_alpha,
                gamma=focal_gamma,
                label_smoothing=self.label_smoothing,
            ).to(self.device)
            logger.info(
                "Loss: focal | alpha=%.3f gamma=%.3f smoothing=%.3f (pos_weight ignored; configured=%.2f) | lambda=%.3f",
                focal_alpha, focal_gamma, self.label_smoothing, self.ri_pos_weight, self.lambda_ri
            )
        else:
            self.loss_name = "bce"
            self.ri_loss_fn = LabelSmoothingBCEWithLogitsLoss(
                smoothing=self.label_smoothing,
                pos_weight=torch.tensor(
                    [self.ri_pos_weight], device=self.device),
            ).to(self.device)
            logger.info(
                "Loss: bce | pos_weight=%.2f smoothing=%.3f | lambda=%.3f",
                self.ri_pos_weight, self.label_smoothing, self.lambda_ri
            )

        # Optimizer
        lr = float(cfg_get(config, "training.learning_rate", 1e-3))
        wd = float(cfg_get(config, "training.weight_decay", 1e-5))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd)

        # AMP / Gradient handling
        self.use_amp = bool(
            cfg_get(config, "training.use_amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            logger.info("AMP: enabled")

        self.grad_clip_norm = float(
            cfg_get(config, "training.gradient_clip_norm", 1.0))
        self.grad_accum_steps = int(
            cfg_get(config, "training.gradient_accumulation_steps", 1))

        # Thresholding config (validation-only selection)
        # Prefer evaluation.thresholding, fallback to thresholding section if present.
        thr_cfg = cfg_get(config, "evaluation.thresholding", None) or cfg_get(
            config, "thresholding", {}) or {}
        self.thresholding_enabled = bool(thr_cfg.get("enabled", True))
        self.thresholding_method = str(
            thr_cfg.get("method", "precision_at_recall"))
        self.thresholding_min_recall = float(thr_cfg.get("min_recall", 0.90))
        self.thresholding_fallback = float(thr_cfg.get(
            "fallback_threshold", cfg_get(config, "training.pred_threshold", 0.5)))
        self.selection_metric = str(thr_cfg.get("selection_metric", cfg_get(
            config, "selection.metric", "pr_auc"))).lower()

        # Calibration config (validation-only fit)
        cal_cfg = cfg_get(config, "calibration", {}) or {}
        self.calibration_enabled = bool(cal_cfg.get("enabled", False))
        self.calibration_method = str(cal_cfg.get("method", "platt")).lower()
        self.calibration_max_iter = int(cal_cfg.get("max_iter", 200))
        self.calibration_lr = float(cal_cfg.get("lr", 0.01))

        # Scheduler
        self.scheduler: Optional[object] = None
        self.scheduler_type = str(
            cfg_get(config, "training.scheduler.type", "none")).lower()
        if self.scheduler_type == "reduce_on_plateau":
            s = cfg_get(
                config, "training.scheduler.reduce_on_plateau", {}) or {}
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=str(s.get("mode", "max")),
                factor=float(s.get("factor", 0.8)),
                patience=int(s.get("patience", 10)),
                threshold=float(s.get("threshold", 1e-3)),
                cooldown=int(s.get("cooldown", 0)),
                min_lr=float(s.get("min_lr", 1e-6)),
            )
        elif self.scheduler_type == "onecycle":
            self.scheduler = None  # created in train()
        else:
            self.scheduler = None
            self.scheduler_type = "none"

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if bool(cfg_get(config, "training.tensorboard", True)):
            try:
                tb_root = Path(
                    cfg_get(config, "paths.logs", tempfile.gettempdir()))
                tb_root = tb_root / "tensorboard"
                tb_root.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(tb_root))
                logger.info("TensorBoard: %s", tb_root)
            except Exception as e:
                logger.warning("TensorBoard init failed (%s). Disabled.", e)
                self.writer = None

        # Datasets (allow external override)
        self.train_dataset = train_dataset if train_dataset is not None else PhysicsDataset(
            split="train")
        self.val_dataset = val_dataset if val_dataset is not None else PhysicsDataset(
            split="val")

        if len(self.val_dataset) == 0:
            logger.warning(
                "Validation set is empty. Check your split configuration.")

        # Paths
        self.checkpoint_dir = Path(
            cfg_get(config, "paths.checkpoints", "./models/checkpoints")).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.artifacts_path = self.checkpoint_dir / "best_model_ri_artifacts.json"

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------
    def _forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        self.optimizer.zero_grad(set_to_none=True)
        accum = 0

        for batch_idx, batch in enumerate(loader):
            inputs = batch["input"].to(self.device, non_blocking=True)
            y = batch["ri_label"].to(self.device, non_blocking=True).float()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = self._forward_logits(inputs)
                    loss = self.lambda_ri * self.ri_loss_fn(logits, y)
            else:
                logits = self._forward_logits(inputs)
                loss = self.lambda_ri * self.ri_loss_fn(logits, y)

            if not torch.isfinite(loss):
                logger.warning(
                    "Non-finite loss at batch %d (epoch %d). Skipping step.", batch_idx, epoch)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss / self.grad_accum_steps

            if self.use_amp:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum += 1
            if accum >= self.grad_accum_steps:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()

                if self.scheduler_type == "onecycle" and self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)
                accum = 0

            total_loss += float(loss.item()) * self.grad_accum_steps

        # Flush remaining gradients
        if accum > 0:
            if self.use_amp:
                assert self.scaler is not None
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            if self.scheduler_type == "onecycle" and self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def predict_logits(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Return (logits, y_true) arrays for a loader."""
        self.model.eval()
        logits_list: list[float] = []
        y_list: list[float] = []

        for batch in loader:
            inputs = batch["input"].to(self.device, non_blocking=True)
            y = batch["ri_label"].to(self.device, non_blocking=True).float()
            logits = self._forward_logits(inputs)
            logits_list.extend(
                logits.detach().cpu().numpy().reshape(-1).tolist())
            y_list.extend(y.detach().cpu().numpy().reshape(-1).tolist())

        return np.asarray(logits_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)

    def _metrics_from_scores(self, y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
        y_pred = (y_score >= threshold).astype(np.int32)
        out: Dict[str, float] = {}
        out["auc"] = float(roc_auc_score(y_true, y_score)) if len(
            np.unique(y_true)) >= 2 else 0.0
        out["pr_auc"] = float(average_precision_score(
            y_true, y_score)) if len(np.unique(y_true)) >= 2 else 0.0
        out["precision"] = float(precision_score(
            y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["brier"] = float(brier_score_loss(y_true, y_score))
        out["threshold"] = float(threshold)
        return out

    def validate(self, loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validation pass with (optional) calibration and threshold selection.

        Returns:
            metrics: dict with auc, pr_auc, precision, recall, f1, brier, threshold
            artifacts: dict with calibration + thresholding metadata for reproducibility
        """
        logits, y_true = self.predict_logits(loader)
        probs = 1.0 / (1.0 + np.exp(-logits))

        artifacts: Dict[str, Any] = {
            "thresholding": {},
            "calibration": {},
        }

        # Calibration (fit on validation only)
        if self.calibration_enabled:
            if self.calibration_method != "platt":
                logger.warning(
                    "Unknown calibration.method='%s'. Disabling calibration.", self.calibration_method)
            else:
                # Correct call: fit_platt_scaler expects only logits and y_true
                scaler = fit_platt_scaler(logits, y_true)
                probs = scaler.predict_from_logits(logits).astype(np.float32)
                artifacts["calibration"] = {
                    "enabled": True,
                    "method": "platt",
                    "a": float(scaler.a),
                    "b": float(scaler.b),
                    # stored for info
                    "max_iter": int(self.calibration_max_iter),
                    # stored for info
                    "lr": float(self.calibration_lr),
                }
        else:
            artifacts["calibration"] = {"enabled": False}

        # Thresholding (validation-only)
        if self.thresholding_enabled:
            tcfg = ThresholdConfig(
                method=self.thresholding_method,
                min_recall=self.thresholding_min_recall,
                fallback_threshold=self.thresholding_fallback,
            )
            thr, thr_meta = select_threshold(
                y_true=y_true, y_score=probs, cfg=tcfg)
            artifacts["thresholding"] = {
                "enabled": True,
                "config": asdict(tcfg),
                "selected_threshold": float(thr),
                "meta": {k: float(v) for k, v in thr_meta.items()},
            }
        else:
            thr = float(self.thresholding_fallback)
            artifacts["thresholding"] = {
                "enabled": False, "threshold": float(thr)}

        metrics = self._metrics_from_scores(y_true, probs, float(thr))
        return metrics, artifacts

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        if epochs is None:
            epochs = int(cfg_get(self.config, "training.epochs", 50))
        patience = int(
            cfg_get(self.config, "training.early_stopping_patience", 10))

        fast_mode = os.getenv("CYCLONET_FAST") == "1"
        if fast_mode:
            logger.info("FAST MODE enabled (CYCLONET_FAST=1)")
            epochs = min(int(epochs), 5)

        # DataLoader config
        num_workers = int(
            cfg_get(self.config, "training.dataloader_num_workers", 0))
        pin_memory = self.device.type == "cuda"
        persistent_workers = bool(num_workers > 0)

        # Sampler (optional; disable for calibration stability)
        sampler_cfg = cfg_get(self.config, "training.sampler", {}) or {}
        sampler_type = str(sampler_cfg.get("type", "none")).lower()
        sampler = None
        shuffle = True
        if sampler_type == "weighted":
            labels = np.array(
                [int(self.train_dataset[i]["ri_label"])
                 for i in range(len(self.train_dataset))],
                dtype=np.int64,
            )
            counts = np.bincount(labels, minlength=2)
            counts = np.maximum(counts, 1)
            class_weights = 1.0 / counts
            sample_weights = class_weights[labels]
            replacement = bool(sampler_cfg.get("replacement", True))
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=replacement,
            )
            shuffle = False
            logger.info("Sampler: weighted | counts=%s | replacement=%s",
                        counts.tolist(), replacement)

        batch_size = int(cfg_get(self.config, "training.batch_size", 32))

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        logger.info("Train samples: %d | Val samples: %d", len(
            self.train_dataset), len(self.val_dataset))

        # OneCycleLR
        if self.scheduler_type == "onecycle":
            one = cfg_get(self.config, "training.scheduler.onecycle", {}) or {}
            steps_per_epoch = (
                len(train_loader) + self.grad_accum_steps - 1) // self.grad_accum_steps
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=float(
                    cfg_get(self.config, "training.learning_rate", 1e-3)),
                steps_per_epoch=steps_per_epoch,
                epochs=int(epochs),
                pct_start=float(one.get("pct_start", 0.3)),
                anneal_strategy=str(one.get("anneal_strategy", "cos")),
                div_factor=float(one.get("div_factor", 25.0)),
                final_div_factor=float(one.get("final_div_factor", 10000.0)),
            )
            logger.info("Scheduler: OneCycleLR | steps/epoch=%d",
                        steps_per_epoch)

        # Selection metric
        # - "pr_auc" is recommended for heavy imbalance (scientifically appropriate).
        # - Still log ROC-AUC for completeness.
        best_score = -1.0
        patience_counter = 0
        history: list[Dict[str, float]] = []

        for epoch in range(1, int(epochs) + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics, val_artifacts = self.validate(val_loader)

            history.append({**val_metrics, "train_loss": float(train_loss)})

            lr_now = float(self.optimizer.param_groups[0]["lr"])

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Val/AUC", val_metrics["auc"], epoch)
                self.writer.add_scalar(
                    "Val/PR_AUC", val_metrics["pr_auc"], epoch)
                self.writer.add_scalar(
                    "Val/Precision", val_metrics["precision"], epoch)
                self.writer.add_scalar(
                    "Val/Recall", val_metrics["recall"], epoch)
                self.writer.add_scalar("Val/F1", val_metrics["f1"], epoch)
                self.writer.add_scalar(
                    "Val/Brier", val_metrics["brier"], epoch)
                self.writer.add_scalar(
                    "Val/Threshold", val_metrics["threshold"], epoch)
                self.writer.add_scalar("LR", lr_now, epoch)

            logger.info(
                "Epoch [%3d/%d] Loss=%.6f | Val PR-AUC=%.4f AUC=%.4f | P=%.4f R=%.4f F1=%.4f | Brier=%.4f | thr=%.3f | LR=%.2e",
                epoch, int(epochs), train_loss,
                val_metrics["pr_auc"], val_metrics["auc"],
                val_metrics["precision"], val_metrics["recall"], val_metrics["f1"],
                val_metrics["brier"], val_metrics["threshold"], lr_now,
            )

            # Scheduler step (epoch-level)
            if self.scheduler_type == "reduce_on_plateau" and self.scheduler is not None:
                # Use selection metric if possible; default to PR-AUC.
                plateau_metric = val_metrics.get(
                    self.selection_metric, val_metrics["pr_auc"])
                self.scheduler.step(float(plateau_metric))

            # Best model selection
            current = float(val_metrics.get(
                self.selection_metric, val_metrics["pr_auc"]))
            if current > best_score:
                best_score = current
                patience_counter = 0

                best_path = self.checkpoint_dir / "best_model_ri.pt"
                torch.save(self.model.state_dict(), best_path)

                # Persist artifacts for reproducibility
                payload = {
                    "selection_metric": self.selection_metric,
                    "best_score": float(best_score),
                    "val_metrics": val_metrics,
                    "val_artifacts": val_artifacts,
                    "epoch": int(epoch),
                }
                with open(self.artifacts_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

                logger.info("  → New best %s: %.4f | saved: %s",
                            self.selection_metric, best_score, best_path)
                logger.info("  → Artifacts saved: %s", self.artifacts_path)
            else:
                patience_counter += 1

            if patience_counter >= patience and not fast_mode:
                logger.info(
                    "Early stopping at epoch %d (patience=%d).", epoch, patience)
                break

        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_model_ri.pt"
        torch.save(self.model.state_dict(), last_path)
        logger.info("Training finished. Saved last: %s", last_path)

        if self.writer is not None:
            self.writer.close()

        return {
            "best_score": float(best_score),
            "selection_metric": self.selection_metric,
            "artifacts_path": str(self.artifacts_path),
            "history_len": len(history),
        }
