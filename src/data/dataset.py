from __future__ import annotations

"""
CycloneNet — PhysicsDataset (auditable, leakage-safe, config-driven).

Reads:
- data/interim/{event_id}.npy  (H,W,T,C_total)
- data/interim/{event_id}.json (meta including channel names)
- data/interim/{event_id}_lats.npy / _lons.npy (H,W) for dx/dy estimation
- optional: data/interim/{event_id}_fuel_potential.npy (physical prior)

Uses:
- data/normalized/splits.csv (event_id, split)
- data/normalized/normalization_stats.json (train-only stats for model inputs)

Key properties:
- Selects input channels strictly by name from meta["channels"].
- Optional anti-leakage: total_heat_flux is loss-only -> excluded from model inputs.
- Produces masks for missing dv12/dv24 targets.
- Provides optional physics-guided tensors (prior map + equation consistency fields).

Expected model input tensor:
- x: (C,T,H,W) float32, normalized

Returned dict keys (minimum):
- x, y, dv12, dv24, dv12_mask, dv24_mask, event_id

Optional keys (physics-guided):
- prior_map_t0, prior_mask
- u10_t0, v10_t0, vort_t0, div_t0, dx_m, dy_m, eq_mask
- total_heat_flux_t0, total_heat_flux_mask
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def _p(p: Union[str, Path]) -> Path:
    return Path(str(p)).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        if v is None:
            return float("nan")
        if isinstance(v, str) and v.strip() == "":
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _mask_scalar(x: float) -> float:
    return 1.0 if np.isfinite(x) else 0.0


def _normalize_prob_map(m: np.ndarray) -> np.ndarray:
    m = np.array(m, dtype=np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    m = np.maximum(m, 0.0)
    s = float(m.sum())
    if s <= 0.0:
        return np.zeros_like(m, dtype=np.float32)
    return (m / s).astype(np.float32)


def _estimate_dx_dy_meters(lats_hw: np.ndarray, lons_hw: np.ndarray) -> Tuple[float, float]:
    lat0 = float(np.nanmean(lats_hw))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * float(np.cos(np.deg2rad(lat0)))

    dlat = float(np.nanmedian(np.abs(lats_hw[1:, :] - lats_hw[:-1, :])))
    dlon = float(np.nanmedian(np.abs(lons_hw[:, 1:] - lons_hw[:, :-1])))

    dy = max(1e-6, dlat * m_per_deg_lat)
    dx = max(1e-6, dlon * m_per_deg_lon)
    return dx, dy


class PhysicsDataset(Dataset):
    """
    Config-driven dataset.

    Always call:
        PhysicsDataset(cfg, split="train")
        PhysicsDataset(cfg, split="val")
        PhysicsDataset(cfg, split="test")
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str,
        splits_csv: Optional[Union[str, Path]] = None,
        interim_dir: Optional[Union[str, Path]] = None,
        norm_stats_json: Optional[Union[str, Path]] = None,
        augment: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.split = split

        # Paths
        self.interim_dir = _p(interim_dir or cfg_get(
            cfg, "paths.interim_data", "./data/interim"))
        self.splits_csv = _p(splits_csv or cfg_get(
            cfg, "paths.splits_csv", "./data/normalized/splits.csv"))
        self.stats_path = _p(norm_stats_json or cfg_get(
            cfg, "paths.normalization_stats", "./data/normalized/normalization_stats.json"))

        if not self.splits_csv.exists():
            raise FileNotFoundError(
                f"Splits CSV not found: {self.splits_csv}. Run: python run.py normalize")
        if not self.stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {self.stats_path}. Run: python run.py normalize")

        df = pd.read_csv(self.splits_csv)
        if "event_id" not in df.columns or "split" not in df.columns:
            raise ValueError(
                "splits_csv must contain columns: event_id, split")
        df = df[df["split"] == split].copy()
        self.event_ids: List[str] = df["event_id"].astype(str).tolist()
        if not self.event_ids:
            raise RuntimeError(
                f"No events for split '{split}' in {self.splits_csv}")

        # Input channel names
        input_names = cfg_get(cfg, "model.input_channels_names", None)
        if not input_names:
            raise ValueError(
                "config.yaml must define model.input_channels_names")
        input_names = list(input_names)

        # Anti-leakage: total heat flux loss-only
        self.thf_name = str(cfg_get(
            cfg, "physics_guided.losses.total_heat_flux_channel_name", "total_heat_flux_Wpm2"))
        exclude_thf = bool(
            cfg_get(cfg, "physics_guided.losses.exclude_total_heat_flux_from_input", True))
        if exclude_thf and self.thf_name in input_names:
            input_names = [c for c in input_names if c != self.thf_name]
            logger.info(
                f"Removed '{self.thf_name}' from model inputs (physics-loss-only).")
        self.input_channels_names = input_names

        # Normalization stats must match input names exactly
        stats = _load_json(self.stats_path)
        stats_channels = list(stats.get("channels", []))
        if stats_channels != self.input_channels_names:
            raise ValueError(
                "Normalization stats channels do not match model.input_channels_names.\n"
                f"model.input_channels_names: {self.input_channels_names}\n"
                f"stats.channels:           {stats_channels}\n"
                "Fix: run `python run.py normalize`."
            )
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(
            stats["std"], dtype=torch.float32).clamp(min=1e-6)

        self.augment = bool(augment) and split == "train"
        self.seed = int(cfg_get(cfg, "splits.seed", 1337))

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eid = self.event_ids[idx]

        cube_path = self.interim_dir / f"{eid}.npy"
        meta_path = self.interim_dir / f"{eid}.json"
        lats_path = self.interim_dir / f"{eid}_lats.npy"
        lons_path = self.interim_dir / f"{eid}_lons.npy"
        fp_path = self.interim_dir / f"{eid}_fuel_potential.npy"

        if not cube_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing cube/meta for event_id={eid}")

        meta = _load_json(meta_path)
        chs = list(meta.get("channels", []))
        if not chs:
            raise KeyError(f"Meta missing channels list: {meta_path}")

        cube = np.load(cube_path).astype(np.float32)  # (H,W,T,C_total)

        # Input channel indices by name
        missing = [c for c in self.input_channels_names if c not in chs]
        if missing:
            raise KeyError(f"Event {eid} missing required channels: {missing}")

        idx_in = [chs.index(c) for c in self.input_channels_names]
        x_np = cube[:, :, :, idx_in]  # (H,W,T,Cin)
        x = torch.from_numpy(x_np).permute(
            3, 2, 0, 1).contiguous()  # (C,T,H,W)

        # Normalize x only
        mean = self.mean.view(-1, 1, 1, 1)
        std = self.std.view(-1, 1, 1, 1)
        x = (x - mean) / std
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Targets
        y = torch.tensor(float(int(meta.get("ri_label", 0))),
                         dtype=torch.float32)

        dv12_val = _safe_float(meta.get("dv12_kt"))
        dv24_val = _safe_float(meta.get("dv24_kt"))
        dv12 = torch.tensor(dv12_val if np.isfinite(
            dv12_val) else 0.0, dtype=torch.float32)
        dv24 = torch.tensor(dv24_val if np.isfinite(
            dv24_val) else 0.0, dtype=torch.float32)
        dv12_mask = torch.tensor(_mask_scalar(dv12_val), dtype=torch.float32)
        dv24_mask = torch.tensor(_mask_scalar(dv24_val), dtype=torch.float32)

        # t0 convention: index 0
        t0 = 0
        H, W = cube.shape[0], cube.shape[1]

        # Prior map (fuel potential)
        prior_map_t0 = torch.zeros((1, H, W), dtype=torch.float32)
        prior_mask = torch.tensor(0.0, dtype=torch.float32)
        if fp_path.exists():
            fp = np.load(fp_path).astype(np.float32)
            if fp.ndim == 2:
                p = fp
            elif fp.ndim == 3:
                p = fp[:, :, t0] if fp.shape[2] > 1 else fp[:, :, 0]
            else:
                p = None
            if p is not None:
                prior_map_t0 = torch.from_numpy(
                    _normalize_prob_map(p)).unsqueeze(0)
                prior_mask = torch.tensor(1.0, dtype=torch.float32)

        # Equation fields at t0 if present
        def _ch(name: str) -> Optional[np.ndarray]:
            if name not in chs:
                return None
            return cube[:, :, t0, chs.index(name)].astype(np.float32)

        u_t0 = _ch("u10_mps")
        v_t0 = _ch("v10_mps")
        vort_t0 = _ch("vort_1ps")
        div_t0 = _ch("div_1ps")

        eq_mask = torch.tensor(0.0, dtype=torch.float32)
        u10_t0 = torch.zeros((1, H, W), dtype=torch.float32)
        v10_t0 = torch.zeros((1, H, W), dtype=torch.float32)
        vort_out = torch.zeros((1, H, W), dtype=torch.float32)
        div_out = torch.zeros((1, H, W), dtype=torch.float32)
        dx_m = torch.tensor(0.0, dtype=torch.float32)
        dy_m = torch.tensor(0.0, dtype=torch.float32)

        if u_t0 is not None and v_t0 is not None and vort_t0 is not None and div_t0 is not None:
            u10_t0 = torch.from_numpy(u_t0).unsqueeze(0)
            v10_t0 = torch.from_numpy(v_t0).unsqueeze(0)
            vort_out = torch.from_numpy(vort_t0).unsqueeze(0)
            div_out = torch.from_numpy(div_t0).unsqueeze(0)

            if lats_path.exists() and lons_path.exists():
                lats_hw = np.load(lats_path).astype(np.float32)
                lons_hw = np.load(lons_path).astype(np.float32)
                dx, dy = _estimate_dx_dy_meters(lats_hw, lons_hw)
                dx_m = torch.tensor(float(dx), dtype=torch.float32)
                dy_m = torch.tensor(float(dy), dtype=torch.float32)

            eq_mask = torch.tensor(1.0, dtype=torch.float32)

        # total heat flux (loss-only) at t0 if present
        thf = torch.zeros((1, H, W), dtype=torch.float32)
        thf_mask = torch.tensor(0.0, dtype=torch.float32)
        if self.thf_name in chs:
            thf_np = cube[:, :, t0, chs.index(
                self.thf_name)].astype(np.float32)
            thf = torch.from_numpy(thf_np).unsqueeze(0)
            thf_mask = torch.tensor(1.0, dtype=torch.float32)

        return {
            "event_id": eid,
            "x": x,
            "y": y,
            "dv12": dv12,
            "dv24": dv24,
            "dv12_mask": dv12_mask,
            "dv24_mask": dv24_mask,
            "prior_map_t0": prior_map_t0,
            "prior_mask": prior_mask,
            "u10_t0": u10_t0,
            "v10_t0": v10_t0,
            "vort_t0": vort_out,
            "div_t0": div_out,
            "dx_m": dx_m,
            "dy_m": dy_m,
            "eq_mask": eq_mask,
            "total_heat_flux_t0": thf,
            "total_heat_flux_mask": thf_mask,
        }
