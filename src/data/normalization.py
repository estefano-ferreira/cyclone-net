from __future__ import annotations

"""
CycloneNet — scientifically strict normalization and training-eligibility gate.

This module does NOT repair or impute missing data. Its role is to:

1. Audit each preprocessed event for scientific training eligibility.
2. Build valid/rejected manifests.
3. Compute normalization statistics using TRAIN split only and only for
   scientifically eligible events.
4. Preserve full auditability through explicit rejection reasons.

Scientific policy
-----------------
- No artificial NaN manipulation is allowed.
- Events with physically invalid or scientifically non-auditable core inputs
  are rejected from training.
- Auxiliary channels such as heat fluxes and fuel priors are tracked, but they
  do not define core input validity for the released model.
- Normalization statistics are computed only from train events that passed the
  eligibility gate.

Core input channels for the released model
------------------------------------------
These channels must be physically valid and sufficiently complete:
- sst_K
- mslp_Pa
- u10_mps
- v10_mps
- wind_mps
- vort_1ps
- div_1ps
- grad_mslp_Pa_per_m
- sst_anom_K
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import cfg_get, load_config

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore

logger = logging.getLogger(__name__)


CORE_CHANNELS_DEFAULT = [
    "sst_K",
    "mslp_Pa",
    "u10_mps",
    "v10_mps",
    "wind_mps",
    "vort_1ps",
    "div_1ps",
    "grad_mslp_Pa_per_m",
    "sst_anom_K",
]

VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class EligibilityThresholds:
    """Eligibility thresholds for training validity."""

    sst_range_K: Tuple[float, float]
    msl_range_Pa: Tuple[float, float]
    wind_abs_max_mps: float
    max_core_nan_fraction_per_channel: float
    max_core_nan_fraction_total: float
    require_fuel_prior_for_training: bool


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nan_fraction(arr: np.ndarray) -> float:
    return float(np.mean(~np.isfinite(arr)))


def _resolve_input_names(cfg: Dict[str, Any]) -> List[str]:
    """
    Resolve ordered model input channel names and enforce anti-leakage rules.

    If total heat flux is configured as loss-only, it must not be normalized
    as a model input.
    """
    input_names = cfg_get(cfg, "model.input_channels_names", None)
    if not input_names:
        raise ValueError("config.yaml must define model.input_channels_names.")
    input_names = list(input_names)

    exclude_thf = bool(
        cfg_get(cfg, "physics_guided.losses.exclude_total_heat_flux_from_input", True)
    )
    thf_name = str(
        cfg_get(
            cfg,
            "physics_guided.losses.total_heat_flux_channel_name",
            "total_heat_flux_Wpm2",
        )
    )

    if exclude_thf and thf_name in input_names:
        input_names = [c for c in input_names if c != thf_name]
        logger.info(
            "Removed '%s' from normalization inputs because it is configured as loss-only.",
            thf_name,
        )

    if not input_names:
        raise ValueError(
            "Resolved model.input_channels_names is empty after anti-leakage filtering."
        )
    return input_names


def _resolve_core_channels(cfg: Dict[str, Any]) -> List[str]:
    """
    Resolve the core released-model channels that determine scientific training validity.

    By default, this is the same set as the model input channels after anti-leakage
    filtering. If the config does not define them explicitly, the released 9-channel
    default is used and then filtered by the input channel list.
    """
    input_names = _resolve_input_names(cfg)
    configured = cfg_get(cfg, "normalization.core_channels", None)

    if configured:
        core = list(configured)
    else:
        core = [c for c in CORE_CHANNELS_DEFAULT if c in input_names]

    if not core:
        raise ValueError(
            "Resolved core channel list is empty. Check model.input_channels_names "
            "and normalization.core_channels."
        )
    return core


def _resolve_thresholds(cfg: Dict[str, Any]) -> EligibilityThresholds:
    """
    Resolve scientific eligibility thresholds from config.

    These thresholds are intentionally strict. They define which events are
    suitable for the released model training set.
    """
    qc = cfg_get(cfg, "data.qc", {})
    norm_cfg = cfg_get(cfg, "normalization", {})

    return EligibilityThresholds(
        sst_range_K=tuple(qc.get("sst_range_K", [240.0, 330.0])),
        msl_range_Pa=tuple(qc.get("msl_range_Pa", [80000.0, 110000.0])),
        wind_abs_max_mps=float(qc.get("wind_abs_max_mps", 80.0)),
        max_core_nan_fraction_per_channel=float(
            norm_cfg.get("max_core_nan_fraction_per_channel", 0.05)
        ),
        max_core_nan_fraction_total=float(
            norm_cfg.get("max_core_nan_fraction_total", 0.02)
        ),
        require_fuel_prior_for_training=bool(
            norm_cfg.get("require_fuel_prior_for_training", False)
        ),
    )


def _channel_indices(meta_channels: List[str], requested: List[str]) -> Dict[str, int]:
    """Return a mapping from channel name to channel index."""
    missing = [c for c in requested if c not in meta_channels]
    if missing:
        raise KeyError(f"Missing required channels in metadata: {missing}")
    return {c: meta_channels.index(c) for c in requested}


def _check_finite_range(
    arr: np.ndarray,
    value_range: Tuple[float, float],
) -> bool:
    """Return True only if all finite values lie inside the specified range."""
    mask = np.isfinite(arr)
    if not np.any(mask):
        return False
    return bool(np.all((arr[mask] >= value_range[0]) & (arr[mask] <= value_range[1])))


def _check_finite_abs_max(arr: np.ndarray, abs_max: float) -> bool:
    """Return True only if all finite absolute values are below or equal to abs_max."""
    mask = np.isfinite(arr)
    if not np.any(mask):
        return False
    return bool(np.all(np.abs(arr[mask]) <= abs_max))


def _safe_event_id_from_path(path: Path) -> str:
    return path.stem


def _validate_event_artifacts(
    event_id: str,
    interim_dir: Path,
) -> Tuple[bool, List[str]]:
    """
    Verify that the minimal artifact set exists for an event.

    Required artifacts:
    - event cube
    - metadata JSON
    - latitude grid
    - longitude grid
    """
    reasons: List[str] = []

    required_paths = [
        interim_dir / f"{event_id}.json",
        interim_dir / f"{event_id}.npy",
        interim_dir / f"{event_id}_lats.npy",
        interim_dir / f"{event_id}_lons.npy",
    ]
    for p in required_paths:
        if not p.exists():
            reasons.append(f"missing_artifact:{p.name}")

    return len(reasons) == 0, reasons


def audit_event_for_training(
    event_id: str,
    interim_dir: Path,
    core_channels: List[str],
    thresholds: EligibilityThresholds,
) -> Dict[str, Any]:
    """
    Audit a single event for scientific training eligibility.

    Scientific policy
    -----------------
    This function does not repair, impute, interpolate, or otherwise manipulate
    missing values. Its purpose is only to decide whether an event is suitable
    for scientific training under the released CycloneNet assumptions.

    Eligibility requirements
    ------------------------
    An event is considered training-eligible only if:
    - all required artifacts exist;
    - metadata is internally consistent;
    - temporal integrity is valid;
    - the storm identifier (SID) is present;
    - all released-model core channels exist in the cube metadata;
    - all released-model core channels are fully finite;
    - all released-model core channels remain within physically plausible ranges
      where applicable;
    - aggregate NaN checks also remain within configured thresholds
      (kept for auditability, even though full finiteness is required);
    - the fuel prior is valid only if requested by configuration.

    Notes
    -----
    - Core channel validity determines training eligibility.
    - Fuel-prior validity is tracked separately because the released model treats
      it as auxiliary weak supervision rather than physical ground truth.
    - Rejection reasons are explicit to preserve full auditability.
    """
    eligible = True
    reasons: List[str] = []

    artifacts_ok, artifact_reasons = _validate_event_artifacts(event_id, interim_dir)
    if not artifacts_ok:
        return {
            "event_id": event_id,
            "train_eligible": False,
            "fuel_prior_valid": False,
            "rejection_reasons": artifact_reasons,
            "core_nan_fraction_total": None,
            "core_nan_fraction_per_channel": {},
        }

    meta_path = interim_dir / f"{event_id}.json"
    cube_path = interim_dir / f"{event_id}.npy"
    fuel_path = interim_dir / f"{event_id}_fuel_potential.npy"

    meta = _load_json(meta_path)
    cube = np.load(cube_path).astype(np.float32)

    if cube.ndim != 4:
        return {
            "event_id": event_id,
            "train_eligible": False,
            "fuel_prior_valid": False,
            "rejection_reasons": [f"invalid_cube_ndim:{cube.ndim}"],
            "core_nan_fraction_total": None,
            "core_nan_fraction_per_channel": {},
        }

    timestamps = meta.get("timestamps", [])
    temporal_integrity_ok = bool(meta.get("temporal_integrity_ok", False))
    sid = str(meta.get("sid", "")).strip()
    channels = list(meta.get("channels", []))

    # ------------------------------------------------------------------
    # Metadata consistency checks
    # ------------------------------------------------------------------
    if not temporal_integrity_ok:
        eligible = False
        reasons.append("temporal_integrity_failed")

    if len(timestamps) != cube.shape[2]:
        eligible = False
        reasons.append("timestamp_count_mismatch")

    if not sid:
        eligible = False
        reasons.append("missing_sid")

    if not channels:
        eligible = False
        reasons.append("missing_channels_metadata")

    if cube.shape[-1] != len(channels):
        eligible = False
        reasons.append("channel_count_mismatch")

    try:
        idx = _channel_indices(channels, core_channels)
    except KeyError as exc:
        return {
            "event_id": event_id,
            "sid": sid,
            "storm_name": meta.get("storm_name", ""),
            "ri_label": meta.get("ri_label", None),
            "train_eligible": False,
            "fuel_prior_valid": False,
            "rejection_reasons": [str(exc)],
            "core_nan_fraction_total": None,
            "core_nan_fraction_per_channel": {},
        }

    # ------------------------------------------------------------------
    # Core-channel scientific validity checks
    #
    # These are the channels actually used by the released model.
    # For scientific strictness, every core channel must be fully finite.
    # Fraction-based NaN metrics are still computed and stored for auditability.
    # ------------------------------------------------------------------
    core_nan_fraction_per_channel: Dict[str, float] = {}
    core_arrays: List[np.ndarray] = []

    for channel_name in core_channels:
        arr = cube[..., idx[channel_name]]
        core_arrays.append(arr)

        frac = _nan_fraction(arr)
        core_nan_fraction_per_channel[channel_name] = frac

        # Strict scientific rule:
        # any non-finite value in a core input channel makes the event
        # ineligible for training.
        if not np.isfinite(arr).all():
            eligible = False
            reasons.append(f"non_finite_core_input:{channel_name}")

        # Fraction thresholds are retained as explicit audit diagnostics.
        if frac > thresholds.max_core_nan_fraction_per_channel:
            eligible = False
            reasons.append(
                f"core_nan_fraction_per_channel_exceeded:{channel_name}:{frac:.6f}"
            )

    core_stack = np.stack(core_arrays, axis=-1)
    core_nan_fraction_total = _nan_fraction(core_stack)

    # Aggregate strict finiteness check for the full released-model input tensor.
    if not np.isfinite(core_stack).all():
        eligible = False
        reasons.append("non_finite_core_inputs_total")

    if core_nan_fraction_total > thresholds.max_core_nan_fraction_total:
        eligible = False
        reasons.append(
            f"core_nan_fraction_total_exceeded:{core_nan_fraction_total:.6f}"
        )

    # ------------------------------------------------------------------
    # Physical plausibility checks on physically interpretable channels
    # ------------------------------------------------------------------
    if "sst_K" in idx:
        if not _check_finite_range(cube[..., idx["sst_K"]], thresholds.sst_range_K):
            eligible = False
            reasons.append("sst_K_out_of_physical_range")

    if "mslp_Pa" in idx:
        if not _check_finite_range(cube[..., idx["mslp_Pa"]], thresholds.msl_range_Pa):
            eligible = False
            reasons.append("mslp_Pa_out_of_physical_range")

    for wind_name in ["u10_mps", "v10_mps", "wind_mps"]:
        if wind_name in idx:
            if not _check_finite_abs_max(
                cube[..., idx[wind_name]], thresholds.wind_abs_max_mps
            ):
                eligible = False
                reasons.append(f"{wind_name}_out_of_physical_range")

    # ------------------------------------------------------------------
    # Auxiliary fuel-prior validity
    #
    # The fuel prior is tracked separately because it is auxiliary weak
    # supervision rather than physical ground truth in the released model.
    # ------------------------------------------------------------------
    fuel_prior_valid = False
    if fuel_path.exists():
        fuel = np.load(fuel_path).astype(np.float32)
        fuel_prior_valid = np.isfinite(fuel).all()

        if thresholds.require_fuel_prior_for_training and not fuel_prior_valid:
            eligible = False
            reasons.append("fuel_prior_invalid")
    else:
        if thresholds.require_fuel_prior_for_training:
            eligible = False
            reasons.append("missing_fuel_prior")

    # ------------------------------------------------------------------
    # Final audited record
    # ------------------------------------------------------------------
    return {
        "event_id": event_id,
        "sid": sid,
        "storm_name": meta.get("storm_name", ""),
        "ri_label": meta.get("ri_label", None),
        "train_eligible": bool(eligible),
        "fuel_prior_valid": bool(fuel_prior_valid),
        "rejection_reasons": reasons,
        "core_nan_fraction_total": float(core_nan_fraction_total),
        "core_nan_fraction_per_channel": core_nan_fraction_per_channel,
    }

def build_training_manifests(
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Audit all preprocessed events and write valid/rejected manifests.

    This manifest stage is the scientific gate between preprocessing and training.
    """
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    normalized_dir = Path(
        cfg_get(cfg, "paths.normalized_dir", "./data/normalized")
    ).resolve()
    valid_manifest_path = Path(
        cfg_get(cfg, "paths.valid_manifest", str(normalized_dir / "valid_events.csv"))
    ).resolve()
    rejected_manifest_path = normalized_dir / "rejected_events.csv"
    report_path = normalized_dir / "normalization_report.json"

    core_channels = _resolve_core_channels(cfg)
    thresholds = _resolve_thresholds(cfg)

    normalized_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(interim_dir.glob("era5_*.json"))
    if not json_files:
        raise RuntimeError(f"No event metadata JSON files found in {interim_dir}")

    iterable: Iterable[Path] = json_files
    if tqdm is not None:
        iterable = tqdm(json_files, desc="Audit eligibility", unit="event")

    audited_rows: List[Dict[str, Any]] = []
    for meta_path in iterable:
        event_id = _safe_event_id_from_path(meta_path)
        audited_rows.append(
            audit_event_for_training(
                event_id=event_id,
                interim_dir=interim_dir,
                core_channels=core_channels,
                thresholds=thresholds,
            )
        )

    df = pd.DataFrame(audited_rows)
    valid_df = df[df["train_eligible"] == True].copy()
    rejected_df = df[df["train_eligible"] == False].copy()

    if valid_df.empty:
        raise RuntimeError(
            "No scientifically eligible events were found. "
            "Check preprocessing outputs and normalization thresholds."
        )

    valid_cols = ["event_id", "sid", "storm_name", "ri_label"]
    valid_df[valid_cols].to_csv(valid_manifest_path, index=False)
    rejected_df.to_csv(rejected_manifest_path, index=False)

    report = {
        "n_events_total": int(len(df)),
        "n_events_valid": int(len(valid_df)),
        "n_events_rejected": int(len(rejected_df)),
        "valid_manifest": str(valid_manifest_path),
        "rejected_manifest": str(rejected_manifest_path),
        "core_channels": core_channels,
        "thresholds": {
            "sst_range_K": list(thresholds.sst_range_K),
            "msl_range_Pa": list(thresholds.msl_range_Pa),
            "wind_abs_max_mps": thresholds.wind_abs_max_mps,
            "max_core_nan_fraction_per_channel": thresholds.max_core_nan_fraction_per_channel,
            "max_core_nan_fraction_total": thresholds.max_core_nan_fraction_total,
            "require_fuel_prior_for_training": thresholds.require_fuel_prior_for_training,
        },
        "notes": (
            "Events were accepted only if the released-model core channels were "
            "physically plausible, sufficiently complete, temporally valid, and "
            "fully auditable. No NaN imputation was applied."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "Eligibility audit complete | total=%d | valid=%d | rejected=%d",
        len(df),
        len(valid_df),
        len(rejected_df),
    )

    return {
        "valid_manifest_path": valid_manifest_path,
        "rejected_manifest_path": rejected_manifest_path,
        "report_path": report_path,
        "n_valid": len(valid_df),
        "n_rejected": len(rejected_df),
        "core_channels": core_channels,
    }


def _read_train_ids_from_splits(splits_csv: Path) -> List[str]:
    """Read the train event IDs from a splits CSV."""
    df = pd.read_csv(splits_csv)
    if "event_id" not in df.columns or "split" not in df.columns:
        raise ValueError("splits_csv must contain columns: event_id, split")

    unknown_splits = sorted(set(df["split"].unique()) - VALID_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unexpected split names in splits_csv: {unknown_splits}")

    train_ids = df[df["split"] == "train"]["event_id"].astype(str).tolist()
    if not train_ids:
        raise RuntimeError(f"No train events found in splits CSV: {splits_csv}")
    return train_ids


def compute_norm_stats_from_splits(
    interim_dir: Path,
    splits_csv: Path,
    out_path: Path,
    input_names: List[str],
) -> Dict[str, Any]:
    """
    Compute train-only normalization statistics from scientifically valid events.

    This function assumes that:
    - splits_csv was created from the valid manifest only;
    - therefore all train IDs are already scientifically eligible.

    For additional safety, this function still rejects any train event that
    contains non-finite values in the released model input channels.
    """
    interim_dir = Path(interim_dir).resolve()
    splits_csv = Path(splits_csv).resolve()
    out_path = Path(out_path).resolve()

    train_ids = _read_train_ids_from_splits(splits_csv)

    c = len(input_names)
    sum_c = np.zeros(c, dtype=np.float64)
    sumsq_c = np.zeros(c, dtype=np.float64)
    count_c = np.zeros(c, dtype=np.int64)

    used_events = 0
    skipped_events = 0
    skipped_reasons: Dict[str, int] = {
        "missing_artifacts": 0,
        "missing_channels": 0,
        "non_finite_input_values": 0,
    }

    iterable: Iterable[str] = train_ids
    if tqdm is not None:
        iterable = tqdm(train_ids, desc="Normalize (train-only, strict)", unit="event")

    for event_id in iterable:
        meta_path = interim_dir / f"{event_id}.json"
        npy_path = interim_dir / f"{event_id}.npy"

        if not meta_path.exists() or not npy_path.exists():
            skipped_events += 1
            skipped_reasons["missing_artifacts"] += 1
            continue

        meta = _load_json(meta_path)
        all_channels = list(meta.get("channels", []))
        if not all_channels or not all(ch in all_channels for ch in input_names):
            skipped_events += 1
            skipped_reasons["missing_channels"] += 1
            continue

        idx = [all_channels.index(ch) for ch in input_names]
        cube = np.load(npy_path).astype(np.float64)
        x = cube[..., idx].reshape(-1, c)

        if not np.isfinite(x).all():
            skipped_events += 1
            skipped_reasons["non_finite_input_values"] += 1
            continue

        sum_c += np.sum(x, axis=0)
        sumsq_c += np.sum(x * x, axis=0)
        count_c += x.shape[0]
        used_events += 1

    if used_events == 0:
        raise RuntimeError(
            "No scientifically valid train events contributed to normalization. "
            "Check the eligibility gate and the generated splits."
        )

    denom = np.maximum(count_c, 1).astype(np.float64)
    mean = (sum_c / denom).astype(np.float32)
    var = (sumsq_c / denom) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var).astype(np.float32)

    out = {
        "channels": list(input_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count_c.tolist(),
        "notes": (
            "Train-only normalization statistics computed from scientifically valid "
            "events only. No NaN imputation or finite-value masking was applied to "
            "input channels. Any non-finite input value caused the event to be skipped."
        ),
        "debug": {
            "used_events": used_events,
            "skipped_events": skipped_events,
            "skipped_reasons": skipped_reasons,
            "splits_csv": str(splits_csv),
            "interim_dir": str(interim_dir),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Saved normalization stats to: %s", out_path)
    logger.info("Normalization channels: %s", out["channels"])
    logger.info("Used train events: %d | Skipped train events: %d", used_events, skipped_events)

    return out


def compute_norm_stats(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Entrypoint for the normalization stage.

    Workflow
    --------
    1. Audit all interim events and build valid/rejected manifests.
    2. Read storm-based train/val/test splits from the valid manifest.
    3. Compute train-only normalization statistics using only scientifically valid
       train events and only for the released model input channels.

    Notes
    -----
    This function expects that splits.csv was built from the valid manifest.
    If run.py follows the correct order, that condition is satisfied automatically.
    """
    if cfg is None:
        cfg = load_config("config.yaml")

    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    splits_csv = Path(cfg_get(cfg, "paths.splits_csv", "./data/normalized/splits.csv")).resolve()
    out_path = Path(
        cfg_get(cfg, "paths.normalization_stats", "./data/normalized/normalization_stats.json")
    ).resolve()

    build_training_manifests(cfg)

    if not splits_csv.exists():
        raise FileNotFoundError(
            f"Splits CSV not found: {splits_csv}. The normalization stage requires "
            "storm-based splits generated from the valid manifest."
        )

    input_names = _resolve_input_names(cfg)
    return compute_norm_stats_from_splits(
        interim_dir=interim_dir,
        splits_csv=splits_csv,
        out_path=out_path,
        input_names=input_names,
    )


def main() -> None:
    compute_norm_stats()


if __name__ == "__main__":
    main()