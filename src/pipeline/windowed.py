# src/pipeline/windowed.py
"""
Windowed ERA5 processing: download -> extract -> verify -> discard.

Processes the event archive in small year windows so the full 1980-present
period never requires disk space for all raw ERA5 files at once. Raw monthly
NetCDF files for a window are deleted ONLY after a mandatory verification
step confirms that every expected event in the window is accounted for and
every produced artifact is structurally valid. Each window writes a
provenance manifest (downloads with checksums, per-event outcomes,
verification results, deletion record) to ``outputs/provenance/``.

Safety invariants:
  * Raw data is never deleted unless verification passed in the same run.
  * Deletion is restricted to ``era5_{year}_*.nc`` files of the window's own
    years; nothing else in the raw directory is ever touched.
  * A window whose manifest has ``status == "completed"`` is skipped on
    resume; a failed window keeps its raw files so it can be retried.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)

# Fallback estimate when no monthly file exists on disk to measure
# (measured average over the 2020-2024 Atlantic-subset downloads).
DEFAULT_MONTH_MB = 95.0

MANIFEST_STATUS_COMPLETED = "completed"
MANIFEST_STATUS_FAILED = "failed_verification"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _provenance_dir(cfg: Dict[str, Any]) -> Path:
    out = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve().parent / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    return out


def manifest_path(cfg: Dict[str, Any], y0: int, y1: int) -> Path:
    return _provenance_dir(cfg) / f"window_{y0}_{y1}.json"


def _load_manifest(cfg: Dict[str, Any], y0: int, y1: int) -> Optional[Dict[str, Any]]:
    p = manifest_path(cfg, y0, y1)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt manifest %s — treating window as not done.", p)
        return None


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _window_events(event_list_csv: Path, y0: int, y1: int) -> pd.DataFrame:
    """Rows of the global event list whose t0 falls inside [y0, y1]."""
    # keep_default_na=False: the basin code "NA" (North Atlantic) must survive;
    # empty fields (pandas' NaN representation on write) still map to NaN.
    df = pd.read_csv(event_list_csv, keep_default_na=False, na_values=[""])
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M")
    else:
        raise ValueError("event list must have 'timestamp' or 'datetime'")
    df = df.assign(_dt=dt)
    return df[(df["_dt"].dt.year >= y0) & (df["_dt"].dt.year <= y1)].drop(columns=["_dt"])


def _event_id_for_row(row: pd.Series) -> str:
    """Reproduce preprocess_scientific's event_id (era5_YYYY_MM_DD_HHMM_SID)."""
    from src.processors.preprocess_scientific import to_event_id

    ts = pd.to_datetime(row["timestamp"] if "timestamp" in row else row["datetime"])
    return to_event_id(ts.to_pydatetime(), str(row["sid"]))


# ---------------------------------------------------------------------------
# Volume estimation (reported BEFORE downloading)
# ---------------------------------------------------------------------------

def estimate_window_download(cfg: Dict[str, Any], y0: int, y1: int) -> Dict[str, Any]:
    """Estimate download volume for a window from the months its events need.

    Uses the measured average size of monthly files already on disk when
    available, otherwise a fixed measured constant (DEFAULT_MONTH_MB).
    """
    event_list = Path(cfg_get(cfg, "paths.event_list", "./data/event_list_augmented.csv"))
    ev = _window_events(event_list, y0, y1)
    ts = pd.to_datetime(ev["timestamp"])
    months = sorted({(int(y), int(m)) for y, m in zip(ts.dt.year, ts.dt.month)})

    raw_dir = Path(cfg_get(cfg, "paths.raw_data", "./data/raw")).resolve()
    existing = list(raw_dir.glob("era5_*_*.nc"))
    avg_mb = (float(np.mean([p.stat().st_size for p in existing])) / 1e6) if existing else DEFAULT_MONTH_MB

    already = {(int(p.stem.split("_")[1]), int(p.stem.split("_")[2])) for p in existing}
    to_download = [m for m in months if m not in already]
    return {
        "window": [y0, y1],
        "n_events": int(len(ev)),
        "months_needed": [f"{y}-{m:02d}" for y, m in months],
        "months_to_download": [f"{y}-{m:02d}" for y, m in to_download],
        "avg_month_mb_measured": round(avg_mb, 1),
        "estimated_download_mb": round(avg_mb * len(to_download), 1),
    }


# ---------------------------------------------------------------------------
# Verification (mandatory gate before any deletion)
# ---------------------------------------------------------------------------

def _verify_artifact(event_id: str, interim_dir: Path, cfg: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Validate one extracted event artifact set.

    Returns (status, problem) where status is one of:
      "extracted"      — full artifact set present and structurally valid
      "not_extracted"  — no artifact set (event skipped/failed upstream)
      "invalid"        — artifacts exist but fail a structural check (BLOCKS deletion)
    """
    meta_path = interim_dir / f"{event_id}.json"
    cube_path = interim_dir / f"{event_id}.npy"
    lats_path = interim_dir / f"{event_id}_lats.npy"
    lons_path = interim_dir / f"{event_id}_lons.npy"

    present = [p for p in (meta_path, cube_path, lats_path, lons_path) if p.exists()]
    if not present:
        return "not_extracted", None
    if len(present) < 4:
        missing = [p.name for p in (meta_path, cube_path, lats_path, lons_path) if not p.exists()]
        return "invalid", f"partial artifact set, missing: {missing}"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return "invalid", f"metadata JSON not parseable: {exc}"

    channels = meta.get("channels")
    if not channels:
        return "invalid", "metadata has no 'channels' list"

    t_expected = len(cfg_get(cfg, "data.offsets_hours", [0, -6, -12, -18, -24]))
    px = int(cfg_get(cfg, "data.window_size_px", 40))
    max_nan = float(cfg_get(cfg, "data.qc.max_nan_fraction_per_channel", 0.5))

    try:
        cube = np.load(cube_path, mmap_mode="r")
    except Exception as exc:
        return "invalid", f"cube not loadable (truncated?): {exc}"

    expected_shape = (px, px, t_expected, len(channels))
    if tuple(cube.shape) != expected_shape:
        return "invalid", f"cube shape {tuple(cube.shape)} != expected {expected_shape}"

    # NaN tolerance per channel (same threshold the QC gate uses).
    for ci, ch in enumerate(channels):
        frac = float(np.isnan(np.asarray(cube[:, :, :, ci], dtype=np.float32)).mean())
        if frac > max_nan:
            return "invalid", f"channel '{ch}' NaN fraction {frac:.2f} > {max_nan}"

    for grid_path in (lats_path, lons_path):
        try:
            grid = np.load(grid_path)
        except Exception as exc:
            return "invalid", f"{grid_path.name} not loadable: {exc}"
        if px not in grid.shape:
            return "invalid", f"{grid_path.name} shape {grid.shape} inconsistent with window {px}px"

    return "extracted", None


def verify_window(cfg: Dict[str, Any], events: pd.DataFrame, downloaded_months: List[Tuple[int, int]],
                  raw_dir: Path) -> Dict[str, Any]:
    """Mandatory verification gate for one window.

    Passes only if:
      * every month required by the window's events has its monthly file on disk;
      * zero artifacts are structurally invalid (truncated cube, bad shape,
        unparseable JSON, NaN beyond tolerance).
    Events that produced no artifacts are recorded (the preprocess QC
    legitimately rejects some events) but do not block deletion by
    themselves — invalid artifacts and missing raw months do.
    """
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()

    missing_months = [f"{y}-{m:02d}" for y, m in downloaded_months
                      if not (raw_dir / f"era5_{y}_{m:02d}.nc").exists()]

    per_event: List[Dict[str, Any]] = []
    counts = {"extracted": 0, "not_extracted": 0, "invalid": 0}
    for _, row in events.iterrows():
        eid = _event_id_for_row(row)
        status, problem = _verify_artifact(eid, interim_dir, cfg)
        counts[status] += 1
        entry: Dict[str, Any] = {"event_id": eid, "status": status}
        if problem:
            entry["problem"] = problem
        per_event.append(entry)

    passed = (len(missing_months) == 0) and (counts["invalid"] == 0)
    return {
        "passed": bool(passed),
        "n_expected_events": int(len(events)),
        "counts": counts,
        "missing_raw_months": missing_months,
        "invalid_events": [e for e in per_event if e["status"] == "invalid"],
        "per_event": per_event,
    }


# ---------------------------------------------------------------------------
# Window orchestration
# ---------------------------------------------------------------------------

def process_window(cfg: Dict[str, Any], y0: int, y1: int, delete_raw: bool = True) -> Dict[str, Any]:
    """Run download -> extract -> verify -> (conditional) discard for one window.

    Raises RuntimeError if verification fails; raw files are kept in that case.
    """
    from src.downloaders.era5 import ERA5Downloader, generate_required_timestamps
    from src.processors.preprocess_scientific import run_preprocess

    if y1 < y0:
        raise ValueError(f"invalid window: [{y0}, {y1}]")

    raw_dir = Path(cfg_get(cfg, "paths.raw_data", "./data/raw")).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    prov_dir = _provenance_dir(cfg)
    event_list = Path(cfg_get(cfg, "paths.event_list", "./data/event_list_augmented.csv"))

    events = _window_events(event_list, y0, y1)
    logger.info("Window %d-%d: %d expected events", y0, y1, len(events))

    estimate = estimate_window_download(cfg, y0, y1)
    logger.info("Window %d-%d volume estimate: %s MB across %d monthly files",
                y0, y1, estimate["estimated_download_mb"], len(estimate["months_to_download"]))

    manifest: Dict[str, Any] = {
        "window": [y0, y1],
        "created_at": _utcnow(),
        "status": "in_progress",
        "estimate": estimate,
        "config_digest": {
            "variables": cfg_get(cfg, "download.variables", []),
            "grid": cfg_get(cfg, "download.grid", None),
            "spatial_subset": cfg_get(cfg, "download.spatial_subset", None),
            "offsets_hours": cfg_get(cfg, "data.offsets_hours", None),
            "window_size_px": cfg_get(cfg, "data.window_size_px", None),
        },
    }

    if len(events) == 0:
        # Nothing to do for this window (no recorded events, e.g. quiet years).
        manifest.update({"status": MANIFEST_STATUS_COMPLETED, "note": "no events in window",
                         "verification": {"passed": True, "n_expected_events": 0}})
        manifest_path(cfg, y0, y1).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    # Per-window event list feeds the standard preprocess unchanged.
    window_events_csv = prov_dir / f"window_{y0}_{y1}_events.csv"
    events.to_csv(window_events_csv, index=False)

    # 1. Download (window-scoped): regenerate required_timestamps for THIS
    # window only — it is a cross-window cache otherwise.
    cfg_win = copy.deepcopy(cfg)
    cfg_win["download"]["years"] = [y0, y1]
    cfg_win["paths"]["event_list"] = str(window_events_csv)
    ts_csv = raw_dir / "required_timestamps.csv"
    if ts_csv.exists():
        ts_csv.unlink()
    generate_required_timestamps(window_events_csv, ts_csv, year_range=(y0, y1))

    disk_before_download = _dir_size_bytes(raw_dir)
    ERA5Downloader(cfg_win).download_required_batch()

    ts_df = pd.read_csv(ts_csv)
    window_months = sorted({(int(y), int(m)) for y, m in
                            ts_df[["year", "month"]].drop_duplicates().itertuples(index=False)})

    downloads = []
    for y, m in window_months:
        p = raw_dir / f"era5_{y}_{m:02d}.nc"
        if p.exists():
            downloads.append({"file": p.name, "size_bytes": p.stat().st_size, "sha256": _sha256(p)})
    manifest["downloads"] = downloads

    # 2. Extract this window's events through the standard preprocess.
    run_preprocess(cfg_win)

    # 3. Mandatory verification gate.
    verification = verify_window(cfg, events, window_months, raw_dir)
    # The full per-event list can be large; keep it in a sidecar CSV.
    per_event = verification.pop("per_event")
    pd.DataFrame(per_event).to_csv(prov_dir / f"window_{y0}_{y1}_verification.csv", index=False)
    manifest["verification"] = verification
    manifest["disk"] = {
        "raw_dir_bytes_before_download": disk_before_download,
        "raw_dir_bytes_after_extract": _dir_size_bytes(raw_dir),
    }

    if not verification["passed"]:
        manifest["status"] = MANIFEST_STATUS_FAILED
        manifest_path(cfg, y0, y1).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.error("Window %d-%d FAILED verification — raw files kept. Manifest: %s",
                     y0, y1, manifest_path(cfg, y0, y1))
        raise RuntimeError(f"window {y0}-{y1} failed verification; raw data NOT deleted")

    # 4. Discard raw ERA5 for this window — only after verification passed.
    # Release cached NetCDF handles first: Windows cannot delete open files.
    from src.processors.preprocess_scientific import close_month_cache
    close_month_cache()

    deletion: Dict[str, Any] = {"performed": False, "freed_bytes": 0, "deleted_files": []}
    if delete_raw:
        freed = 0
        deleted = []
        for year in range(y0, y1 + 1):
            # Strictly limited pattern: only this window's ERA5 monthlies.
            for p in sorted(raw_dir.glob(f"era5_{year}_*.nc")):
                freed += p.stat().st_size
                deleted.append(p.name)
                p.unlink()
        deletion = {"performed": True, "freed_bytes": freed, "deleted_files": deleted}
        logger.info("Window %d-%d: deleted %d raw files, freed %.2f GB",
                    y0, y1, len(deleted), freed / 1e9)

    manifest["deletion"] = deletion
    manifest["disk"]["raw_dir_bytes_after_discard"] = _dir_size_bytes(raw_dir)
    manifest["status"] = MANIFEST_STATUS_COMPLETED
    manifest["completed_at"] = _utcnow()
    manifest_path(cfg, y0, y1).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_all_windows(cfg: Dict[str, Any], start_year: int, end_year: int,
                    window_years: int = 2, delete_raw: bool = True) -> List[Dict[str, Any]]:
    """Iterate windows of ``window_years`` from start to end (inclusive).

    Resumable: windows whose manifest is already ``completed`` are skipped.
    Stops at the first window that fails verification (its raw data is kept).
    """
    results = []
    y = start_year
    while y <= end_year:
        y0, y1 = y, min(y + window_years - 1, end_year)
        existing = _load_manifest(cfg, y0, y1)
        if existing and existing.get("status") == MANIFEST_STATUS_COMPLETED:
            logger.info("Window %d-%d already completed (manifest) — skipping.", y0, y1)
            results.append(existing)
        else:
            results.append(process_window(cfg, y0, y1, delete_raw=delete_raw))
        y = y1 + 1
    return results
