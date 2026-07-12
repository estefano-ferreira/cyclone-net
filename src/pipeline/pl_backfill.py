# src/pipeline/pl_backfill.py
"""
Pressure-level (PL) channel BACKFILL for events preprocessed before PL
extraction existed.

Background
----------
The 1980-2023 archive was built by ``src.pipeline.windowed``: download ERA5
surface data -> preprocess event cubes into ``data/interim/`` -> verify ->
discard raw. Events from 2020-2023 already carry 14 channels (the surface
12 plus ``shear_850_200_mps`` and ``rh_mid``) because pressure-level raw
files still existed on disk when they were preprocessed. Events from
1980-2019 have only the surface 12 channels; their surface raw is long gone,
so full reprocessing is impossible. This module appends the two missing PL
channels to those 12-channel cubes WITHOUT touching the surface channels,
using only the monthly ``era5pl_wind_YYYY_MM.nc`` / ``era5pl_rh_YYYY_MM.nc``
files (``src.downloaders.era5_pressure.ERA5PressureDownloader``).

It reuses ``src.processors.pressure_channels.extract_pressure_volume`` VERBATIM
(never reimplemented) so the backfilled channels are extracted with exactly
the same grid/level/timestep logic used for 2020-2023 events at preprocess
time -- the two archives stay co-registered.

Safety invariants (mirrors ``src.pipeline.windowed``):
  * Idempotent: an event whose ``channels`` already contains
    ``shear_850_200_mps`` is left untouched (outcome ``skipped_already_present``).
  * All-or-nothing PER EVENT: ``extract_pressure_volume`` itself is
    all-or-nothing (any missing timestep/level -> ``None``); on top of that,
    every event write is temp-file + ``os.replace`` so a mid-event exception
    NEVER leaves a partially-written ``.npy``/``.json`` on disk (outcome
    ``failed``, original files unchanged, safe to retry later).
  * Verification gate BEFORE any deletion: every event marked ``appended``
    is re-opened from disk and its channel/unit/shape/finiteness invariants
    re-checked. Any failure blocks deletion for the WHOLE window (manifest
    status ``verification_failed``) even if other events already succeeded.
  * Deletion is restricted to ``era5pl_wind_{year}_*.nc`` /
    ``era5pl_rh_{year}_*.nc`` for years INSIDE this window only -- surface
    ``era5_*.nc`` files and PL files of other years (notably 2020-2023,
    which must keep their PL raw) are never touched.
  * A window whose manifest has ``status == "completed"`` is skipped on
    resume (see ``run_all_backfill_windows``), matching
    ``windowed.MANIFEST_STATUS_COMPLETED``.

Year-scoping the downloader without mutating config.yaml on disk
-----------------------------------------------------------------
``ERA5PressureDownloader`` reads its year restriction from
``download.years`` and its required-timestamps source from
``paths.event_list``. To scope a download to exactly this window's years
(never touching config.yaml, never leaking into other windows) we
deep-copy the in-memory ``cfg``, override ``cfg_win["download"]["years"]``
and ``cfg_win["paths"]["event_list"]`` to a window-local events CSV, and
force ``required_timestamps.csv`` to regenerate scoped to this window --
exactly the pattern ``windowed.process_window`` already uses for the
surface downloader.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.processors.preprocess_scientific import close_month_cache
from src.processors.pressure_channels import (
    PL_UNITS,
    RH_CHANNEL,
    SHEAR_CHANNEL,
    extract_pressure_volume,
)
from src.pipeline.windowed import MANIFEST_STATUS_COMPLETED
from src.utils.config import cfg_get

logger = logging.getLogger(__name__)

# Distinct from windowed.MANIFEST_STATUS_FAILED: this failure mode is
# specifically the PL-append verification gate, not a download/extract gate.
MANIFEST_STATUS_VERIFICATION_FAILED = "verification_failed"


# ---------------------------------------------------------------------------
# Small helpers (deliberately duplicated from windowed.py rather than
# importing its private helpers, to keep this module decoupled from that
# module's internals -- windowed.py itself is never modified).
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
    return _provenance_dir(cfg) / f"pl_window_{y0}_{y1}.json"


def _load_manifest(cfg: Dict[str, Any], y0: int, y1: int) -> Optional[Dict[str, Any]]:
    p = manifest_path(cfg, y0, y1)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt PL backfill manifest %s -- treating window as not done.", p)
        return None


def make_windows(start_year: int, end_year: int, window_years: int = 2) -> List[Tuple[int, int]]:
    """Same 2-year-step tiling convention as ``windowed.run_all_windows``."""
    windows: List[Tuple[int, int]] = []
    y = start_year
    while y <= end_year:
        y1 = min(y + window_years - 1, end_year)
        windows.append((y, y1))
        y = y1 + 1
    return windows


# ---------------------------------------------------------------------------
# Event discovery
# ---------------------------------------------------------------------------

def _event_year(json_path: Path) -> Optional[int]:
    """Parse the event year from ``era5_YYYY_MM_DD_HHMM_SID.json``."""
    parts = json_path.stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _year_event_jsons(interim_dir: Path, year_start: int, year_end: int) -> List[Path]:
    """All era5_*.json metadata files whose event year falls in [year_start, year_end].

    Deliberately includes events that ALREADY have the PL channels -- the
    idempotent skip is recorded as an explicit per-event outcome
    (``skipped_already_present``), not a silent omission.
    """
    if not interim_dir.exists():
        return []
    out = []
    for json_path in sorted(interim_dir.glob("era5_*.json")):
        year = _event_year(json_path)
        if year is not None and year_start <= year <= year_end:
            out.append(json_path)
    return out


# ---------------------------------------------------------------------------
# PL raw-file inventory + year-scoped download (no config.yaml mutation)
# ---------------------------------------------------------------------------

def _pl_file_inventory(raw_dir: Path, year_start: int, year_end: int) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for year in range(year_start, year_end + 1):
        for pattern in (f"era5pl_wind_{year}_*.nc", f"era5pl_rh_{year}_*.nc"):
            for p in sorted(raw_dir.glob(pattern)):
                files.append({"file": p.name, "size_bytes": p.stat().st_size, "sha256": _sha256(p)})
    return files


def _window_events_from_list(event_list_csv: Path, y0: int, y1: int) -> pd.DataFrame:
    """Rows of the global event list whose t0 falls inside [y0, y1].

    Mirrors ``windowed._window_events`` (duplicated intentionally -- see
    module docstring on decoupling from windowed.py internals).
    """
    df = pd.read_csv(event_list_csv)
    if "timestamp" in df.columns:
        dt = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M")
    else:
        raise ValueError("event list must have 'timestamp' or 'datetime'")
    df = df.assign(_dt=dt)
    return df[(df["_dt"].dt.year >= y0) & (df["_dt"].dt.year <= y1)].drop(columns=["_dt"])


def ensure_pl_raw_for_window(cfg: Dict[str, Any], year_start: int, year_end: int, prov_dir: Path) -> None:
    """Download PL monthly files for THIS window's years only.

    Scopes the download in-memory: deep-copies ``cfg`` and overrides
    ``download.years`` / ``paths.event_list`` on the COPY only -- config.yaml
    on disk is never written. Regenerates ``required_timestamps.csv`` from a
    window-local events CSV so a stale cross-window cache never causes this
    window's months to be silently skipped (same trick as
    ``windowed.process_window`` uses for the surface downloader).

    Tests must monkeypatch this function to a no-op: it opens
    ``ERA5PressureDownloader``, which requires real CDS credentials and
    network access.
    """
    from src.downloaders.era5_pressure import ERA5PressureDownloader

    event_list = Path(cfg_get(cfg, "paths.event_list", "./data/event_list_augmented.csv"))
    events = _window_events_from_list(event_list, year_start, year_end)
    if events.empty:
        return

    window_events_csv = prov_dir / f"pl_window_{year_start}_{year_end}_events.csv"
    events.to_csv(window_events_csv, index=False)

    cfg_win = copy.deepcopy(cfg)
    cfg_win.setdefault("download", {})["years"] = [year_start, year_end]
    cfg_win.setdefault("paths", {})["event_list"] = str(window_events_csv)

    raw_dir = Path(cfg_get(cfg, "paths.raw_data", "./data/raw")).resolve()
    ts_csv = raw_dir / "required_timestamps.csv"
    if ts_csv.exists():
        ts_csv.unlink()

    ERA5PressureDownloader(cfg_win).download_required_batch()


# ---------------------------------------------------------------------------
# Per-event append (all-or-nothing, atomic temp+replace)
# ---------------------------------------------------------------------------

def _append_event(json_path: Path, npy_path: Path, cfg: Dict[str, Any], raw_dir: Path,
                  meta: Dict[str, Any]) -> Dict[str, Any]:
    """Append the PL volume to ONE event. Returns a per-event outcome record.

    On any exception, original ``.npy``/``.json`` are guaranteed intact: all
    computation and sanity checks happen BEFORE any write, and the write
    itself is temp-file + ``os.replace`` (atomic rename, not an in-place
    modification).
    """
    event_id = json_path.stem
    pre_sha: Optional[str] = None
    tmp_npy: Optional[Path] = None
    tmp_json: Optional[Path] = None
    try:
        pre_sha = _sha256(npy_path)
        cube = np.load(npy_path)

        # Guard against a half-migrated event (crash between the two
        # os.replace calls of a previous run: cube already extended, metadata
        # still old). The append would otherwise be caught only by the later
        # channel-count check with a confusing message.
        if cube.shape[-1] != len(meta.get("channels") or []):
            raise ValueError(
                f"cube/metadata channel mismatch before append "
                f"({cube.shape[-1]} vs {len(meta.get('channels') or [])}) — half-migrated event?")

        dt0 = datetime.strptime(str(meta["timestamp"]), "%Y-%m-%d %H:%M")
        lat0 = float(meta["center_lat"])
        lon0 = float(meta["center_lon"])
        offsets_hours = list(cfg_get(cfg, "data.offsets_hours", [0, -6, -12, -18, -24]))
        window_size_px = int(cfg_get(cfg, "data.window_size_px", 40))

        pl = extract_pressure_volume(
            raw_dir=raw_dir,
            dt0=dt0,
            offsets_hours=offsets_hours,
            lat0=lat0,
            lon0=lon0,
            window_size_px=window_size_px,
            cfg=cfg,
        )
        if pl is None:
            return {"event_id": event_id, "outcome": "pl_unavailable"}

        pl_volume, pl_names, pl_units = pl

        # Sanity checks BEFORE any write.
        expected_pl_shape = (cube.shape[0], cube.shape[1], cube.shape[2], 2)
        if tuple(pl_volume.shape) != expected_pl_shape:
            raise ValueError(f"pl_volume shape {tuple(pl_volume.shape)} != expected {expected_pl_shape}")
        if not np.all(np.isfinite(pl_volume)):
            raise ValueError("pl_volume contains non-finite values")

        old_channels = list(meta["channels"])
        new_cube = np.concatenate([cube, pl_volume], axis=-1).astype(np.float32)
        new_channels = old_channels + list(pl_names)
        if new_cube.shape[-1] != len(new_channels):
            raise ValueError(f"channel count mismatch: cube has {new_cube.shape[-1]}, metadata has {len(new_channels)}")
        if len(new_channels) != len(old_channels) + 2:
            raise ValueError(f"expected exactly two new channels, got {len(new_channels) - len(old_channels)}")

        new_units = dict(meta.get("units", {}))
        new_units.update(pl_units)

        new_meta = dict(meta)
        new_meta["channels"] = new_channels
        new_meta["units"] = new_units
        new_meta["cube_shape"] = list(new_cube.shape)

        unique = uuid.uuid4().hex
        tmp_npy = npy_path.with_name(f"{event_id}.npy.tmp-{unique}.npy")
        tmp_json = json_path.with_name(f"{event_id}.json.tmp-{unique}.json")

        np.save(tmp_npy, new_cube)
        tmp_json.write_text(json.dumps(new_meta, indent=2), encoding="utf-8")

        os.replace(tmp_npy, npy_path)
        tmp_npy = None  # already at final location; nothing left to clean up
        os.replace(tmp_json, json_path)
        tmp_json = None

        post_sha = _sha256(npy_path)
        return {
            "event_id": event_id,
            "outcome": "appended",
            "npy_sha256_before": pre_sha,
            "npy_sha256_after": post_sha,
            "n_channels_before": len(old_channels),
            "n_channels_after": len(new_channels),
        }
    except Exception as exc:
        result: Dict[str, Any] = {"event_id": event_id, "outcome": "failed", "problem": str(exc)}
        if pre_sha is not None:
            result["npy_sha256_before"] = pre_sha
        return result
    finally:
        for tmp in (tmp_npy, tmp_json):
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Verification gate (mandatory before ANY deletion)
# ---------------------------------------------------------------------------

def verify_appended_event(json_path: Path, npy_path: Path) -> Tuple[bool, Optional[str]]:
    """Re-open a freshly-appended event from disk and re-check its invariants.

    Returns (ok, problem). Never trusts the in-memory result of
    ``_append_event`` -- reads back exactly what was persisted.
    """
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"metadata JSON not parseable: {exc}"

    channels = meta.get("channels") or []
    units = meta.get("units") or {}
    if SHEAR_CHANNEL not in channels or RH_CHANNEL not in channels:
        return False, "metadata is missing one or both PL channel names"
    if SHEAR_CHANNEL not in units or RH_CHANNEL not in units:
        return False, "metadata is missing one or both PL channel units"

    try:
        cube = np.load(npy_path, mmap_mode="r")
    except Exception as exc:
        return False, f"cube not loadable (truncated?): {exc}"

    if cube.shape[-1] != len(channels):
        return False, f"cube channel count {cube.shape[-1]} != metadata channel count {len(channels)}"

    idx_shear = channels.index(SHEAR_CHANNEL)
    idx_rh = channels.index(RH_CHANNEL)
    pl_slice = np.asarray(cube[:, :, :, [idx_shear, idx_rh]], dtype=np.float32)
    if not np.all(np.isfinite(pl_slice)):
        return False, "PL channel slice contains non-finite values"

    return True, None


# ---------------------------------------------------------------------------
# Window orchestration
# ---------------------------------------------------------------------------

def backfill_window(cfg: Dict[str, Any], year_start: int, year_end: int, *,
                    dry_run: bool = False) -> Dict[str, Any]:
    """Backfill PL channels for every eligible event in [year_start, year_end].

    Steps: discover target events -> (unless dry_run) download PL raw scoped
    to this window -> append per event (all-or-nothing) -> verify every
    appended event -> (only if verification passed) delete this window's PL
    raw files. Writes ``outputs/provenance/pl_window_{y0}_{y1}.json``.

    ``dry_run=True`` only classifies events (skip vs needs-backfill) and
    returns a preview manifest; it never downloads, writes, or deletes
    anything, and does not persist a manifest file (so it can never be
    mistaken for a completed run on resume).
    """
    if year_end < year_start:
        raise ValueError(f"invalid window: [{year_start}, {year_end}]")

    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    raw_dir = Path(cfg_get(cfg, "paths.raw_data", "./data/raw")).resolve()
    prov_dir = _provenance_dir(cfg)

    manifest: Dict[str, Any] = {
        "window": [year_start, year_end],
        "created_at": _utcnow(),
        "status": "in_progress",
        "config_digest": {
            "wind_levels": cfg_get(cfg, "download.pressure_levels.wind_levels", [850, 200]),
            "rh_levels": cfg_get(cfg, "download.pressure_levels.rh_levels", [700, 600, 500]),
            "offsets_hours": cfg_get(cfg, "data.offsets_hours", None),
            "window_size_px": cfg_get(cfg, "data.window_size_px", None),
        },
    }

    year_jsons = _year_event_jsons(interim_dir, year_start, year_end)
    manifest["n_events_in_window"] = len(year_jsons)

    if not year_jsons:
        manifest.update({
            "status": MANIFEST_STATUS_COMPLETED,
            "note": "no events in this window",
            "outcome_counts": {},
            "verification": {"passed": True, "n_appended_checked": 0, "problems": []},
            "deletion": {"performed": False, "freed_bytes": 0, "deleted_files": []},
            "completed_at": _utcnow(),
        })
        manifest_path(cfg, year_start, year_end).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    # Classify without any writes: which events already have the PL channels
    # (idempotent skip) vs which still need them.
    classified: List[Tuple[Path, Optional[Dict[str, Any]]]] = []
    any_needs_backfill = False
    for jp in year_jsons:
        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            classified.append((jp, None))
            continue
        classified.append((jp, meta))
        if SHEAR_CHANNEL not in (meta.get("channels") or []):
            any_needs_backfill = True

    if dry_run:
        counts = {"already_has_pl": 0, "needs_backfill": 0, "unreadable_metadata": 0}
        for _, meta in classified:
            if meta is None:
                counts["unreadable_metadata"] += 1
            elif SHEAR_CHANNEL in (meta.get("channels") or []):
                counts["already_has_pl"] += 1
            else:
                counts["needs_backfill"] += 1
        manifest.update({"status": "dry_run", "outcome_counts": counts})
        return manifest

    if any_needs_backfill:
        ensure_pl_raw_for_window(cfg, year_start, year_end, prov_dir)

    per_event: List[Dict[str, Any]] = []
    for json_path, meta in classified:
        event_id = json_path.stem
        if meta is None:
            per_event.append({"event_id": event_id, "outcome": "failed", "problem": "metadata JSON not parseable"})
            continue
        if SHEAR_CHANNEL in (meta.get("channels") or []):
            per_event.append({"event_id": event_id, "outcome": "skipped_already_present"})
            continue
        npy_path = interim_dir / f"{event_id}.npy"
        if not npy_path.exists():
            per_event.append({"event_id": event_id, "outcome": "failed", "problem": "cube .npy missing"})
            continue
        per_event.append(_append_event(json_path, npy_path, cfg, raw_dir, meta))

    outcome_counts: Dict[str, int] = {}
    for e in per_event:
        outcome_counts[e["outcome"]] = outcome_counts.get(e["outcome"], 0) + 1
    manifest["outcome_counts"] = outcome_counts

    # NOTE: name must not collide with pl_window_{y0}_{y1}_events.csv, which
    # ensure_pl_raw_for_window writes as the downloader's input event list.
    events_csv = prov_dir / f"pl_window_{year_start}_{year_end}_outcomes.csv"
    pd.DataFrame(per_event).to_csv(events_csv, index=False)
    manifest["per_event_csv"] = str(events_csv)

    manifest["pl_downloads"] = _pl_file_inventory(raw_dir, year_start, year_end)

    # Mandatory verification gate: re-open every event marked 'appended' from
    # disk and re-check invariants. A failure here blocks deletion for the
    # WHOLE window, even if most events verify fine.
    appended_ids = [e["event_id"] for e in per_event if e["outcome"] == "appended"]
    problems: List[Dict[str, str]] = []
    for event_id in appended_ids:
        ok, problem = verify_appended_event(interim_dir / f"{event_id}.json", interim_dir / f"{event_id}.npy")
        if not ok:
            problems.append({"event_id": event_id, "problem": problem or "unknown"})
    verification = {
        "passed": len(problems) == 0,
        "n_appended_checked": len(appended_ids),
        "problems": problems,
    }
    manifest["verification"] = verification

    if not verification["passed"]:
        manifest["status"] = MANIFEST_STATUS_VERIFICATION_FAILED
        manifest["deletion"] = {"performed": False, "freed_bytes": 0, "deleted_files": []}
        manifest_path(cfg, year_start, year_end).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.error("PL backfill window %d-%d FAILED verification -- PL raw kept. Manifest: %s",
                    year_start, year_end, manifest_path(cfg, year_start, year_end))
        return manifest

    # Deletion -- only after verification passed. Release cached NetCDF
    # handles first: Windows blocks deletion of files with an open handle.
    close_month_cache()
    freed = 0
    deleted: List[str] = []
    for year in range(year_start, year_end + 1):
        # Strictly limited patterns: only this window's own PL monthlies.
        # Surface era5_{year}_*.nc files and other years' era5pl_* files
        # (e.g. the 2020-2023 archive) are never matched by these globs.
        for pattern in (f"era5pl_wind_{year}_*.nc", f"era5pl_rh_{year}_*.nc"):
            for p in sorted(raw_dir.glob(pattern)):
                freed += p.stat().st_size
                deleted.append(p.name)
                p.unlink()
    manifest["deletion"] = {"performed": True, "freed_bytes": freed, "deleted_files": deleted}
    manifest["status"] = MANIFEST_STATUS_COMPLETED
    manifest["completed_at"] = _utcnow()
    manifest_path(cfg, year_start, year_end).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("PL backfill window %d-%d: appended=%d skipped=%d unavailable=%d failed=%d, freed %.2f MB",
               year_start, year_end,
               outcome_counts.get("appended", 0), outcome_counts.get("skipped_already_present", 0),
               outcome_counts.get("pl_unavailable", 0), outcome_counts.get("failed", 0), freed / 1e6)
    return manifest


def run_all_backfill_windows(cfg: Dict[str, Any], windows: Sequence[Tuple[int, int]], *,
                             dry_run: bool = False) -> List[Dict[str, Any]]:
    """Iterate ``backfill_window`` over ``windows``, resumable via manifests.

    A window whose manifest is already ``completed`` is skipped. Stops at
    the first window that does not complete (verification failure or any
    other non-completed status) so its raw PL files stay in place for
    inspection/retry -- mirrors ``windowed.run_all_windows``.
    """
    results: List[Dict[str, Any]] = []
    for y0, y1 in windows:
        existing = _load_manifest(cfg, y0, y1)
        if existing and existing.get("status") == MANIFEST_STATUS_COMPLETED:
            logger.info("PL backfill window %d-%d already completed (manifest) -- skipping.", y0, y1)
            results.append(existing)
            continue
        manifest = backfill_window(cfg, y0, y1, dry_run=dry_run)
        results.append(manifest)
        if not dry_run and manifest.get("status") != MANIFEST_STATUS_COMPLETED:
            logger.error("PL backfill window %d-%d did not complete (status=%s) -- stopping.",
                        y0, y1, manifest.get("status"))
            break
    return results
