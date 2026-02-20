"""I/O utilities for NetCDF (ERA5) — safe and reproducible."""

from __future__ import annotations

import logging
import os
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
from pathlib import Path
from typing import List

import xarray as xr

logger = logging.getLogger(__name__)


def get_short_path_windows(long_path: str | Path) -> str:
    """Return short path (8.3) on Windows; else original."""
    if os.name != 'nt':
        return str(long_path)
    p = Path(long_path).resolve()
    if not p.exists():
        return str(p)
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    GetShortPathNameW.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(260)
    GetShortPathNameW(str(p), buffer, 260)
    return buffer.value if buffer.value else str(p)


@contextmanager
def open_netcdf_robust(path: str | Path):
    """Open NetCDF safely, trying netcdf4 first then h5netcdf."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"NetCDF not found: {p}")

    last_err = None
    if os.name == 'nt':
        short_path = get_short_path_windows(p)
        if short_path != str(p):
            try:
                ds = xr.open_dataset(short_path, engine='netcdf4')
                yield ds
                return
            except Exception as e:
                last_err = e
                logger.debug("netcdf4 with short path failed: %s", e)

    try:
        ds = xr.open_dataset(p, engine='h5netcdf')
        yield ds
        return
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Failed to open NetCDF {p}: {last_err}") from last_err


def get_variable(ds: xr.Dataset, candidates: List[str]):
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"None of variables found: {candidates}")
