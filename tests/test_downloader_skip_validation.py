"""Skip-if-exists must not reuse invalid raw files.

A hard kill mid-download can leave an empty OR truncated (>0 bytes but not a
readable NetCDF) shell on disk (the downloader's exception handler never
runs, so the partial file is not unlinked). The skip path must treat both as
missing, and a download that produces an empty file must count as a failed
attempt. No network: the CDS client is faked.
"""
import functools
import os
import tempfile
from pathlib import Path

import netCDF4

from src.downloaders.era5 import ERA5Downloader, usable_netcdf
from src.downloaders.era5_pressure import ERA5PressureDownloader


class _FakeCDSClient:
    """Records retrieve calls and writes a small non-empty file."""

    def __init__(self):
        self.calls = []

    def retrieve(self, dataset, request, target):
        self.calls.append((dataset, request, target))
        Path(target).write_bytes(b"fake-netcdf-payload")


@functools.lru_cache(maxsize=1)
def _valid_nc_bytes() -> bytes:
    """Bytes of a minimal valid NetCDF file.

    Generated once in a cwd-anchored (ASCII) temp dir: the netCDF C library
    cannot CREATE files under paths with non-ASCII components (e.g. pytest's
    tmp root under an accented user dir), while Python's own IO can write the
    bytes anywhere.
    """
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as td:
        p = Path(td) / "valid.nc"
        # cwd-relative ASCII form: the C library rejects the absolute path
        # because of the accented user-dir prefix.
        with netCDF4.Dataset(os.path.relpath(p), "w", format="NETCDF3_CLASSIC") as ds:
            ds.createDimension("x", 1)
            var = ds.createVariable("v", "f4", ("x",))
            var[:] = [1.0]
        return p.read_bytes()


def _write_valid_nc(path: Path) -> None:
    path.write_bytes(_valid_nc_bytes())


def _pl_downloader(tmp_path: Path) -> ERA5PressureDownloader:
    """Bypass __init__ (it builds a real cdsapi.Client) and set only the
    attributes _download_month uses."""
    d = object.__new__(ERA5PressureDownloader)
    d.raw_dir = tmp_path
    d.year_range = None
    d.dataset = "fake-dataset"
    d.grid = [0.25, 0.25]
    d.area = [60, -140, 0, -20]
    d.max_retries = 3
    d.retry_delay = 0
    d.wind_levels = ["850", "200"]
    d.rh_levels = ["700", "600", "500"]
    d.c = _FakeCDSClient()
    return d


def test_usable_netcdf_classification(tmp_path):
    empty = tmp_path / "empty.nc"
    empty.write_bytes(b"")
    truncated = tmp_path / "truncated.nc"
    truncated.write_bytes(b"NOT-A-NETCDF" * 100)  # >0 bytes, unopenable
    valid = tmp_path / "valid.nc"
    _write_valid_nc(valid)
    missing = tmp_path / "missing.nc"

    assert usable_netcdf(empty) is False
    assert usable_netcdf(truncated) is False
    assert usable_netcdf(valid) is True
    assert usable_netcdf(missing) is False


def test_pl_skip_reuses_only_readable_files(tmp_path):
    d = _pl_downloader(tmp_path)
    _write_valid_nc(tmp_path / "era5pl_wind_1985_06.nc")
    _write_valid_nc(tmp_path / "era5pl_rh_1985_06.nc")

    d._download_month(1985, 6, days=[6], hours=[18])

    assert d.c.calls == []  # both readable -> both skipped, no re-download


def test_pl_empty_shell_is_redownloaded(tmp_path):
    d = _pl_downloader(tmp_path)
    empty = tmp_path / "era5pl_wind_1985_06.nc"
    empty.write_bytes(b"")  # interrupted-download shell
    _write_valid_nc(tmp_path / "era5pl_rh_1985_06.nc")

    written = d._download_month(1985, 6, days=[6], hours=[18])

    assert [Path(t).name for _, _, t in d.c.calls] == ["era5pl_wind_1985_06.nc"]
    assert [p.name for p in written] == ["era5pl_wind_1985_06.nc"]
    assert empty.stat().st_size > 0


def test_pl_truncated_shell_is_redownloaded(tmp_path):
    d = _pl_downloader(tmp_path)
    truncated = tmp_path / "era5pl_wind_1985_06.nc"
    truncated.write_bytes(b"NOT-A-NETCDF" * 100)  # >0 bytes but unreadable
    _write_valid_nc(tmp_path / "era5pl_rh_1985_06.nc")

    d._download_month(1985, 6, days=[6], hours=[18])

    assert [Path(t).name for _, _, t in d.c.calls] == ["era5pl_wind_1985_06.nc"]


def test_pl_empty_file_after_download_exhausts_retries(tmp_path):
    d = _pl_downloader(tmp_path)

    class _EmptyWriter(_FakeCDSClient):
        def retrieve(self, dataset, request, target):
            self.calls.append((dataset, request, target))
            Path(target).write_bytes(b"")

    d.c = _EmptyWriter()

    written = d._download_month(1985, 6, days=[6], hours=[18])

    assert written == []
    # 2 jobs (wind + rh) x max_retries attempts each, empty result unlinked.
    assert len(d.c.calls) == 2 * d.max_retries
    assert not (tmp_path / "era5pl_wind_1985_06.nc").exists()
    assert not (tmp_path / "era5pl_rh_1985_06.nc").exists()


def test_surface_find_existing_ignores_unreadable_files(tmp_path):
    d = object.__new__(ERA5Downloader)
    d.raw_dir = tmp_path

    (tmp_path / "era5_1985_06.nc").write_bytes(b"")
    assert d.find_existing_monthly_file(1985, 6) is None

    (tmp_path / "era5_1985_06_v1.nc").write_bytes(b"NOT-A-NETCDF" * 100)
    assert d.find_existing_monthly_file(1985, 6) is None

    _write_valid_nc(tmp_path / "era5_1985_06_v2.nc")
    found = d.find_existing_monthly_file(1985, 6)
    assert found is not None and found.name == "era5_1985_06_v2.nc"
