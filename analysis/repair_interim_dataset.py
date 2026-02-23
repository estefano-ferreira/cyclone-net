from __future__ import annotations
import json
from pathlib import Path
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

INTERIM = Path("data/interim")
NORMALIZED = Path("data/normalized")

# --- finite differences on (H,W) grids ---


def _finite_diff_x(a: np.ndarray, dx: float) -> np.ndarray:
    out = np.empty_like(a, dtype=np.float32)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dx)
    out[:, 0] = (a[:, 1] - a[:, 0]) / dx
    out[:, -1] = (a[:, -1] - a[:, -2]) / dx
    return out


def _finite_diff_y(a: np.ndarray, dy: float) -> np.ndarray:
    out = np.empty_like(a, dtype=np.float32)
    out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dy)
    out[0, :] = (a[1, :] - a[0, :]) / dy
    out[-1, :] = (a[-1, :] - a[-2, :]) / dy
    return out


def estimate_dx_dy_meters(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    lat0 = float(np.nanmean(lats))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = float(np.nanmedian(np.abs(lats[1:, :] - lats[:-1, :])))
    dlon = float(np.nanmedian(np.abs(lons[:, 1:] - lons[:, :-1])))
    dy = max(1e-6, dlat * m_per_deg_lat)
    dx = max(1e-6, dlon * m_per_deg_lon)
    return dx, dy


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u*u + v*v).astype(np.float32)


# --- Heat flux proxy (bulk) ---
# If you later add real t2m/d2m, you can update this function accordingly.
Lv = 2.5e6
Cp = 1005.0
rho_air = 1.2
epsilon = 0.622


def saturation_vapor_pressure(temp_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    return 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def specific_humidity_from_dewpoint(dewpoint_k: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    e = saturation_vapor_pressure(dewpoint_k)
    return epsilon * e / (pressure_pa - (1.0 - epsilon) * e)


def compute_heat_flux_proxy(
    sst_k: np.ndarray, u10: np.ndarray, v10: np.ndarray, msl_pa: np.ndarray,
    Ce: float = 1.2e-3, Ch: float = 1.2e-3
) -> dict:
    """
    Deterministic bulk-flux proxy.
    NOTE: This is a proxy unless t2m/d2m are provided from ERA5.
    """
    U = wind_speed(u10, v10)
    # Simple proxy assumptions:
    t2m = sst_k - 1.0
    d2m = t2m - 2.0

    q_air = specific_humidity_from_dewpoint(d2m, msl_pa)
    q_sat = specific_humidity_from_dewpoint(sst_k, msl_pa)

    lhf = rho_air * Lv * Ce * U * (q_sat - q_air)
    shf = rho_air * Cp * Ch * U * (sst_k - t2m)
    thf = lhf + shf
    return {
        "latent_heat_flux_Wpm2": lhf.astype(np.float32),
        "sensible_heat_flux_Wpm2": shf.astype(np.float32),
        "total_heat_flux_Wpm2": thf.astype(np.float32),
    }


def main() -> None:
    json_files = sorted(INTERIM.glob("era5_*.json"))
    if not json_files:
        raise FileNotFoundError("No era5_*.json found in data/interim")

    iterable = json_files if tqdm is None else tqdm(
        json_files, desc="Repair interim dataset", unit="event")

    fixed = 0
    skipped = 0

    for jp in iterable:
        eid = jp.stem
        npy_path = INTERIM / f"{eid}.npy"
        lats_path = INTERIM / f"{eid}_lats.npy"
        lons_path = INTERIM / f"{eid}_lons.npy"

        if not (npy_path.exists() and lats_path.exists() and lons_path.exists()):
            skipped += 1
            continue

        meta = json.loads(jp.read_text(encoding="utf-8"))
        chs = list(meta.get("channels", []))
        if not chs:
            skipped += 1
            continue

        required = {"sst_K", "mslp_Pa", "u10_mps", "v10_mps"}
        if not required.issubset(set(chs)):
            skipped += 1
            continue

        cube = np.load(npy_path).astype(np.float32)  # (H,W,T,C)
        lats = np.load(lats_path).astype(np.float32)
        lons = np.load(lons_path).astype(np.float32)

        dx, dy = estimate_dx_dy_meters(lats, lons)

        isst = chs.index("sst_K")
        imsl = chs.index("mslp_Pa")
        iu10 = chs.index("u10_mps")
        iv10 = chs.index("v10_mps")

        sst = cube[:, :, :, isst]
        msl = cube[:, :, :, imsl]
        u10 = cube[:, :, :, iu10]
        v10 = cube[:, :, :, iv10]

        H, W, T = sst.shape

        # Rebuild per timestep for channels that exist
        for t in range(T):
            sst_t = sst[:, :, t]
            msl_t = msl[:, :, t]
            u_t = u10[:, :, t]
            v_t = v10[:, :, t]

            if "wind_mps" in chs:
                cube[:, :, t, chs.index("wind_mps")] = wind_speed(u_t, v_t)

            if "vort_1ps" in chs:
                dv_dx = _finite_diff_x(v_t, dx)
                du_dy = _finite_diff_y(u_t, dy)
                cube[:, :, t, chs.index("vort_1ps")] = (
                    dv_dx - du_dy).astype(np.float32)

            if "div_1ps" in chs:
                du_dx = _finite_diff_x(u_t, dx)
                dv_dy = _finite_diff_y(v_t, dy)
                cube[:, :, t, chs.index("div_1ps")] = (
                    du_dx + dv_dy).astype(np.float32)

            if "grad_mslp_Pa_per_m" in chs:
                dp_dx = _finite_diff_x(msl_t, dx)
                dp_dy = _finite_diff_y(msl_t, dy)
                cube[:, :, t, chs.index("grad_mslp_Pa_per_m")] = np.sqrt(
                    dp_dx*dp_dx + dp_dy*dp_dy).astype(np.float32)

            if "sst_anom_K" in chs:
                mu = float(np.nanmean(sst_t))
                cube[:, :, t, chs.index("sst_anom_K")] = (
                    sst_t - mu).astype(np.float32)

            # Heat flux proxy channels (only if present)
            if any(name in chs for name in ["latent_heat_flux_Wpm2", "sensible_heat_flux_Wpm2", "total_heat_flux_Wpm2"]):
                flux = compute_heat_flux_proxy(sst_t, u_t, v_t, msl_t)
                for k, arr in flux.items():
                    if k in chs:
                        cube[:, :, t, chs.index(k)] = arr

        np.save(npy_path, cube.astype(np.float32))
        fixed += 1

    print(f"Done. fixed={fixed} skipped={skipped}")

    # Remove old normalization stats (they were computed on corrupted channels)
    stats = NORMALIZED / "normalization_stats.json"
    if stats.exists():
        stats.unlink()
        print(f"Removed old stats: {stats}")

    print("Now run: python run.py normalize")


if __name__ == "__main__":
    main()
