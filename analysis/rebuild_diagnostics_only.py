# analysis/rebuild_diagnostics_only.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

# --- use the corrected diagnostics implementation ---
from src.physics.diagnostics import (
    estimate_dx_dy_meters,
    wind_speed,
    vorticity,
    divergence,
    grad_mslp_mag,
    sst_anomaly,
)
from src.physics.heat_flux import compute_heat_fluxes


INTERIM = Path("data/interim")

BASE = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps"]
DERIVED = [
    "wind_mps",
    "vort_1ps",
    "div_1ps",
    "grad_mslp_Pa_per_m",
    "sst_anom_K",
    "latent_heat_flux_Wpm2",
    "sensible_heat_flux_Wpm2",
    "total_heat_flux_Wpm2",
]


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main() -> None:
    json_files = sorted(INTERIM.glob("era5_*.json"))
    if not json_files:
        raise FileNotFoundError("No era5_*.json found in data/interim")

    for jp in tqdm(json_files, desc="Rebuild diagnostics", unit="event"):
        eid = jp.stem
        npy_path = INTERIM / f"{eid}.npy"
        lats_path = INTERIM / f"{eid}_lats.npy"
        lons_path = INTERIM / f"{eid}_lons.npy"

        if not npy_path.exists() or not lats_path.exists() or not lons_path.exists():
            continue

        meta = load_json(jp)
        chs = list(meta.get("channels", []))
        if not chs:
            continue

        cube = np.load(npy_path).astype(np.float32)  # (H,W,T,C)
        lats = np.load(lats_path).astype(np.float32)
        lons = np.load(lons_path).astype(np.float32)

        # Find indices for base channels (must exist)
        try:
            i_sst = chs.index("sst_K")
            i_msl = chs.index("mslp_Pa")
            i_u10 = chs.index("u10_mps")
            i_v10 = chs.index("v10_mps")
        except ValueError:
            # not a compatible cube
            continue

        sst = cube[:, :, :, i_sst]
        msl = cube[:, :, :, i_msl]
        u10 = cube[:, :, :, i_u10]
        v10 = cube[:, :, :, i_v10]

        dx, dy = estimate_dx_dy_meters(lats, lons)

        # Recompute derived per timestep
        H, W, T = sst.shape
        wind = np.empty((H, W, T), np.float32)
        vort = np.empty((H, W, T), np.float32)
        div = np.empty((H, W, T), np.float32)
        gradp = np.empty((H, W, T), np.float32)
        anom = np.empty((H, W, T), np.float32)

        lhf = np.empty((H, W, T), np.float32)
        shf = np.empty((H, W, T), np.float32)
        thf = np.empty((H, W, T), np.float32)

        for t in range(T):
            sst_t = sst[:, :, t]
            msl_t = msl[:, :, t]
            u_t = u10[:, :, t]
            v_t = v10[:, :, t]

            wind[:, :, t] = wind_speed(u_t, v_t)
            vort[:, :, t] = vorticity(u_t, v_t, dx, dy)
            div[:, :, t] = divergence(u_t, v_t, dx, dy)
            gradp[:, :, t] = grad_mslp_mag(msl_t, dx, dy)
            anom[:, :, t] = sst_anomaly(sst_t)

            flux = compute_heat_fluxes(
                sst=sst_t,
                u10=u_t,
                v10=v_t,
                msl=msl_t,
                t2m=None,
                d2m=None,
                Ce=1.2e-3,
                Ch=1.2e-3,
            )
            lhf[:, :, t] = flux["latent_heat_flux"].astype(np.float32)
            shf[:, :, t] = flux["sensible_heat_flux"].astype(np.float32)
            thf[:, :, t] = flux["total_heat_flux"].astype(np.float32)

        # Build new cube in the SAME channel order declared in meta["channels"]
        # We preserve existing channels list and overwrite the arrays by name.
        name_to_vol = {
            "sst_K": sst,
            "mslp_Pa": msl,
            "u10_mps": u10,
            "v10_mps": v10,
            "wind_mps": wind,
            "vort_1ps": vort,
            "div_1ps": div,
            "grad_mslp_Pa_per_m": gradp,
            "sst_anom_K": anom,
            "latent_heat_flux_Wpm2": lhf,
            "sensible_heat_flux_Wpm2": shf,
            "total_heat_flux_Wpm2": thf,
        }

        # overwrite cube channels that exist
        for name, vol in name_to_vol.items():
            if name in chs:
                ci = chs.index(name)
                cube[:, :, :, ci] = vol

        np.save(npy_path, cube.astype(np.float32))

        # meta consistency guard
        meta["cube_shape"] = list(cube.shape)
        save_json(jp, meta)

    print("Done. Diagnostics rebuilt without touching NetCDF.")


if __name__ == "__main__":
    main()
