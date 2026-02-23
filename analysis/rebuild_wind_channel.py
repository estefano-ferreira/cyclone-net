from __future__ import annotations
import json
from pathlib import Path
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

INTERIM = Path("data/interim")


def main() -> None:
    json_files = sorted(INTERIM.glob("era5_*.json"))
    if not json_files:
        raise FileNotFoundError("No era5_*.json files found in data/interim/")

    iterable = json_files
    if tqdm is not None:
        iterable = tqdm(json_files, desc="Rebuild wind_mps", unit="event")

    fixed = 0
    skipped = 0

    for jp in iterable:
        eid = jp.stem
        npy_path = INTERIM / f"{eid}.npy"
        if not npy_path.exists():
            skipped += 1
            continue

        meta = json.loads(jp.read_text(encoding="utf-8"))
        chs = list(meta.get("channels", []))
        if not chs:
            skipped += 1
            continue

        needed = {"u10_mps", "v10_mps", "wind_mps"}
        if not needed.issubset(set(chs)):
            skipped += 1
            continue

        cube = np.load(npy_path).astype(np.float32)  # (H,W,T,C)
        iu = chs.index("u10_mps")
        iv = chs.index("v10_mps")
        iw = chs.index("wind_mps")

        u = cube[:, :, :, iu]
        v = cube[:, :, :, iv]
        cube[:, :, :, iw] = np.sqrt(u * u + v * v).astype(np.float32)

        np.save(npy_path, cube.astype(np.float32))
        fixed += 1

    print(f"Done. fixed={fixed} skipped={skipped}")


if __name__ == "__main__":
    main()
