"""T5 — package the CycloneNet dataset v2 release (manifest-first).

Modes
-----
DEFAULT (dry-run): enumerate + verify everything, write a manifest preview
and verification report under outputs/results/t5_packaging/. NO staging, no
copies, nothing under data/ touched (data/ is read-only in every mode).

--build: stage the package under --dest (default: ./dist/dataset_v2/),
per-year cube shards, sidecars rewritten (fuel_potential_saved=false — the
distributed sidecar describes the distributed package; divergence recorded
in package_manifest.json), docs/licenses copied, CHECKSUMS.sha256 +
package_manifest.json written, then per-year zips + one metadata zip.

Verification is assertive: any deviation from the canonical numbers aborts.

Author decisions applied (docs/dataset_release_plan.md §8, 2026-07-16):
Zenodo / separate dataset DOI / per-year shards / rejected events as CSV
only / fuel-potential priors excluded / dataset CC BY 4.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV_KWARGS = dict(keep_default_na=False, na_values=[""])

# Canonical numbers — the package must match these EXACTLY or the build aborts.
CANONICAL = {
    "valid_events": 16_780,
    "storms": 992,
    "ri_positives": 799,
    "ri_nulls": 19,
    "ri_negatives": 15_962,
    "event_list_rows": 32_989,
    "label_diff_rows": 32_989,
    "adt_files": 761,
    "splits": {"train": 11_150, "val": 2_951, "test": 2_679},
    "provenance_manifests": {"window": 22, "pl_window": 20,
                             "basin_metadata_repair": 2,
                             "dv24_label_correction": 1},
}

PACKAGE_NAME = "cyclonenet-dataset-v2"

# Effective dataset license text for the PACKAGE (the repo's LICENSE-DATA is
# the prospective record; the packaged copy states the terms in force).
PACKAGE_LICENSE = """\
CycloneNet Dataset v2 - License

This dataset is licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0).

You are free to share and adapt the dataset for any purpose, including
commercially, provided you give appropriate credit.

Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode

MANDATORY ATTRIBUTIONS: this dataset contains and derives from third-party
data whose attribution requirements must be preserved in any redistribution.
See the NOTICE file in this package (Copernicus/ERA5, IBTrACS v04r00,
Copernicus Marine).
"""

RELEASE_DOCS = {
    "DATA_DICTIONARY.md": ROOT / "docs" / "release" / "DATA_DICTIONARY.md",
    "TECHNICAL_VALIDATION.md": ROOT / "docs" / "release" / "TECHNICAL_VALIDATION.md",
    "NOTICE": ROOT / "docs" / "release" / "NOTICE",
}

TOP_LEVEL_DATA = {
    "event_list_augmented.csv": ROOT / "data" / "event_list_augmented.csv",
    "valid_events.csv": ROOT / "data" / "normalized" / "valid_events.csv",
    "splits.csv": ROOT / "data" / "normalized" / "splits.csv",
    "frozen_splits.json": ROOT / "data" / "normalized" / "frozen_splits.json",
    "normalization_stats.json": ROOT / "data" / "normalized" / "normalization_stats.json",
    "rejected_events.csv": ROOT / "data" / "normalized" / "rejected_events.csv",
    "label_diff_v1_v2.csv": ROOT / "data" / "normalized" / "label_diff_v1_v2.csv",
}

PER_EVENT_SUFFIXES = (".npy", ".json", "_lats.npy", "_lons.npy")  # required
OPTIONAL_SUFFIXES = ("_adt.npy",)  # present for a subset
EXCLUDED_SUFFIXES = ("_fuel_potential.npy",)  # author decision: not shipped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"ABORT: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1 — enumerate + verify sources (both modes)
# ---------------------------------------------------------------------------

def enumerate_and_verify() -> dict:
    print("[1/4] Enumerating and verifying sources against canonical numbers...")

    for name, path in {**TOP_LEVEL_DATA, **RELEASE_DOCS}.items():
        if not path.exists():
            fail(f"missing required source: {name} ({path.relative_to(ROOT)})")

    ve = pd.read_csv(TOP_LEVEL_DATA["valid_events.csv"], **CSV_KWARGS)
    if len(ve) != CANONICAL["valid_events"]:
        fail(f"valid_events rows {len(ve)} != {CANONICAL['valid_events']}")
    if ve["sid"].nunique() != CANONICAL["storms"]:
        fail(f"storm count {ve['sid'].nunique()} != {CANONICAL['storms']}")

    ri = pd.to_numeric(ve["ri_label"], errors="coerce")
    counts = {"pos": int((ri == 1).sum()), "neg": int((ri == 0).sum()),
              "null": int(ri.isna().sum())}
    if counts != {"pos": CANONICAL["ri_positives"], "neg": CANONICAL["ri_negatives"],
                  "null": CANONICAL["ri_nulls"]}:
        fail(f"label counts {counts} != canonical")

    sp = pd.read_csv(TOP_LEVEL_DATA["splits.csv"], **CSV_KWARGS)
    split_counts = sp["split"].value_counts().to_dict()
    if split_counts != CANONICAL["splits"]:
        fail(f"split counts {split_counts} != {CANONICAL['splits']}")

    el = pd.read_csv(TOP_LEVEL_DATA["event_list_augmented.csv"], **CSV_KWARGS)
    if len(el) != CANONICAL["event_list_rows"]:
        fail(f"event list rows {len(el)} != {CANONICAL['event_list_rows']}")
    if "wind_kt_shift_12" in el.columns:
        fail("event list still carries wind_kt_shift_* columns (pre-v2 file?)")

    ld = pd.read_csv(TOP_LEVEL_DATA["label_diff_v1_v2.csv"], **CSV_KWARGS)
    if len(ld) != CANONICAL["label_diff_rows"]:
        fail(f"label_diff rows {len(ld)} != {CANONICAL['label_diff_rows']}")

    # Per-event artifacts: every valid event must have all required files.
    interim = ROOT / "data" / "interim"
    missing, adt_count = [], 0
    files: list[tuple[str, Path, int]] = []  # (package-relative, source, year)
    for eid in ve["event_id"]:
        year = int(eid.split("_")[1])
        for suf in PER_EVENT_SUFFIXES:
            p = interim / f"{eid}{suf}"
            if not p.exists():
                missing.append(f"{eid}{suf}")
                continue
            files.append((f"cubes/{year}/{eid}{suf}", p, year))
        for suf in OPTIONAL_SUFFIXES:
            p = interim / f"{eid}{suf}"
            if p.exists():
                adt_count += 1
                files.append((f"cubes/{year}/{eid}{suf}", p, year))
    if missing:
        fail(f"{len(missing)} missing per-event files (e.g. {missing[:5]})")
    if adt_count != CANONICAL["adt_files"]:
        fail(f"ADT extras {adt_count} != {CANONICAL['adt_files']}")

    # Provenance manifests (JSON only).
    prov_dir = ROOT / "outputs" / "provenance"
    prov_files = []
    for prefix, expected in CANONICAL["provenance_manifests"].items():
        found = sorted(prov_dir.glob(f"{prefix}_*.json"))
        # window_* glob also matches pl_window_* is impossible (prefix anchored),
        # but window_*_something.json variants are not manifests — filter to
        # exactly prefix_<digits/stamps>.json by excluding csv companions (glob
        # already .json) and pl_ collisions:
        if prefix == "window":
            found = [p for p in found if not p.name.startswith("pl_window")]
        if len(found) != expected:
            fail(f"provenance {prefix}_*.json: found {len(found)}, expected {expected}")
        prov_files.extend(found)
    for p in prov_files:
        files.append((f"provenance/{p.name}", p, -1))

    total_bytes = sum(p.stat().st_size for _, p, _ in files)
    print(f"  events={len(ve)} storms={ve['sid'].nunique()} labels={counts} "
          f"adt={adt_count} provenance={len(prov_files)}")
    print(f"  per-event+provenance files: {len(files)} "
          f"({total_bytes / 2**30:.2f} GiB)")

    return {"valid_events": ve, "files": files, "counts": counts,
            "split_counts": split_counts, "total_bytes": total_bytes,
            "n_provenance": len(prov_files)}


# ---------------------------------------------------------------------------
# Step 2 — dry-run report
# ---------------------------------------------------------------------------

def write_dryrun_report(inv: dict) -> None:
    out_dir = ROOT / "outputs" / "results" / "t5_packaging"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run",
        "canonical": CANONICAL,
        "verified": {
            "labels": inv["counts"],
            "splits": inv["split_counts"],
            "n_files": len(inv["files"]),
            "n_provenance_manifests": inv["n_provenance"],
            "total_bytes": inv["total_bytes"],
            "total_gib": round(inv["total_bytes"] / 2**30, 3),
        },
        "package_layout": {
            "shards": "cubes/<year>/ -> one zip per year + one metadata zip",
            "excluded": ["*_fuel_potential.npy (author decision)",
                          "rejected-event cubes", "ERA5 raws", "checkpoints"],
            "sidecar_rewrite": "fuel_potential_saved=false in staged copies only",
        },
    }
    path = out_dir / f"dryrun_{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[2/4] Dry-run report: {path.relative_to(ROOT)}")
    print("DRY-RUN COMPLETE. Use --build to stage and zip.")


# ---------------------------------------------------------------------------
# Step 3 — build (stage + verify staged + checksums + manifest)
# ---------------------------------------------------------------------------

def build(inv: dict, dest_root: Path) -> None:
    staging = dest_root / PACKAGE_NAME
    free = shutil.disk_usage(dest_root.anchor).free
    need = int(inv["total_bytes"] * 2.2)  # staging + zips + slack
    if free < need:
        fail(f"insufficient disk: need ~{need / 2**30:.1f} GiB, free {free / 2**30:.1f} GiB")
    if staging.exists():
        fail(f"staging dir exists, refusing to overwrite: {staging} (remove it first)")
    staging.mkdir(parents=True)
    print(f"[2/4] Staging into {staging} ...")

    checksums: dict[str, str] = {}
    rewritten = 0

    for rel, src, _year in inv["files"]:
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith(".json") and rel.startswith("cubes/"):
            # Sidecar: rewrite fuel_potential_saved (package describes package).
            meta = json.loads(src.read_text(encoding="utf-8"))
            if meta.get("fuel_potential_saved"):
                rewritten += 1
            meta["fuel_potential_saved"] = False
            dst.write_text(json.dumps(meta, allow_nan=False, indent=2),
                           encoding="utf-8")
            # Verify: staged differs from source ONLY in that field.
            back = json.loads(dst.read_text(encoding="utf-8"))
            src_meta = json.loads(src.read_text(encoding="utf-8"))
            src_meta["fuel_potential_saved"] = False
            if back != src_meta:
                fail(f"sidecar rewrite corrupted fields: {rel}")
        else:
            shutil.copyfile(src, dst)
            if sha256_file(dst) != sha256_file(src):
                fail(f"staged copy hash mismatch: {rel}")
        checksums[rel] = sha256_file(dst)

    for rel, src in TOP_LEVEL_DATA.items():
        dst = staging / rel
        shutil.copyfile(src, dst)
        if sha256_file(dst) != sha256_file(src):
            fail(f"staged copy hash mismatch: {rel}")
        checksums[rel] = sha256_file(dst)

    for rel, src in RELEASE_DOCS.items():
        dst = staging / rel
        shutil.copyfile(src, dst)
        checksums[rel] = sha256_file(dst)
    (staging / "LICENSE").write_text(PACKAGE_LICENSE, encoding="utf-8")
    checksums["LICENSE"] = sha256_file(staging / "LICENSE")

    # Re-verify labels from STAGED artifacts (the carimbo comes from the
    # package, not from the process that built it).
    ve_staged = pd.read_csv(staging / "valid_events.csv", **CSV_KWARGS)
    ri = pd.to_numeric(ve_staged["ri_label"], errors="coerce")
    staged_counts = {"pos": int((ri == 1).sum()), "neg": int((ri == 0).sum()),
                     "null": int(ri.isna().sum())}
    if staged_counts != inv["counts"]:
        fail(f"STAGED label counts {staged_counts} != verified {inv['counts']}")
    n_sidecars = sum(1 for r in checksums if r.startswith("cubes/") and r.endswith(".json"))
    if n_sidecars != CANONICAL["valid_events"]:
        fail(f"staged sidecars {n_sidecars} != {CANONICAL['valid_events']}")

    # CHECKSUMS + package manifest.
    lines = [f"{h}  {rel}" for rel, h in sorted(checksums.items())]
    (staging / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    manifest = {
        "package": PACKAGE_NAME,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "canonical": CANONICAL,
        "verified_staged": {"labels": staged_counts,
                            "files": len(checksums),
                            "sidecars": n_sidecars},
        "sidecar_divergence": {
            "field": "fuel_potential_saved",
            "value_in_package": False,
            "rewritten_from_true": rewritten,
            "reason": "fuel-potential priors are not distributed (heuristic "
                      "semantics refuted, ERRATA item 4); the distributed "
                      "sidecar describes the distributed package. Local "
                      "pipeline sidecars are unchanged.",
        },
        "excluded": ["*_fuel_potential.npy", "rejected-event cubes",
                      "ERA5 raws (re-downloadable via config-template)",
                      "model checkpoints"],
        "source_repo": "https://github.com/estefano-ferreira/cyclone-net",
        "references": {
            "technical_validation": "TECHNICAL_VALIDATION.md",
            "label_provenance": "label_diff_v1_v2.csv",
            "authoritative_assessment": "report_v5_20260716_152525 (repo)",
        },
    }
    (staging / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  staged {len(checksums)} files; sidecars rewritten "
          f"(fuel_potential_saved true->false): {rewritten}")

    # Step 4 — zips (per-year + metadata), Zenodo <=100 files.
    print("[3/4] Zipping shards...")
    zips_dir = dest_root / f"{PACKAGE_NAME}-zips"
    zips_dir.mkdir(exist_ok=True)
    zip_hashes = {}
    years = sorted({rel.split("/")[1] for rel in checksums if rel.startswith("cubes/")})
    for year in years:
        zpath = zips_dir / f"{PACKAGE_NAME}-cubes-{year}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for rel in sorted(r for r in checksums if r.startswith(f"cubes/{year}/")):
                zf.write(staging / rel, rel)
        zip_hashes[zpath.name] = sha256_file(zpath)
    meta_zip = zips_dir / f"{PACKAGE_NAME}-metadata.zip"
    with zipfile.ZipFile(meta_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in sorted(r for r in checksums if not r.startswith("cubes/")):
            zf.write(staging / rel, rel)
        zf.write(staging / "CHECKSUMS.sha256", "CHECKSUMS.sha256")
        zf.write(staging / "package_manifest.json", "package_manifest.json")
    zip_hashes[meta_zip.name] = sha256_file(meta_zip)

    (zips_dir / "ZIP_CHECKSUMS.sha256").write_text(
        "\n".join(f"{h}  {n}" for n, h in sorted(zip_hashes.items())) + "\n",
        encoding="utf-8")
    total_zip = sum((zips_dir / n).stat().st_size for n in zip_hashes)
    print(f"[4/4] {len(zip_hashes)} zips ({total_zip / 2**30:.2f} GiB) -> {zips_dir}")
    if len(zip_hashes) + 1 > 100:
        fail("Zenodo 100-file cap exceeded")
    print(f"Zenodo upload: {len(zip_hashes) + 1} files (incl. ZIP_CHECKSUMS). BUILD COMPLETE.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true",
                    help="stage + zip (default: dry-run only)")
    ap.add_argument("--dest", type=Path, default=ROOT / "dist",
                    help="destination root for staging/zips (default ./dist)")
    args = ap.parse_args()

    inv = enumerate_and_verify()
    if not args.build:
        write_dryrun_report(inv)
        return 0
    build(inv, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
