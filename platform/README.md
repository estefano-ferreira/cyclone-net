# CycloneNet Event Explorer V0

## What It Is

**CycloneNet Event Explorer V0** is a static forensic viewer of historical tropical-cyclone best-track observations from the International Best Track Archive for Climate Stewardship (IBTrACS). The platform displays only facts: observed location, central pressure, maximum sustained wind, and derived intensity metrics across a century of records.

**Phase 1 Status:** Historical observations only — facts. No forecasts, no operational predictions.

**Future Work (FuelMap hypothesis layer):** A future enhancement layer proposing environmental relationships between antecedent precipitation, sea-surface temperature, and cyclone intensification is planned as a research hypothesis. This hypothesis layer will be clearly marked as **NOT externally validated** and will not be deployed without explicit peer review and publication.

## Architecture

The platform is a **unidirectional data pipeline** from the scientific pipeline to a static web frontend:

```
data/event_list_augmented.csv (IBTrACS, RI labels, wind deltas)
    ↓
platform/build/build_events.py (compute dv6_kt, generate JSON) [local, auditable]
    ↓
platform/site/data/ (static JSON artifacts, versioned in repo)
    ↓
platform/site/[HTML/JS frontend] (pure client-side, no server backend)
    ↓
GitHub Pages (CI publishes platform/site only)
```

**Design principle:** The build is **local and auditable**; the CI pipeline only publishes. Data artifacts are versioned in the repository so that every published version can be reproduced from git history. This decoupling allows researchers to inspect the build process and verify data transformations before deployment.

### Structure

- **platform/build/** — Python build layer
  - `build_events.py` — Reads CSV, computes derived metrics, generates versioned JSON artifacts with SHA256 manifests
  - `README.md` — Build instructions and data conventions

- **platform/site/** — Static web content
  - `data/` — Generated static JSON (events index, GeoJSON, definitions, manifest)
  - `index.html`, `js/`, `css/`, `vendor/` — Pure client-side frontend, no server backend
  - `404.html` — GitHub Pages error handler

### Integrity & Versioning

- Every artifact in `platform/site/data/` has a SHA256 checksum in `manifest.json`.
- The browser verifies integrity on load before rendering any data (see `js/loader.js`).
- `manifest.json` includes the git commit hash and build timestamp for reproducibility.

### Key Principles

1. **Read-only pipeline output** — Build script reads (never modifies) `data/event_list_augmented.csv`. RI labels and wind deltas (dv12, dv24) are reused directly from the scientific pipeline.

2. **Lightweight JSON** — No cube data, no ERA5 downloads. Only best-track observations: location, wind, pressure, and derived deltas.

3. **Atomic data updates** — Build script atomically swaps directories (`data_old` → deleted, `data_build_tmp` → `data`). Frontend never sees inconsistent state.

4. **Versioned metadata** — `manifest.json` includes git hash and build timestamp; SHA256 checksums for all artifacts enable integrity verification.

5. **Self-documenting definitions** — `definitions.json` contains machine-readable citations (RI criterion, delta convention, IBTrACS provenance) for the frontend to display.

## How to Regenerate Data

From project root, run:

```bash
venv\Scripts\python.exe platform\build\build_events.py
```

Or with any Python environment that has `pandas` and `pyyaml`:

```bash
python platform\build\build_events.py
```

This regenerates `platform/site/data/` with a fresh manifest.json (including current git hash and timestamp). Commit the result to version it.

See `platform/build/README.md` for detailed build instructions, data conventions, and verification steps.

## How to Run Locally

### Static Server (Recommended)

From project root:

```bash
python -m http.server 8123 --directory platform\site
```

Then open **http://localhost:8123/** in your browser.

**Why not `file://`?** Browsers restrict cross-origin fetch and XMLHttpRequest from file:// URLs for security reasons. The static server bypasses this restriction so the browser can load and verify the JSON data artifacts.

### Browser Console

Once running, open the browser console (F12) to inspect:
- Manifest and SHA256 verification logs
- Any data-loading errors
- Event and geospatial queries

## Publishing to GitHub Pages

1. **Enable GitHub Pages:**
   - Repository → Settings → Pages
   - Source: **GitHub Actions**

2. **Trigger deployment:**
   - The workflow `.github/workflows/deploy-platform.yml` is triggered on:
     - Push to `main` affecting `platform/site/**`
     - Manual trigger via `workflow_dispatch`

3. **What is published:**
   - Only the contents of `platform/site/` (no build artifacts, no venv, no source data)

4. **CI philosophy:**
   - No build step in CI. Data generation happens locally and is committed to the repository, making it auditable and reproducible from git history.
   - CI only checks out and uploads the artifact.

## Data Sources & Citations

All displayed values are **historical observations**, not predictions or forecasts.

### IBTrACS
- **Dataset:** International Best Track Archive for Climate Stewardship, version 04r00
- **Citation:** Knapp, K. R., M. C. Kruk, D. P. Levinson, H. J. Diamond, and C. J. Neumann, 2010: The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying Tropical Cyclone Data. Bulletin of the American Meteorological Society, 91, 363–376. https://doi.org/10.1175/2009BAMS2755.1
- **NOAA source:** https://www.ncei.noaa.gov/products/international-best-track-archive-climate-stewardship-ibtracs

### Rapid Intensification (RI)
- **RI criterion:** A 24-hour intensity increase ≥ 30 knots (≈ 15 m/s)
- **Citation:** Kaplan, J., and M. DeMaria, 2003: Large-scale characteristics of rapidly intensifying tropical cyclones in the Atlantic and eastern Pacific basins. Weather and Forecasting, 18, 1093–1108.

### Wind Speed Derivatives
- **dv6_kt, dv12_kt, dv24_kt:** 6-, 12-, and 24-hour changes in maximum sustained wind (knots)
- **Convention:** Positive = intensification, negative = decay

## Notes for Contributors

- **Do not hand-edit** `platform/site/data/` — it is generated by `platform/build/build_events.py` and ignored in `.gitignore`
- **Commit manifest:** After running the build, commit the updated manifest and data to preserve the audit trail
- **Frontend only:** All validation, charting, and mapping happens client-side in the browser — no server API calls, no backend state
- **Future extensions:** Chart styles, hypothesis layers, and additional metrics should be added as new frontend code in `platform/site/js/` or via new optional data files in `platform/site/data/`
