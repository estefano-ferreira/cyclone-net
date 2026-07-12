# 🌪️ CycloneNet — A Forensic Engineering Framework for Atmospheric Analysis

**CycloneNet** is an open‑source software framework designed for the **forensic audit of tropical cyclones**. It provides an automated, reproducible pipeline that ingests historical meteorological data (ERA5 reanalysis, IBTrACS) and produces geospatially localized diagnostic maps of thermodynamic conditions associated with rapid intensification (RI).

Unlike operational forecasting models, CycloneNet is built as a **high‑recall diagnostic tool** with a strong emphasis on **auditability, transparency, and reproducibility**. It is the result of applying robust software engineering principles to complex geospatial data, creating a verifiable foundation for retrospective storm analysis.

---

## 👨‍💻 Developer's Vision

> _"As a **Software Engineer** with extensive experience in building mission‑critical systems, I've long been driven by a practical question: Can we systematically trace the energy sources of past hurricanes with the same rigor we apply to software systems?_  
> _My expertise is in architecture and code, not in atmospheric physics. This project is an application of robust software engineering principles to complex geospatial data. I built CycloneNet to create a transparent, automated pipeline that converts raw climate data into actionable forensic insights._  
> _By meticulously auditing historical storms, we build a verifiable foundation. This isn't about replacing physics‑based models; it's about creating a new, complementary tool for analysis—a bridge between data engineering and atmospheric science."_  
> — **Estefano Senhor Ferreira**

---

## 🔍 Philosophy & Design Goals

- **Forensic Traceability**  
  Every step – from data download to final heatmap – is logged and versioned. All intermediate artifacts (cubes, metadata, grids) are stored in a structured format, enabling independent verification and replay of any analysis.

- **Reproducible Science**  
  The pipeline is entirely configuration‑driven (`config.yaml`). Splits by storm identifier (SID) prevent data leakage, and normalization statistics are computed exclusively on the training set. A complete audit trail (Git commit hash, runtime snapshot) allows exact reconstruction of any experiment.

- **High‑Sensitivity Detection**  
  The system is tuned to maximise recall (true positive rate) – a deliberate trade‑off to ensure that no potential intensification signature is missed in historical records. This **safety‑first bias** is documented and can be adjusted via the configuration.

- **Geospatial Attribution**  
  The model produces continuous coordinates (via soft‑argmax on a learned FuelMap) that point to the region of highest thermodynamic relevance within a 40×40 grid‑point window (approx. 10°×10°). The resulting “target lock” can be compared directly to the storm centre or, in future releases, to a physically derived energy proxy such as Tropical Cyclone Heat Potential (TCHP).

---

## 🧱 Architecture Overview

The framework is organised into several modular stages, each with a clear responsibility:

```text
cyclone-net/
├── config.yaml                 # Single source of truth for all parameters
├── run.py                      # Pipeline orchestrator (prepare, download, preprocess, train, evaluate)
├── src/
│   ├── downloaders/            # ERA5 (monthly) and IBTrACS downloaders – original NetCDF files are never modified
│   │   ├── era5.py             # Downloads monthly ERA5 files from Copernicus CDS
│   │   ├── ibtracs.py          # Downloads IBTrACS best‑track CSV
│   │   └── tchp.py             # (Optional) Downloads TCHP data for validation only
│   ├── processors/              # Core scientific processing
│   │   ├── ibtracs.py          # Builds event list with RI labels (ΔV ≥30 kt/24h)
│   │   ├── preprocess_scientific.py # Extracts spatio‑temporal cubes (H,W,T,C) with diagnostic channels
│   │   ├── preprocess_tchp.py  # Adds TCHP maxima to event metadata (validation)
│   │   └── ri_labeling.py      # RI labeling logic
│   ├── data/                    # Dataset and data management
│   │   ├── dataset.py           # PhysicsDataset – loads normalized cubes, targets, and optional physics tensors
│   │   ├── normalization.py     # Train‑only mean/std computation for input channels
│   │   └── splits.py            # Storm‑level splits (by SID)
│   ├── models/                  # Neural network architectures
│   │   ├── cyclone_net_physics_guided.py # Main model (3D CNN + FuelMap head)
│   │   ├── cyclone_net_ri_only.py       # Simpler baseline (optional)
│   │   └── sta.py                # Spatio‑temporal attention module (experimental)
│   ├── physics/                  # Physics‑guided components
│   │   ├── diagnostics.py        # Finite‑difference calculations (vorticity, divergence, gradients)
│   │   ├── fuel_potential.py     # Heuristic prior map (SST anomaly × wind × (1+convergence))
│   │   ├── heat_flux.py          # Bulk aerodynamic heat fluxes (latent, sensible, total)
│   │   └── physics_guided_losses.py # KL alignment, equation consistency, TV/L1
│   ├── training/                  # Training loop and utilities
│   │   └── trainer.py            # Config‑driven trainer with threshold selection (recall‑targeted)
│   ├── evaluation/                # Evaluation and metrics
│   │   ├── evaluate.py           # Full evaluation: metrics, soft‑argmax, TCHP validation
│   │   ├── metrics.py            # ROC‑AUC, PR‑AUC (via sklearn), Brier, precision/recall
│   │   ├── spatial_metrics.py    # Advanced spatial validation (overlap, rank correlation)
│   │   └── calibration_metrics.py # Reliability, ECE, MCE
│   └── utils/                     # Helper modules
│       ├── config.py              # YAML loading, path resolution
│       ├── git.py                 # Git commit hash for provenance
│       ├── snapshot.py            # Runtime snapshot (config + versions + git)
│       ├── geometry_utils.py      # Soft‑argmax, coordinate conversions
│       └── io_utils.py            # Robust NetCDF opening (Windows‑compatible)
└── outputs/                       # All results: metrics, predictions, logs, checkpoints
    ├── results/
    ├── logs/
    └── models/checkpoints/
```

**Key implementation notes:**

- **Immutable Raw Data** – ERA5 monthly files are downloaded once and never altered; all derived products (cubes, grids) are stored separately.
- **Storm‑Level Splits** – Data are split by SID to guarantee that no storm appears in more than one set (train/val/test).
- **Unique Event Identification** – Each event now has an ID combining SID and timestamp (e.g., `AL052005_2005_08_27_0600`), eliminating any risk of collision.
- **Physical Unit Checks & NaN‑Free Guarantee** – SST and MSLP are normalised to Kelvin and Pascal; unrealistic values cause event rejection. After preprocessing, every cube is verified to contain **no NaN or Inf values** – any such event is discarded.
- **Self‑Contained Metadata** – Each event’s JSON contains the full list of timestamps, centre coordinates, channel names, and (if available) TCHP maxima – enabling validation independent of the original event list.
- **Physics‑guided losses** – Controlled entirely by the `training.physics` block in `config.yaml`; when every weight there is `0.0` the model degrades to a plain 3D‑CNN. Active terms by default:
  - **`lambda_prior_align`** – KL alignment of the learned FuelMap with a physical prior map (SST anomaly × wind speed × (1+convergence), or total heat flux when available).
  - **`lambda_forward`** – a forward physical constraint: the energy localized by the FuelMap over the prior map must predict the 24 h intensity change (`dv24`), tying "localized surface energy → intensification".
  - **`lambda_tv` / `lambda_l1`** – smoothness / sparsity regularizers keeping the FuelMap physically plausible (compact, contiguous).
  - **`lambda_consistency`** (off by default) – equation consistency between vorticity/divergence recomputed from the wind field and the stored diagnostic channels. Because both sides derive from the same input wind, this term is **near‑degenerate** and is documented as a weak representational regularizer, **not** a physical‑discovery constraint.

  _Note: Heat flux channels (latent, sensible, total) are computed during preprocessing but are currently **not used as model inputs**; they are retained for future integration._

---

## 📋 Prerequisites

- **Python 3.10 or higher** – The codebase uses modern type hints with the union syntax (`dict | None`), introduced in Python 3.10. **Versions below 3.10 are not supported.** Recommended versions: 3.12 or 3.14 (tested).
- Access to the **Copernicus Climate Data Store (CDS) API** – Required for ERA5 downloads. Create a `.cdsapirc` file in your home directory with your credentials (see [CDS documentation](https://cds.climate.copernicus.eu/api-how-to)).
- Internet connection for downloading input data (IBTrACS, ERA5, and optionally TCHP validation data).

---

## 🔐 Security / Credentials

**Credentials never live in this repository — not in `config.yaml`, not in any
tracked file.** The pipeline reads them exclusively from standard external
credential stores:

| Service | Where the credential lives | Setup |
|---|---|---|
| CDS (ERA5) | `~/.cdsapirc` | [CDS API how-to](https://cds.climate.copernicus.eu/api-how-to) |
| Copernicus Marine (TCHP/SLA) | copernicusmarine's own store or `COPERNICUSMARINE_SERVICE_USERNAME`/`_PASSWORD` | run `copernicusmarine login` once |

Defense in depth (see `security_layer/`):

1. **Redaction at the source** — run snapshots (`run_snapshot.json`) redact
   every credential-shaped value before touching disk
   (`security_layer/secret_guard.redact_secrets`, applied in
   `src/utils/snapshot.py`); the MCP `read_result` tool redacts on output too.
2. **Pre-commit hook** — contributors MUST run
   `bash security_layer/install_hook.sh` once per clone; commits containing
   credential-shaped strings or forbidden files (`config.yaml`,
   `run_snapshot.json`, `.cdsapirc`, `.netrc`, `.env`) are blocked.
3. **CI enforcement** — `.github/workflows/security.yml` runs the same
   scanner over all tracked files on every push/PR; the hook can be
   bypassed locally, CI cannot.

If a credential ever leaks: rotate it FIRST, then untrack/scrub
(`git filter-repo` — on a fresh clone, never on a dirty working tree).

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/estefano-ferreira/cyclone-net.git
cd cyclone-net
```

### 2. Set up a Python virtual environment (3.10+)

Ensure you have a compatible Python version installed:

```bash
python --version   # should show 3.10.x, 3.11.x, 3.12.x, or 3.14.x
```

If necessary, download and install the latest Python from [python.org/downloads](https://www.python.org/downloads/).

Create and activate a virtual environment:

```bash
# Create the environment (replace 'python' with the path to the correct version if needed)
python -m venv venv

# Activation:
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate
```

> **Windows tip:** If you have multiple Python versions, you can explicitly use the desired one with the `py` launcher, e.g., `py -3.12 -m venv venv` (replace `3.12` with your version).

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is present, install the core packages manually:

```bash
pip install torch xarray pandas numpy scikit-learn tqdm pyyaml netCDF4 h5netcdf scipy
```

### 4. Configure Copernicus CDS API

Create a file named `.cdsapirc` in your home directory with the following content (replace with your actual URL and key):

```
url: https://cds.climate.copernicus.eu/api
key: <your-uid>:<your-api-key>
```

### 5. Run the full pipeline

```bash
# 1. Download IBTrACS and prepare the event list (with RI labels)
python run.py prepare

# 2. Download missing ERA5 monthly files (adjust years in config.yaml if needed)
python run.py download-era5

# 3. (Optional) Download TCHP data for spatial validation
python run.py download-tchp

# 4. Extract scientific cubes (40×40×5×C) with diagnostic channels (wind speed, vorticity, divergence, etc.)
python run.py preprocess

# 5. (Optional) Add TCHP maxima to event metadata
python run.py preprocess-tchp

# 6. Compute normalization statistics on the training split (only after preprocessing)
python run.py normalize

# 7. Train the model (physics‑guided losses are enabled by default)
python run.py train

# 8. Evaluate on the test set (includes spatial error metrics if TCHP metadata is available)
python run.py evaluate

# (Optional) Generate full spatial metrics (requires TCHP maps)
python run.py evaluate --full-spatial
```

All results (metrics, predictions, logs) will be saved in `outputs/`.

---

## 🔬 New in This Release: Enhanced Preprocessing & TCHP Validation

### Physics‑guided preprocessing

The preprocessing step (`preprocess_scientific.py`) now:

- Computes **diagnostic channels**: wind speed, vorticity, divergence, MSLP gradient magnitude, and SST anomaly – using finite differences with careful handling of grid spacing.
- Calculates **heat fluxes** (latent, sensible, total) when `t2m` and `d2m` are available in the ERA5 files. These are stored in the cubes but **not used as model inputs** (they are reserved for future physics‑guided losses or validation).
- Generates a **physical prior map** (fuel potential) – by default the total heat flux (if available) or a heuristic product of SST anomaly, wind speed, and convergence. This map is saved alongside each event (`*_fuel_potential.npy`) and used to supervise the model’s FuelMap via KL divergence during training.
- **Guarantees data integrity**: Each cube is checked for NaNs/Infs; any event containing invalid values is discarded, ensuring the dataset presented to the model is clean.

### Tropical Cyclone Heat Potential (TCHP) validation

To assess the model’s ability to locate the thermodynamic energy source, we now integrate external TCHP data **for validation only**. The following steps are available:

- **Download TCHP data** (optional, set `download.tchp.enabled: true` in `config.yaml`). The downloader automatically selects the appropriate source (NOAA ERDDAP for years ≥2022, AOML FTP for 1993–2021).
- **Add TCHP maxima to metadata** with `python run.py preprocess-tchp`. For each event, the corresponding TCHP file is opened, the region around the cyclone centre is extracted, and the location of the maximum TCHP value (after smoothing) is stored in the event’s JSON metadata (`tchp_max_lat`, `tchp_max_lon`, `tchp_max_value`).
- **Spatial evaluation** during `evaluate`: if TCHP metadata exists, the script computes the great‑circle distance between the predicted FuelMap peak (via soft‑argmax) and the TCHP maximum. Advanced spatial metrics (top‑10 overlap, rank correlation) are also reported when the full TCHP map is available (enable `--full-spatial` flag).

Example test‑set output now includes:

```json
{
  "tchp_mean_dist_km": 112.4,
  "tchp_median_dist_km": 98.2,
  "tchp_std_dist_km": 67.1,
  "tchp_min_dist_km": 12.3,
  "tchp_max_dist_km": 423.5,
  "tchp_n": 1842
}
```

> **Note:** TCHP data is used **exclusively for validation** and never as a model input, preserving the scientific integrity of the experiment.

---

## 📊 Final Test‑Set Performance

> ⚠️ **Numbers below are superseded and pending regeneration.** They were produced by an
> earlier pipeline revision in which (a) the physics‑guided losses were inadvertently
> inactive and (b) the validation threshold was chosen by max‑F1. The current code makes
> the physics losses active by default (see `training.physics` in `config.yaml`) and selects
> the threshold via `precision_at_recall` honouring `training.eval_target_recall`. These
> metrics will be regenerated after re‑training with the corrected pipeline. The table is
> retained only as a historical reference point.

The model was evaluated on a held‑out test set of 2,193 samples (15% of all storms, never seen during training or validation). The threshold was selected to reach the configured target recall on the validation set and then applied unchanged to the test set. **No external validation data (e.g., TCHP) was used in this evaluation; therefore spatial error metrics are not reported.** Future work will integrate TCHP data to assess the model’s localisation accuracy.

| Metric                   | Test Value | Interpretation                                     |
| ------------------------ | ---------- | -------------------------------------------------- |
| **ROC‑AUC**              | 0.8329     | Good discriminative power.                         |
| **PR‑AUC**               | 0.3470     | Precision‑recall trade‑off for the minority class. |
| **Recall (Sensitivity)** | **0.905**  | **High sensitivity** – captures >90% of RI events. |
| **Precision**            | 0.187      | Acceptable given the recall target.                |
| **F1‑score**             | 0.310      | Harmonic mean of precision and recall.             |
| **Brier score**          | 0.074      | Well‑calibrated probabilistic outputs.             |
| **Threshold**            | 0.0666     | Operating point chosen for recall ≥90%.            |
| **Positive samples**     | 211        | RI events in the test set.                         |
| **Negative samples**     | 1982       | Non‑RI events in the test set.                     |

The high recall of **90.5%** satisfies the forensic mandate of capturing nearly all intensification events, even at the cost of a moderate number of false positives (precision 18.7%). A detailed per‑sample breakdown (including examples of false positives such as Hurricane Isaac) is available in [`BENCHMARK.md`](./BENCHMARK.md).

---

## ⚠️ Important Distinctions

- **Diagnostic, not predictive** – The framework is validated on historical data (hindcast) and has not been tested for real‑time forecasting.
- **Engineering‑first** – The primary contribution is a robust, auditable data pipeline; the neural network is a proof‑of‑concept that demonstrates the integration path.
- **Deliberate bias** – High recall is achieved by accepting a moderate number of false positives (e.g., the Isaac case). This trade‑off is configurable and fully documented.
- **Spatial validation (now implemented)** – TCHP (Tropical Cyclone Heat Potential) spatial validation is wired into the pipeline. Run `python run.py preprocess-tchp` to enrich event metadata with audited TCHP peak locations, then `python run.py evaluate --spatial` to compute the great‑circle distance between the predicted FuelMap peak and the TCHP peak, **plus a skill comparison against a naive "predict the storm centre" baseline** (the model only demonstrates spatial skill if it beats that baseline). The historical benchmark table above predates this and therefore reports no geographic error; those numbers will accompany the regenerated metrics.
- **Heat flux channels** – Although computed, latent and sensible heat fluxes are **not part of the model inputs** in the current version. They are stored for future enhancements.
- **Interpretability** – The model localises the energy source via **soft‑argmax on the learned FuelMap**. Gradient‑based attribution methods (e.g., integrated gradients) are implemented in `interpretability.py` but are **not yet integrated** into the evaluation pipeline; they remain experimental.
- **Equation consistency** – The loss that enforces vorticity/divergence derived from wind fields to match diagnostic channels is implemented, but its accuracy depends on the grid spacing (0.25°). This is documented in the code.

---

## 🐍 Python Version Note

All code is written with **type hints using the union syntax** (`X | Y`), available from Python 3.10 onward. Therefore, **Python 3.9 or older will not work**. We recommend keeping your environment up‑to‑date with Python 3.12 or 3.14, which have active support and have been thoroughly tested with this project.

---

## 📜 License

This project is licensed under the **Creative Commons Attribution‑NonCommercial 4.0 International (CC BY‑NC 4.0)**. Commercial use requires explicit written permission from the author.

© 2026 Estefano Senhor Ferreira
