# 🌪️ CycloneNet — A Forensic Engineering Framework for Atmospheric Analysis

**CycloneNet** is an open‑source software framework designed for the **forensic audit of tropical cyclones**. It provides an automated, reproducible pipeline that ingests historical meteorological data (ERA5 reanalysis, IBTrACS) and produces geospatially referenced **hypothesis maps** of thermodynamic conditions associated with rapid intensification (RI) — spatial energy‑source attribution from these maps was tested externally and is **not supported** (see *Spatial validation* below).

Unlike operational forecasting models, CycloneNet is built as a **high‑recall diagnostic tool** with a strong emphasis on **auditability, transparency, and reproducibility**. It is the result of applying robust software engineering principles to complex geospatial data, creating a verifiable foundation for retrospective storm analysis.

## 🌐 Live Platform — Interactive Event Explorer

**https://estefano-ferreira.github.io/cyclone-net/**

A static, client‑side explorer of **historical, observed data only**: storm tracks, intensity (wind/pressure) curves, and RI‑candidate markers from the IBTrACS best‑track record. It displays **no model predictions and makes no forecasts** — every value shown is an observation, served with SHA‑256 integrity verification against a build manifest (the browser refuses to render tampered or corrupted artifacts). Consistent with the project's validated‑negative result, the FuelMap layer is **not** part of the explorer: its slot documents that the spatial energy‑source hypothesis was tested against TCHP and is unsupported (see *Spatial validation* below and [ERRATA.md](./ERRATA.md)).

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

- **Geospatial Attribution (hypothesis — validated NEGATIVE)**  
  The model produces continuous coordinates (via soft‑argmax on a learned FuelMap) within a 40×40 grid‑point window (approx. 10°×10°). This “target lock” was tested against audited Tropical Cyclone Heat Potential (TCHP) peaks and a naive storm‑centre baseline: the FuelMap does **not** localize the energy source beyond storm position (n=226, p=0.30 vs the centre baseline). The coordinates are retained for transparency and auditability, not as a validated attribution. See [ERRATA.md](./ERRATA.md) and `docs/fuelmap_validation.md`.

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
- **Unique Event Identification** – Each event now has an ID combining timestamp and SID in the format `era5_{YYYY_MM_DD_HHMM}_{SID}` (e.g., `era5_2015_08_28_1200_2015238N10255`), eliminating any risk of collision between concurrent storms.
- **Physical Unit Checks & NaN‑Free Guarantee** – SST and MSLP are normalised to Kelvin and Pascal; unrealistic values cause event rejection. After preprocessing, every cube is verified to contain **no NaN or Inf values** – any such event is discarded.
- **Self‑Contained Metadata** – Each event’s JSON contains the full list of timestamps, centre coordinates, channel names, and (if available) TCHP maxima – enabling validation independent of the original event list.
- **Physics‑guided losses** – Controlled entirely by the `training.physics` block in `config.yaml`; when every weight there is `0.0` the model degrades to a plain 3D‑CNN. Active terms by default:
  - **`lambda_prior_align`** – KL alignment of the learned FuelMap with a physical prior map (SST anomaly × wind speed × (1+convergence), or total heat flux when available).
  - **`lambda_forward`** – a forward physical constraint: the energy the FuelMap concentrates over the (heuristic) prior map must predict the 24 h intensity change (`dv24`), tying "localized surface energy → intensification" **as training‑time supervision** — not a validated physical attribution.
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
| FTP hosts (e.g. AOML archive) | `~/.netrc` (`machine <host> login <user> password <pw>`, `chmod 600 ~/.netrc`) | optional — the AOML archive is public and defaults to anonymous login |

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

To **test the hypothesis** that the FuelMap localizes the thermodynamic energy source, we validate it against external TCHP data (**validation only** — never a model input). **The result is negative (below).** The following steps are available:

- **Download TCHP data** (optional, set `download.tchp.enabled: true` in `config.yaml`). The downloader automatically selects the appropriate source (NOAA ERDDAP for years ≥2022, AOML FTP for 1993–2021).
- **Add TCHP maxima to metadata** with `python run.py preprocess-tchp`. For each event, the corresponding TCHP file is opened, the region around the cyclone centre is extracted, and the location of the maximum TCHP value (after smoothing) is stored in the event’s JSON metadata (`tchp_max_lat`, `tchp_max_lon`, `tchp_max_value`).
- **Spatial evaluation** during `evaluate`: if TCHP metadata exists, the script computes the great‑circle distance between the predicted FuelMap peak (via soft‑argmax) and the TCHP maximum. Advanced spatial metrics (top‑10 overlap, rank correlation) are also reported when the full TCHP map is available (enable `--full-spatial` flag).

**Result of this validation (honest):** on the 226 eligible test events (TCHP publicly gridded 2022+), the FuelMap peak lies a median **539 km** from the audited TCHP peak, versus **561 km** for the naive storm‑centre baseline — closer in only 46% of events (**p = 0.30**, sign‑flip permutation). It beats a random‑point null (p = 0.0003), i.e. it tracks the storm, but shows **no localization skill beyond storm position**. A dynamic displacement test with a pure‑physics‑prior control attributes the FuelMap's behavior during RI to the enthalpy‑flux prior's arithmetic, not learned skill. Full protocol and numbers: `docs/fuelmap_validation.md`.

> **Note:** TCHP data is used **exclusively for validation** and never as a model input, preserving the scientific integrity of the experiment.

---

## 📊 Final Test‑Set Performance

The model was trained and evaluated on the full **1980–2023** North Atlantic sector archive: **16,780 valid events / 802 RI positives / 992 storms**, with hash‑deterministic storm‑level splits (adding storms never reassigns existing ones). The held‑out test split — **2,679 events / 115 RI positives / 153 storms** — was never used during development. The threshold was selected on the validation split via `precision_at_recall` and applied unchanged to the test set. These numbers are reproducible from the public repository and dataset.

| Metric                   | Test Value | Interpretation                                              |
| ------------------------ | ---------- | ----------------------------------------------------------- |
| **ROC‑AUC**              | **0.796** [95% CI 0.753–0.837] | CI entirely above chance.               |
| **PR‑AUC**               | **0.251** [95% CI 0.179–0.331] | 5.8× the 4.3% prevalence; CI above chance. |
| **Recall (Sensitivity)** | **0.852**  | High sensitivity at the forensic operating point.           |
| **Precision**            | 0.070      | The accepted cost of the recall‑first mandate.              |
| **F1‑score**             | 0.129      | Dominated by the deliberate recall bias.                    |
| **Brier score**          | 0.0372     | Calibrated probabilistic outputs (ECE 0.011).               |
| **Threshold**            | 0.0097     | Chosen on validation (142 positives) for recall ≥ 0.85.     |
| **Positive samples**     | 115        | RI events in the test set.                                  |
| **Negative samples**     | 2,564      | Non‑RI events in the test set.                              |

This is the first version of the project with **statistically demonstrable skill**: both AUC confidence intervals sit entirely above chance. Earlier releases (44 and then 96 test‑relevant positives) had CIs spanning chance — the diagnostic verdict "the bottleneck is sample size" was confirmed by intervention (17× data expansion). A per‑sample breakdown with real test‑set examples is in [`BENCHMARK.md`](./BENCHMARK.md); the correction history is in [`ERRATA.md`](./ERRATA.md).

---

## ⚠️ Important Distinctions

- **Diagnostic, not predictive** – The framework is validated on historical data (hindcast) and has not been tested for real‑time forecasting.
- **Engineering‑first** – The primary contribution is a robust, auditable data pipeline; the neural network is a proof‑of‑concept that demonstrates the integration path.
- **Deliberate bias** – High recall is achieved by accepting many false positives (precision 0.070 at 4.3% prevalence). This trade‑off is configurable and fully documented.
- **Spatial validation (executed — negative)** – TCHP spatial validation was run against the held‑out test set: the FuelMap peak does **not** beat the naive "predict the storm centre" baseline (median 539 km vs 561 km, n=226, p=0.30). The framework's validated contribution is the RI **classification** skill and the auditable pipeline; spatial energy‑source attribution remains an unsupported hypothesis. Protocol: `python run.py preprocess-tchp` then `python run.py evaluate --spatial`; full analysis in `docs/fuelmap_validation.md`.
- **Heat flux channels** – Although computed, latent and sensible heat fluxes are **not part of the model inputs** in the current version. They are stored for future enhancements.
- **Interpretability** – The model produces a spatial FuelMap via **soft‑argmax on a learned logit map**; this is retained for transparency but does **NOT** localize the energy source (see *Spatial validation* above). Gradient‑based attribution methods (e.g., integrated gradients) in `interpretability.py` remain experimental and not integrated into the evaluation pipeline.
- **Equation consistency** – The loss that enforces vorticity/divergence derived from wind fields to match diagnostic channels is implemented, but its accuracy depends on the grid spacing (0.25°). This is documented in the code.

---

## 🐍 Python Version Note

All code is written with **type hints using the union syntax** (`X | Y`), available from Python 3.10 onward. Therefore, **Python 3.9 or older will not work**. We recommend keeping your environment up‑to‑date with Python 3.12 or 3.14, which have active support and have been thoroughly tested with this project.

---

## 📜 License

This project is licensed under the **Creative Commons Attribution‑NonCommercial 4.0 International (CC BY‑NC 4.0)**. Commercial use requires explicit written permission from the author.

© 2026 Estefano Senhor Ferreira
