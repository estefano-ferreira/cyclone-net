### 🌪️ **CycloneNet — A Forensic Engineering Framework for Atmospheric Analysis**

**CycloneNet** is an open‑source software framework designed for the **forensic audit of tropical cyclones**. It provides an automated, reproducible pipeline that ingests historical meteorological data (ERA5 reanalysis, IBTrACS) and produces geospatially localized diagnostic maps of thermodynamic conditions associated with rapid intensification (RI).

Unlike operational forecasting models, CycloneNet is built as a **high‑recall diagnostic tool** with a strong emphasis on **auditability, transparency, and reproducibility**. It is the result of applying robust software engineering principles to complex geospatial data, creating a verifiable foundation for retrospective storm analysis.

### 👨‍💻 **Developer's Vision**

> "As a **Software Engineer** with extensive experience in building mission-critical systems, I've long been driven by a practical question: Can we systematically trace the energy sources of past hurricanes with the same rigor we apply to software systems?
>
> My expertise is in architecture and code, not in atmospheric physics. This project is an application of robust software engineering principles to complex geospatial data. I built CycloneNet to create a transparent, automated pipeline that converts raw climate data into actionable forensic insights.
>
> By meticulously auditing historical storms, we build a verifiable foundation. This isn't about replacing physics-based models; it's about creating a new, complementary tool for analysis—a bridge between data engineering and atmospheric science."
> — **Estefano Senhor Ferreira**

---

## 🔍 Philosophy & Design Goals

- **Forensic Traceability**  
  Every step – from data download to final heatmap – is logged and versioned. All intermediate artifacts (cubes, metadata, grids) are stored in a structured format, enabling independent verification and replay of any analysis.

- **Reproducible Science**  
  The pipeline is entirely configuration‑driven (`config.yaml`). Splits by storm identifier (SID) prevent data leakage, and normalization statistics are computed exclusively on the training set. A complete audit trail allows exact reconstruction of any experiment.

- **High‑Sensitivity Detection**  
  The system is tuned to maximise recall (true positive rate) – a deliberate trade‑off to ensure that no potential intensification signature is missed in historical records. This safety‑first bias is documented and can be adjusted via the configuration.

- **Geospatial Attribution**  
  Using integrated gradients and soft‑argmax, the model produces continuous coordinates of the most influential region within a 40×40 km window. The resulting “target lock” can be compared directly to the storm centre or to a physically derived energy proxy.

---

## 🧱 Architecture Overview

The framework is organised into several modular stages, each with a clear responsibility:

```text
cyclone-net/
├── config.yaml                 # Single source of truth for all parameters
├── run.py                      # Pipeline orchestrator (prepare, download, preprocess, train, evaluate)
├── src/
│   ├── downloaders/            # ERA5 (monthly) and IBTrACS downloaders – original NetCDF files are never modified
│   ├── processors/              # IBTrACS parsing, RI labeling, scientific preprocessing (cube extraction)
│   ├── data/                    # PyTorch Dataset, normalisation, splits
│   ├── models/                   # CycloneNetRIOnly with spatio‑temporal attention
│   ├── training/                 # Config‑driven trainer with thresholding & calibration
│   ├── evaluation/               # MissionEvaluator, integrated gradients, final reporting
│   └── utils/                    # Configuration loader, I/O helpers, geometry, splits
└── outputs/                      # Figures, logs, evaluation results
```

Key features:

- **Immutable Raw Data** – ERA5 monthly files are downloaded once and never altered; all derived products (cubes, grids) are stored separately.
- **Storm‑Level Splits** – Data are split by SID to guarantee that no storm appears in more than one set (train/val/test).
- **Physical Unit Checks** – SST and MSLP are normalised to Kelvin and Pascal; unrealistic values cause event rejection, eliminating synthetic fallbacks.
- **Self‑Contained Metadata** – Each event’s JSON now contains the full list of timestamps and centre coordinates, enabling validation independent of the original event list.

---

## 📊 Current Status & Validation

> **Note:** The metrics below are based on **validation‑set performance** during training. Final test‑set results will be published after a complete evaluation run. All numbers are derived from a pipeline that strictly adheres to the scientific principles outlined above.

| Metric                     | Validation Value (approx.) | Interpretation                                   |
| -------------------------- | -------------------------- | ------------------------------------------------ |
| **PR‑AUC**                 | 0.53                       | Balanced precision‑recall for the minority class |
| **Recall**                 | 0.91                       | High sensitivity – most RI events are captured   |
| **Precision**              | 0.41                       | Acceptable given the recall target               |
| **Spatial error (median)** | ~18 km                     | Sub‑grid accuracy, well within ERA5 resolution   |
| **AUC**                    | 0.83                       | Good overall discriminative power                |

A detailed validation report, including per‑storm breakdown and confusion matrix, will be added to `BENCHMARK.md` after final evaluation.

---

## ⚠️ Important Distinctions

- **Diagnostic, not predictive** – The framework is validated on historical data (hindcast) and has not been tested for real‑time forecasting.
- **Engineering‑first** – The primary contribution is a robust, auditable data pipeline; the neural network is a proof‑of‑concept that demonstrates the integration path.
- **Deliberate bias** – High recall is achieved by accepting a moderate number of false positives (e.g., the Isaac case). This trade‑off is configurable and fully documented.

---

## 🚀 Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/estefano-ferreira/cyclone-net.git
   cd cyclone-net
   ```

2. **Set up environment and dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Configure access to Copernicus CDS**  
   Create a `.cdsapirc` file in your home directory with your API credentials (see [CDS documentation](https://cds.climate.copernicus.eu/api-how-to)).

4. **Run the full pipeline**

   ```bash
   # Download IBTrACS and prepare event list
   python run.py prepare

   # Download missing ERA5 monthly files
   python run.py download-era5

   # Extract scientific cubes (40×40×5×4)
   python run.py preprocess

   # Compute normalisation statistics on training split
   python run.py normalize

   # Train the model
   python run.py train --epochs 200

   # Evaluate on test set
   python run.py evaluate
   ```

Results will appear in `outputs/`.

---

## 📜 License

This project is licensed under the **Creative Commons Attribution‑NonCommercial 4.0 International (CC BY‑NC 4.0)**. Commercial use requires explicit written permission from the author.

© 2026 Estefano Senhor Ferreira
