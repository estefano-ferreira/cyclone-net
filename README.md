### 🌪️ **CycloneNet — A Forensic Engineering Framework for Atmospheric Analysis**

**CycloneNet** is an open‑source software framework designed for the **forensic audit of tropical cyclones**. It provides an automated, reproducible pipeline that ingests historical meteorological data (ERA5 reanalysis, IBTrACS) and produces geospatially localized diagnostic maps of thermodynamic conditions associated with rapid intensification (RI).

Unlike operational forecasting models, CycloneNet is built as a **high‑recall diagnostic tool** with a strong emphasis on **auditability, transparency, and reproducibility**. It is the result of applying robust software engineering principles to complex geospatial data, creating a verifiable foundation for retrospective storm analysis.

### 👨‍💻 **Developer's Vision**

> _"As a **Software Engineer** with extensive experience in building mission‑critical systems, I've long been driven by a practical question: Can we systematically trace the energy sources of past hurricanes with the same rigor we apply to software systems?_  
> _My expertise is in architecture and code, not in atmospheric physics. This project is an application of robust software engineering principles to complex geospatial data. I built CycloneNet to create a transparent, automated pipeline that converts raw climate data into actionable forensic insights._  
> _By meticulously auditing historical storms, we build a verifiable foundation. This isn't about replacing physics‑based models; it's about creating a new, complementary tool for analysis—a bridge between data engineering and atmospheric science."_  
> — **Estefano Senhor Ferreira**

---

## 🔍 Philosophy & Design Goals

- **Forensic Traceability**  
  Every step – from data download to final heatmap – is logged and versioned. All intermediate artifacts (cubes, metadata, grids) are stored in a structured format, enabling independent verification and replay of any analysis.

- **Reproducible Science**  
  The pipeline is entirely configuration‑driven (`config.yaml`). Splits by storm identifier (SID) prevent data leakage, and normalization statistics are computed exclusively on the training set. A complete audit trail allows exact reconstruction of any experiment.

- **High‑Sensitivity Detection**  
  The system is tuned to maximise recall (true positive rate) – a deliberate trade‑off to ensure that no potential intensification signature is missed in historical records. This **safety‑first bias** is documented and can be adjusted via the configuration.

- **Geospatial Attribution**  
  The model produces continuous coordinates (via soft‑argmax on a learned FuelMap) that point to the region of highest thermodynamic relevance within a 40×40 km window. The resulting “target lock” can be compared directly to the storm centre or, in future releases, to a physically derived energy proxy such as Tropical Cyclone Heat Potential (TCHP).

---

## 🧱 Architecture Overview

The framework is organised into several modular stages, each with a clear responsibility:

```text
cyclone-net/
├── config.yaml                 # Single source of truth for all parameters
├── run.py                      # Pipeline orchestrator (prepare, download, preprocess, train, evaluate)
├── src/
│   ├── downloaders/            # ERA5 (monthly) and IBTrACS downloaders – original NetCDF files are never modified
│   ├── processors/             # IBTrACS parsing, RI labeling, scientific preprocessing (cube extraction)
│   ├── data/                   # PyTorch Dataset, normalisation, splits
│   ├── models/                 # CycloneNetPhysicsGuided – 3D CNN with optional FuelMap and physics‑guided losses
│   ├── training/               # Config‑driven trainer with thresholding & calibration
│   ├── evaluation/             # Metrics, soft‑argmax localisation, and reporting
│   └── utils/                  # Configuration loader, I/O helpers, geometry, splits
└── outputs/                    # Figures, logs, evaluation results
```

**Key implementation notes:**

- **Immutable Raw Data** – ERA5 monthly files are downloaded once and never altered; all derived products (cubes, grids) are stored separately.
- **Storm‑Level Splits** – Data are split by SID to guarantee that no storm appears in more than one set (train/val/test).
- **Physical Unit Checks** – SST and MSLP are normalised to Kelvin and Pascal; unrealistic values cause event rejection, eliminating synthetic fallbacks.
- **Self‑Contained Metadata** – Each event’s JSON now contains the full list of timestamps and centre coordinates, enabling validation independent of the original event list.
- **Physics‑guided losses** – The training objective includes terms that encourage consistency between wind fields and derived vorticity/divergence, as well as alignment of the FuelMap with a simple physical prior (SST anomaly × wind speed × (1+convergence)).  
  _Note: Heat flux channels (latent, sensible, total) are computed during preprocessing but are currently **not used as model inputs**; they are retained for future integration._

---

## 📊 Final Test‑Set Performance

The model was evaluated on a held‑out test set of 2,193 samples (15% of all storms, never seen during training or validation). The threshold was selected to achieve recall ≥90% on the validation set and then applied unchanged to the test set. **No external validation data (e.g., TCHP) was used in this evaluation; therefore spatial error metrics are not reported.** Future work will integrate TCHP data to assess the model’s localisation accuracy.

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
- **Spatial validation pending** – Integration with TCHP (Tropical Cyclone Heat Potential) is planned for future releases to quantify the model’s ability to pinpoint the exact thermodynamic fuel source. Current evaluation does **not** include geographic error metrics because the necessary external data (TCHP) was not used in this benchmark.
- **Heat flux channels** – Although computed, latent and sensible heat fluxes are **not part of the model inputs** in the current version. They are stored for future enhancements.
- **Interpretability** – The model localises the energy source via **soft‑argmax on the learned FuelMap**. Gradient‑based attribution methods (e.g., integrated gradients) are implemented in `interpretability.py` but are **not yet integrated** into the evaluation pipeline; they remain experimental.

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

   # Extract scientific cubes (40×40×5×C)
   python run.py preprocess

   # Compute normalisation statistics on training split
   python run.py normalize

   # Train the model
   python run.py train

   # Evaluate on test set
   python run.py evaluate
   ```

Results (metrics, predictions, logs) will appear in `outputs/`.

---

## 📜 License

This project is licensed under the **Creative Commons Attribution‑NonCommercial 4.0 International (CC BY‑NC 4.0)**. Commercial use requires explicit written permission from the author.

© 2026 Estefano Senhor Ferreira
