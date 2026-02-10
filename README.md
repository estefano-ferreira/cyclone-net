### 🌪️ **CycloneNet — A Forensic Engineering Framework for Atmospheric Analysis**

**CycloneNet** is a specialized software framework designed for the **forensic audit of tropical cyclones**. Its core purpose is to automate the retrospective analysis of historical storms, pinpointing oceanic and atmospheric conditions associated with periods of rapid intensification (RI).

Unlike predictive forecasting models, CycloneNet is engineered as a **high-sensitivity diagnostic tool**, prioritizing a complete audit trail and reproducible analysis to establish a reliable foundation for research and operational post-analysis.

### 👨‍💻 **Developer's Vision**

> "As a **Software Engineer** with extensive experience in building mission-critical systems, I've long been driven by a practical question: Can we systematically trace the energy sources of past hurricanes with the same rigor we apply to software systems?
>
> My expertise is in architecture and code, not in atmospheric physics. This project is an application of robust software engineering principles to complex geospatial data. I built CycloneNet to create a transparent, automated pipeline that converts raw climate data into actionable forensic insights.
>
> By meticulously auditing historical storms, we build a verifiable foundation. This isn't about replacing physics-based models; it's about creating a new, complementary tool for analysis—a bridge between data engineering and atmospheric science."
> — **Estefano Senhor Ferreira**

---

### 🛡️ **The Forensic Engineering Approach**

CycloneNet approaches cyclone analysis from a **software forensics perspective**, establishing key principles:

- **Auditability & Reproducibility:** Every analysis generates an immutable log, ensuring full traceability from raw data to final conclusions.
- **High-Sensitivity Design:** The system is tuned to maximize detection recall, operating on the principle that in forensic review, missing a potential event is a greater failure than a false alert.
- **Automated Data Pipeline:** Provides a robust, integrated workflow from data acquisition to report generation, minimizing manual effort and error.
- **Geospatial Precision:** Delivers specific geographic coordinates ("Target Locks") for analysis, with accuracy intrinsically linked to the resolution of the source data.

---

### ⚙️ **Core Architecture & Transparency**

CycloneNet's strength lies in its integrated architecture. We believe in full transparency regarding its current implementation:

**🟢 Data Engineering Pipeline (Core Innovation)**
This is the mature and robust foundation of the project:

- **Automated Data Ingestion:** Seamless integration with the **Copernicus Climate Data Store (CDS) API** and NOAA HURDAT2 for acquiring authoritative input data.
- **Data Processing & Healing:** Handles raw NetCDF/GRIB data, performing spatial extraction, normalization, and gap-filling to create consistent analysis-ready tensors.
- **Forensic Audit Trail:** Automatically logs all steps, outputs geospatial visualizations, and generates the primary `cyclonenet_scientific.csv` dataset for independent verification.

**🟡 Hybrid Analysis System (Operational Prototype)**
The analytical engine is a purpose-built hybrid system:

- **A Hybrid Model:** Combines initial pattern processing with a transparent, **rules-based expert system** (see `metrics_handler.py`). This "safety-gating" layer applies fundamental physical thresholds (e.g., pressure, wind) to validate outputs, which is the primary driver of the high recall rate.
- **Current State of the ML Component:** The included neural network model (`physics_model.py`) serves as a **functional prototype and architectural placeholder**. It demonstrates the integration pathway and input/output specifications. We openly acknowledge that this component represents a significant **opportunity for future development** to increase analytical autonomy and complexity.

**Key Technical Output: The "Target Lock"**
The framework's geographic precision is a direct and transparent product of the input data's resolution (ERA5 at 0.25°) and deterministic post-processing (`core.py`). The reported ~27.8 km mean error provides a realistic and verifiable performance baseline.

---

## 📊 Scientific Benchmark (Updated 2026-02-09)

Validated against 18 destructive Atlantic hurricanes (1989–2024). The system is calibrated to prioritize safety and detection depth.

| Metric                   | Result       | The "CycloneNet" Edge                                    |
| ------------------------ | ------------ | -------------------------------------------------------- |
| **ROC-AUC**              | **0.9736**   | **Exceptional Discriminative Power.**                    |
| **Recall (Sensitivity)** | **0.9231**   | **Safety-First Bias:** High capture rate of RI triggers. |
| **Brier Score**          | **0.1169**   | **Superior Calibration:** Reliable confidence levels.    |
| **Mean Tracking Error**  | **25.79 km** | **Sub-Pixel Accuracy:** Pinpointing fuel sources.        |

> [!IMPORTANT] > **[View the Full Scientific Validation Report, Confusion Matrix & Per-Storm Analysis](./BENCHMARK.md)**

---

## ⚠️ Disclaimer & Technical Limitations

As an engineering-first project, it is crucial to maintain transparency regarding the current scientific boundaries of this framework. **CycloneNet 1.0** should be evaluated under the following constraints:

- **Diagnostic vs. Predictive Scope:** The current results are based on **hindcast validation**. The model functions as a **high-fidelity diagnostic tool**, mapping signatures in historical data. It has not yet been benchmarked for prospective real-time forecasting.
- **Temporal Coupling (Data Leakage):** Current evaluation metrics focus on **spatial association** (Target Lock) rather than predictive lead-time. The goal is to verify if the model can "see" the energy source, not predict it days in advance.
- **Engineering Focus:** The core innovation lies in the **Data Engineering pipeline** (ERA5 autonomous ingestion and telemetry healing) rather than a meteorological breakthrough.
- **Conservative Diagnostic Bias:** The high sensitivity is a result of a deliberate engineering gate (0.6 threshold). This may lead to false positives (as seen in the _Hurricane Isaac_ case), which are intended to prioritize safety and ensure no significant thermodynamic signature is ignored.
- **Baseline Status:** This version serves as a baseline for software-driven atmospheric studies and does not yet replace operational models like SHIPS or HWRF.

---

## 📂 Project Structure & Module Tour

The architecture of **CycloneNet** follows a modular design inspired by enterprise systems, ensuring scalability and clear separation of concerns:

```text
cyclonenet/
│
├── requirements.txt          # System dependencies
├── .env                      # Environment & threshold configuration
├── pipeline.py               # Execute the complete forensic pipeline
│
├── notebooks/                # Persistent Audit Trail
│   └── evaluate_metrics.py   # Main orchestration hub (Unified Evaluator)
│
├── data/
│   └── raw/                  # Immutable Source Data
│       ├── hurdat2/          # NOAA HURDAT2 text files (Storm tracks & Ground Truth)
│       └── era5/             # Atmospheric Tensors (GRIB/NetCDF) from Copernicus CDS
│
├── src/                      # Source Code
│   ├── models/               # ANALYTICAL ENGINE (Prototype State)
│   │   ├── physics_model.py  # Core neural network prototype
│   │   ├── core.py           # Forecast logic & coordinate translation
│   │   └── train.py          # Model training utilities
│   │
│   ├── processor/            # Data Engineering Backbone
│   │   ├── downloaders.py    # Copernicus CDS API autonomous integration
│   │   ├── makers.py         # Raw data to structured Tensor Cube conversion
│   │   ├── metrics_handler.py# Confidence calibration & expert system rules
│   │   └── processors.py     # Data parsing & healing logic
│   │
│   ├── utils/                # System Utilities
│   │   └── config.py         # Centralized environment & global params
│   │
│   └── visualization/        # Forensic Evidence Generation
│       └── plotters.py       # Geospatial "Target Lock" mapping tools
│
└── outputs/                  # Persistent Audit Trail
    ├── figures/              # Visual geospatial evidence (Maps)
    ├── logs/                 # Execution traces (pipeline.log)
    └── predictions/          # Scientific CSVs (Target for ETL/BI)

```

---

### 🧭 **Development Roadmap & Call for Collaboration**

**Phase 1: Foundational Framework (Complete)**
A robust, reproducible forensic data pipeline with a hybrid analysis system. ✅

**Phase 2: Model Evolution & Research (Active)**
This is our current focus and we invite community collaboration:

- **Architecture Advancement:** Replacing the prototype model with more sophisticated architectures (e.g., deeper CNNs, Transformers) for improved pattern learning.
- **Physics-Guided ML:** Research into integrating physical constraints during training, moving towards a true _Physics-Informed Neural Network_.
- **Enhanced Explainability:** Implementing XAI techniques to make the system's "hotspot" detection more interpretable.

**Phase 3: Operational Integration (Future)**
Exploring real-time data stream integration and expanded basin analysis.

**We actively encourage contributions**, especially from those interested in the machine learning, atmospheric science, or data engineering challenges outlined in Phase 2. Let's evolve this forensic pipeline together.

---

## 🛠️ Setup & Copernicus API Integration

CycloneNet automates the ingestion of meteorological data. To enable the autonomous download feature:

1. **Create a CDS Account:** Register at [Copernicus Climate Data Store](https://cds.climate.copernicus.eu).
2. **Generate API Key:** Obtain your `UID` and `API Key` from your profile page.
3. **Configure Credentials:** Create a `.cdsapirc` file in your home directory (e.g., `C:\Users\Name\.cdsapirc` or `~/.cdsapirc`) as follows:

```text
url: https://cds.climate.copernicus.eu/api
key: YOUR_API_KEY

```

## 🗃️ Data Governance & HURDAT2 Updates

CycloneNet relies on the **NOAA HURDAT2** dataset for storm tracks and ground-truth intensity. To ensure the framework remains updated as the NHC releases new reanalysis data (typically every April), we use an environment-based configuration.

### How to update the dataset:

1. **Locate the latest link:** Visit the [NHC Data Archive](https://www.nhc.noaa.gov/data/hurdat/).
2. **Update your `.env`:** Change the `HURDAT2_URL` variable:

   ```env
   HURDAT2_URL=https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt
   ```

3. **Force a Refresh:** To bypass the local cache and download the latest version, you can call the download function with `force_download=True` or simply delete the local file in:
   `./data/raw/hurdat2/hurdat2.txt`

> [!NOTE]
> The current version (April 2025 release) includes finalized records for the **2024 Season**, enabling validation against record-breaking events like **Hurricane Milton** and **Hurricane Beryl**.

---

4. **Install Dependencies:**

```bash
pip install -r requirements.txt

```

---

## ⚙️ Environment Configuration (.env)

Before running the pipeline, create a `.env` file in the root directory:

- `CDS_URL`: The endpoint for the Copernicus Climate Data Store API.
- `CDS_KEY`: Your personal API Key for atmospheric data access (Required for ERA5/IFS ingestion).
- `HURDAT2_URL`: The official NOAA source link. This allows the framework to remain updated as new reanalysis data is released (e.g., the April 2025 release for 2024 consolidated data).
- `DATA_DIR`: Destination for ERA5 tensors (e.g., `./data`).
- `OUTPUT_DIR`: Storage for logs and scientific artifacts (e.g., `./outputs`).
- `GENERATE_VALIDATION_CSV`: Set to `True` to update the benchmark database.
- `RI_THRESHOLD`: Sensitivity gate for thermodynamic triggers (Default: 0.6).

> [!WARNING]
> Never commit your `.env` or `.cdsapirc` files to version control. Ensure they are listed in your `.gitignore`.

## ▶️ Running the Pipeline

The system is designed for end-to-end execution. Once configured, the **Unified Evaluator** will autonomously manage data ingestion, execute the "Data Healing" heuristics, and generate the forensic audit logs:

```bash
# 1. Execute the complete forensic pipeline
# This runs data download, processing, model inference, and generates the primary CSV log.

python pipeline.py

```

```bash
# 2. (Optional) Generate the final validation report and plots
# Run this after `pipeline.py` to compile metrics and create the summary report.

python notebooks/evaluate_metrics.py

```

> [!TIP] > **Data Traceability:** > The system generates persistent scientific artifacts in `./outputs/predictions/`. To ensure traceability, the latest run will always update `cyclonenet_scientific.csv`, which serves as the primary data source for the **BENCHMARK.md** report.

---

## 📜 License & Intellectual Property

This project is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Commercial use is strictly prohibited without explicit written authorization from the author.

© 2026 Estefano Senhor Ferreira
