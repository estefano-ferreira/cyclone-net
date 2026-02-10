# 🌪️ CycloneNet — Forensic Engineering for Atmospheric Disaster Analysis

**CycloneNet** is a specialized Atmospheric Auditing Framework designed to pinpoint the thermodynamic fuel sources of tropical cyclones. Unlike traditional predictive models, CycloneNet is a **Safety Engineering System** optimized for retrospective precision and a **High-Sensitivity Design Mandate**, establishing a high standard for the forensic analysis of extreme weather events.

### 👨‍💻 Developer's Vision

> "As a **Software Engineer** with over a decade of experience building robust, mission-critical systems for financial institutions—primarily using C# .NET—I’ve always been driven by a personal curiosity: What is the actual source of energy that makes a hurricane so powerful?
> I am not a physicist or a mathematician, and I don’t claim to be an AI theorist. My expertise lies in code and system architecture. I developed this framework by applying the same architectural rigor I use in the financial sector to atmospheric data, motivated by a real desire to identify and visualize the energy nodes that fuel these storms.
> To reach my goal of locating where a hurricane is feeding in real-time, I first had to master the past. CycloneNet uses a Forensic Engineering approach, auditing 18 of history’s most destructive hurricanes to ensure that our detection logic is flawless before it ever processes a live storm.
> I believe that by mapping these energy nodes with precision in historical data, we create a reliable, battle-tested foundation for real-time life-saving actions. This project is my contribution to that goal: a bridge between the lessons of the past and the resilience of future software engineering."
> — **Estefano Senhor Ferreira**

---

### 🛡️ The Forensic Edge (New Standards)

CycloneNet defines a new category in atmospheric studies by prioritizing **Safety-Critical Engineering over pure Statistics**:

- **Forensic Diagnostic Precision:** Optimized for retrospective accuracy, providing an auditable trail of how and where a storm fed during its intensification.
- **High-Sensitivity Mandate (Safety-First):** Engineered with a zero-miss philosophy. In disaster analysis, overlooking a signature is a failure. CycloneNet is tuned to identify potential thermodynamic triggers, ensuring high detection depth.
- **Thermodynamic Hotspot Auditing:** Generates verifiable geospatial evidence, mapping "Thermodynamic Singularities" with a mean tracking error of only **25.79 km** (sub-pixel precision relative to ERA5).
- **Automated Data Ingestion:** A resilient pipeline integrated with the **Copernicus CDS API**, ensuring that forensic audits are backed by high-quality, standardized atmospheric tensors.

---

### 💎 The Engineering Edge (Core Innovations)

Unlike purely statistical predictors, CycloneNet introduces critical software engineering innovations:

- **Automated ERA5 Pipeline:** Fully integrated with the **Copernicus CDS API**. The system autonomously handles data acquisition based on storm parameters.
- **Adaptive Data Healing:** A proprietary pipeline that automatically repairs telemetry gaps (NaNs) in ERA5 tensors using pattern recognition heuristics, ensuring data integrity even in noisy historical datasets.
- **Physics-Gated Neural Attention:** Our **Spatio-Temporal Attention (STA)** module is constrained by thermodynamic thresholds, preventing "black-box" hallucinations and focusing exclusively on energy-relevant regions.
- **Safety-Oriented Reliability:** Optimized for **High Recall**. In disaster management, a missed event is a catastrophic failure. CycloneNet is engineered with a fail-safe bias to capture the maximum number of potential RI signals for expert review.

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
├── evaluate_metrics.py       # Main orchestration hub (Unified Evaluator)
├── requirements.txt          # System dependencies
├── .env                      # Environment & threshold configuration
│
├── data/
│   └── raw/                  # Immutable Source Data
│       ├── hurdat2/          # NOAA HURDAT2 text files (Storm tracks & Ground Truth)
│       └── era5/             # Atmospheric Tensors (GRIB/NetCDF) from Copernicus CDS
│
├── src/                      # Source Code
│   ├── models/               # Intelligence & AI Logic
│   │   ├── core.py           # Neural Architecture (LSTM + Attention)
│   │   ├── physics_model.py  # Physics-Gated Logic (Thermodynamic constraints)
│   │   └── train.py          # Entry point for diagnostic evaluations
│   │
│   ├── processor/            # Data Engineering Backbone
│   │   ├── downloaders.py    # Copernicus CDS API autonomous integration
│   │   ├── makers.py         # Raw data to structured Tensor Cube conversion
│   │   ├── physics_model.py  # Physics-Gated Logic (Thermodynamic Constraints)
│   │   └── processors.py     # Adaptive Data Healing (Telemetry repair)
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

### 🔍 Module Descriptions

- **Orchestration (`evaluate_metrics.py`)**: The primary execution hub that runs the unified evaluator, processes data healing, and calculates scientific metrics.
- **Intelligence Layer (`src/models/`)**: Contains the **Physics-Gated Neural Attention** logic designed to constrain AI within thermodynamic boundaries.
- **Data Factory (`src/processor/`)**: Manages the end-to-end lifecycle of atmospheric data, from autonomous ingestion to the repair of telemetry gaps (NaNs).
- **Audit Trail (`outputs/`)**: A dedicated directory for immutable evidence. This is the core of our **Forensic Engineering** standard, ensuring every diagnostic run is traceable and verifiable.

---

## 🗺️ Strategic Roadmap

### **Phase 1: Forensic Foundation (1992–2024) [Completed]**

- **Achievement:** High-Sensitivity Design Mandate via Spatio-Temporal Attention.
- **Control:** Implementation of **Adaptive Data Healing** to handle historical ERA5 gaps.

### **Phase 2: Operational Stress Test (2025-2026) [In Progress]**

- **Blind Testing:** Validated against the April 2025 HURDAT2 release (including Hurricane Milton/Beryl).
- **Next Step:** Running against the 2025 preliminary season to verify AUC stability.
- **Confusion Matrix Audit:** Exporting raw FP/FN counts for academic transparency.
- **Model Maturity:** Achieved 0.9736 ROC-AUC, significantly reducing atmospheric noise while maintaining a safe detection margin (0.92 Recall).

### **Phase 3: Real-Time Energy Monitoring & Global Expansion [Future]**

- **Live Stream Integration:** Transition from ERA5 reanalysis to **HRES (High-Resolution)** and **IFS (Integrated Forecasting System)** live feeds for real-time diagnostic auditing.
- **Dynamic Energy Mapping:** Implementation of a live dashboard to visualize "Thermodynamic Feeding Nodes" as they develop in the Atlantic basin.
- **Causal Physics Nodes:** Moving beyond correlation to identify the specific causal triggers of RI using real-time atmospheric tensors.
- **Global Resilience:** Testing the architecture in the Pacific (Typhoon) and Indian Ocean basins to ensure a globally robust monitoring system.

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
# Execute the complete end-to-end pipeline
python pipeline.py

```

```bash
# Execute the Unified Evaluator to generate metrics
python notebooks/evaluate_metrics.py

```

> [!TIP] > **Data Traceability:** > The system generates persistent scientific artifacts in `./outputs/predictions/`. To ensure traceability, the latest run will always update `cyclonenet_scientific.csv`, which serves as the primary data source for the **BENCHMARK.md** report.

---

## 📜 License & Intellectual Property

This project is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Commercial use is strictly prohibited without explicit written authorization from the author.

© 2026 Estefano Senhor Ferreira
