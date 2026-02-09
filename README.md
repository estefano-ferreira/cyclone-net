# 🌪️ CycloneNet — Forensic Engineering for Atmospheric Disaster Analysis

**CycloneNet** is a specialized Atmospheric Auditing Framework designed to pinpoint the thermodynamic fuel sources of tropical cyclones. Unlike traditional predictive models, CycloneNet is a **Safety Engineering System** optimized for retrospective precision and **100% Recall**, establishing a new standard for the forensic analysis of extreme weather events.

---

### 👨‍💻 Developer's Vision

> "As a **Software Engineer** with over a decade of experience building robust, mission-critical systems for financial institutions—primarily using C# .NET—I’ve always been driven by a personal curiosity: What is the actual source of energy that makes a hurricane so powerful?
> I am not a physicist or a mathematician, and I don’t claim to be an AI theorist. My expertise lies in code and system architecture. I developed this framework by applying the same architectural rigor I use in the financial sector to atmospheric data, motivated by a real desire to identify and visualize the energy nodes that fuel these storms.
> To reach my goal of locating where a hurricane is feeding in real-time, I first had to master the past. CycloneNet uses a Forensic Engineering approach, auditing 18 of history’s most destructive hurricanes to ensure that our detection logic is flawless before it ever processes a live storm.
> I believe that by mapping these energy nodes with precision in historical data, we create a reliable, battle-tested foundation for real-time life-saving actions. This project is my contribution to that goal: a bridge between the lessons of the past and the resilience of future software engineering."
> — **Estefano Senhor Ferreira**

---

## 🛡️ The Forensic Edge (New Standards)

CycloneNet defines a new category in atmospheric studies by prioritizing **Security over Statistics**:

- **Forensic Diagnostic Precision:** Optimized for retrospective accuracy, providing an auditable trail of how and where a storm fed during its intensification.
- **100% Recall (Safety-First):** Engineered with a zero-miss mandate. In disaster analysis, overlooking a signature is a failure. CycloneNet identifies every potential thermodynamic trigger.
- **Thermodynamic Hotspot Auditing:** Generates verifiable geospatial evidence for 18 high-impact hurricanes (2004-2024), mapping the "Thermodynamic Singularities" with a mean tracking error of only **26.03 km**.
- **Automated Data Ingestion:** A resilient pipeline integrated with the **Copernicus CDS API**, ensuring that forensic audits are backed by high-quality, standardized atmospheric tensors.

---

## 💎 The Engineering Edge (Core Innovations)

Unlike purely statistical RI predictors, CycloneNet introduces four critical software engineering innovations:

- **Automated ERA5 Pipeline:** Fully integrated with the **Copernicus Climate Data Store (CDS) API**. The system autonomously handles data acquisition, downloading necessary atmospheric tensors based on storm parameters.
- **Adaptive Data Healing:** A proprietary pipeline that automatically repairs telemetry gaps (NaNs) in ERA5 tensors using pattern recognition heuristics, ensuring 100% record integrity even in noisy or corrupted datasets.
- **Physics-Gated Neural Attention:** Our **Spatio-Temporal Attention (STA)** module is constrained by thermodynamic thresholds, preventing "black-box" hallucinations and focusing exclusively on energy-relevant regions.
- **Zero-Miss Reliability:** Optimized for **1.000 Recall**. In disaster management, a missed event is a catastrophic failure. CycloneNet is engineered with a fail-safe bias to detect every potential RI signal.

---

## 📊 Scientific Benchmark (2026 Results)

Validated against 18 of the most destructive Atlantic hurricanes (2004–2024), demonstrating **operational-grade precision**.

| Metric                   | Result       | Interpretation                                  |
| :----------------------- | :----------- | :---------------------------------------------- |
| **ROC-AUC**              | **0.9094**   | Outstanding predictive power.                   |
| **Recall (Sensitivity)** | **1.0000**   | **Zero Miss Rate:** Detected 100% of RI events. |
| **Brier Score**          | **0.1315**   | Reliable probability calibration.               |
| **Mean Tracking Error**  | **26.03 km** | High-precision spatial localization.            |

> [!IMPORTANT] > **[View the Full Scientific Validation Report & Per-Storm Analysis](./BENCHMARK.md)**

---

## ⚠️ Disclaimer & Technical Limitations

As an engineering-first project, it is crucial to maintain transparency regarding the current scientific boundaries of this framework. **CycloneNet 1.0** is a proof-of-concept for data pipeline architecture and should be evaluated under the following constraints:

- **Diagnostic vs. Predictive Scope:** The current results are based on **hindcast validation**. The model functions as a **high-fidelity diagnostic tool**, mapping signatures in historical data. It has not yet been benchmarked for prospective real-time forecasting.
- **Temporal Coupling (Data Leakage):** Current evaluation metrics may be influenced by temporal proximity between feature selection and event occurrence. This architecture is designed to test **spatial association** rather than predictive lead-time.
- **Engineering Focus:** The core innovation of this repository lies in the **Data Engineering pipeline** (ERA5 autonomous ingestion, telemetry healing, and tensor cubing) rather than meteorological breakthrough.
- **Safety-Oriented Bias:** The **1.000 Recall** is a result of a conservative diagnostic gate. In an operational environment, this may lead to false positives (as seen in the _Hurricane Isaac_ case), which are intended to prioritize safety over precision.
- **Baseline Status:** This version does not include direct comparisons with traditional meteorological models (e.g., SHIPS or LGEM). It serves as a baseline for future software-driven atmospheric studies.

## 📂 Project Structure & Module Tour

The architecture of **CycloneNet** follows a modular design inspired by enterprise systems, ensuring scalability and clear separation of concerns:

```text
cyclonenet/
│
├── evaluate_metrics.py       # Main orchestration hub (Unified Evaluator)
├── requirements.txt          # System dependencies
├── .env                      # Environment & threshold configuration
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

```

---

### 🔍 Module Descriptions

- **Orchestration (`evaluate_metrics.py`)**: The primary execution hub that runs the unified evaluator, processes data healing, and calculates scientific metrics.
- **Intelligence Layer (`src/models/`)**: Contains the **Physics-Gated Neural Attention** logic designed to constrain AI within thermodynamic boundaries.
- **Data Factory (`src/processor/`)**: Manages the end-to-end lifecycle of atmospheric data, from autonomous ingestion to the repair of telemetry gaps (NaNs).
- **Audit Trail (`outputs/`)**: A dedicated directory for immutable evidence. This is the core of our **Forensic Engineering** standard, ensuring every diagnostic run is traceable and verifiable.

---

## 🗺️ CycloneNet Strategic Roadmap

The development of CycloneNet is structured into three evolutionary horizons, designed to transform forensic engineering rigor into real-time life-saving intelligence:

### **Phase 1: Forensic Engineering Foundation (Completed)**

- **Focus:** Establishing a zero-miss diagnostic gate using historical data (2004–2024).
- **Achievement:** Audited 18 high-impact hurricanes with a **1.000 Recall** and a **26.03 km** mean tracking error.
- **Architecture:** Developed the automated ERA5 ingestion pipeline and Adaptive Data Healing system.

### **Phase 2: Real-Time Diagnostic Mapping (In Progress)**

- **Goal:** Transitioning the engine from "past-analysis" to "now-analysis".
- **Action:** Integrating **Live Satellite Feeds** and **IFS Operational Data** to replace ERA5 reanalysis.
- **Output:** Generating instant "Target Lock" maps to assist in real-time emergency decision-making.

### **Phase 3: Predictive Intelligence (Future)**

- **Goal:** Utilizing high-fidelity mapping to identify patterns _before_ they manifest as intensification.
- **Target:** Achieving a **6–12 hour lead-time** for Rapid Intensification (RI) alerts.
- **Expansion:** Validating the model across Pacific and Indian Ocean basins to create a global safety framework.

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

4. **Install Dependencies:**

```bash
pip install -r requirements.txt

```

---

## ⚙️ Environment Configuration (.env)

Before running the pipeline, create a `.env` file in the root directory:

- `DATA_DIR`: Destination for ERA5 tensors (e.g., `./data`).
- `OUTPUT_DIR`: Storage for logs and scientific artifacts (e.g., `./outputs`).
- `GENERATE_VALIDATION_CSV`: Set to `True` to update the benchmark database.
- `RI_THRESHOLD`: Sensitivity gate for thermodynamic triggers (Default: 0.6).
- `LOG_VERSION`: Versioning for traceability (e.g., `2026-02-08`).

> [!WARNING]
> Never commit your `.env` or `.cdsapirc` files to version control. Ensure they are listed in your `.gitignore`.

## ▶️ Running the Pipeline

The system is designed for end-to-end execution. Once configured, it will download, process, and evaluate the storm data automatically:

```bash
# Run the Unified Evaluator (Includes Data Healing & Metric Calculation)
python ./notebooks/evaluate_metrics.py

```

---

> [!IMPORTANT] > **Data Traceability & Log Configuration:**
> The evaluation script is linked to a specific version of the scientific logs. If you are running a new validation or using historical data, ensure the `raw_csv` path in your evaluation script matches your generated file:
>
> ```python
> # Located in your evaluation/benchmark script
> raw_csv = os.path.join(base_path, '..', 'outputs', 'cyclonenet_scientific_2026-02-08.csv')
>
> ```

## 📜 License & Intellectual Property

This project is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.

Commercial use is strictly prohibited without explicit written authorization from the author.

© 2026 Estéfano Senhor Ferreira
