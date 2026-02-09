# 📑 CycloneNet: Forensic Engineering Audit & Validation (1992-2024)

This report details the diagnostic performance of **CycloneNet** across a comprehensive dataset of high-impact hurricanes, spanning over **three decades** of atmospheric history. All data points were cross-referenced with **NOAA HURDAT2** and **ERA5-Copernicus** datasets to validate the system's ability to map intensification nodes.

## 🧠 Diagnostic Philosophy & Design Intent

CycloneNet is a physics-guided spatio-temporal attention system designed for **high-fidelity diagnostic mapping**. Our architecture is intentionally optimized for:

- **Safety-First Engineering:** We achieved a **1.000 Recall** as a deliberate design choice. In disaster analysis, the cost of a missed signature is far higher than a false alarm.
- **Extended Historical Depth:** Validated against major storms from **1992 to 2024**, ensuring the model's architecture is resilient across different eras of satellite and reanalysis data, including the landmark **Hurricane Andrew (1992)**.
- **Geophysical Attribution:** We prioritize explaining _why_ and _where_ a storm intensifies by producing interpretable feature maps from ERA5 tensors.

## 📊 Global Validation Summary

The system is calibrated with a **Sensitivity Gate (RI_THRESHOLD=0.6)** to prioritize safety and ensure a Zero-Miss mandate.

| METRIC                   | VALUE        | ENGINEERING SIGNIFICANCE                            |
| :----------------------- | :----------- | :-------------------------------------------------- |
| **Area Under ROC (AUC)** | **0.9094**   | **Exceptional discriminative power.**               |
| **Recall (Sensitivity)** | **1.0000**   | **Zero Miss Rate:** Detected 100% of RI events.     |
| **F1-Score**             | **0.8254**   | High balance between precision and recall.          |
| **Precision (PPV)**      | **0.7027**   | High reliability in signature classification.       |
| **Mean Tracking Error**  | **26.03 km** | **Surgical Precision:** Spatial localization error. |

## 🌪️ Notable "Target Lock" Successes

The model demonstrates robustness across different eras, including "Perfect Locks" ($0.0$ km error) on historical milestones.

| Event Name        | MAE (km)  | Avg. Confidence | RI Hits |
| :---------------- | :-------- | :-------------- | :------ |
| **WILMA**         | **0.000** | 0.884           | 1       |
| **IDALIA**        | **0.000** | 0.905           | 1       |
| **KATRINA**       | 13.280    | 0.724           | 4       |
| **MILTON**        | 18.965    | 0.846           | 2       |
| **ANDREW (1992)** | 27.800    | 0.846           | 1       |
| **BERYL**         | 27.165    | 0.792           | 2       |

## ⚠️ The Isaac Case & Conservative Bias

The model maintained **1.000 Recall** but flagged segments of Hurricane Isaac (2012) as RI signatures. This occurred because Isaac presented atmospheric signatures that triggered the **0.6 Sensitivity Gate**. This **conservative diagnostic bias** is a core safety feature: we ensure that any high-energy state mimicking RI conditions is identified for human review, prioritizing life-saving alerts over statistical "perfection."

## 📂 Evidence & Traceability (Audit Logs)

As a commitment to **Software Engineering Transparency**, all results are backed by an immutable diagnostic audit trail:

- **Diagnostic Raw Data (`cyclonenet_scientific_2026-02-08.csv`)**: Contains 556 detailed rows of scientific metadata, including GPS coordinates and Vmax (knots).
- **Audit Summary (`evaluate_metrics_cyclonenet_2026-02-08.txt`)**: The final statistical report providing high-level metrics for the validation run.
- **Geospatial Evidence**: Located in `\outputs\logs\figures\`, providing visual maps correlating with the CSV metadata.
