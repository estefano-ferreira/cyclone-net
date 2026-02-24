# Interpreting CycloneNet Outputs

This guide explains the meaning of the main outputs produced by CycloneNet and how to interpret them scientifically.

## FuelMap

The **FuelMap** is a spatial heatmap (same resolution as the input patch) that highlights regions the model considers most likely to provide thermodynamic energy supporting rapid intensification.

- **Interpretation**: Higher values (closer to 1) indicate areas where the combination of warm SST, favorable wind patterns, and convergent flow is most consistent with intensification-supportive conditions. The FuelMap is **not** a causal statement; it is a diagnostic hypothesis based on learned patterns.

- **Peak Location**: The model computes the expected location (soft-argmax) of the FuelMap maximum, which is converted to geographic coordinates. This point represents the model's best guess of the primary energy source region at the given time.

- **Validation**: When TCHP data are available, the distance between the FuelMap peak and the actual TCHP maximum is reported. A small distance (< 100 km) indicates strong spatial agreement.

## Rapid Intensification (RI) Probability

The model outputs a probability (0–1) that the storm will undergo rapid intensification (≥30 kt in 24h) at the current time.

- **Interpretation**: This is a calibrated probability (if Platt calibration is applied). For example, a value of 0.8 means that among events with similar features, about 80% experienced RI.

- **Threshold**: A fixed threshold (selected on validation set to achieve high recall) is used to produce binary predictions. The default threshold is saved in `best_threshold.json`.

## Continuous Targets: dv12 and dv24

The model also predicts the expected wind speed change over the next 12 and 24 hours (in knots). These are regression outputs that can be used as additional diagnostics.

## Calibration Metrics

- **Reliability Diagram**: Plots observed frequency vs. predicted probability. A well-calibrated model follows the diagonal.
- **Expected Calibration Error (ECE)**: Average absolute difference between predicted probability and observed frequency.
- **Maximum Calibration Error (MCE)**: Maximum such difference.

Low ECE/MCE (<0.05) indicate good calibration.

## Spatial Metrics (with TCHP)

When full TCHP maps are available, the following metrics are computed:

- **Peak Distance**: Great-circle distance between FuelMap peak and TCHP maximum (km).
- **Top-10 Overlap**: Fraction of the top 10% highest FuelMap pixels that are also in the top 10% of TCHP pixels.
- **Rank Correlation**: Spearman correlation between FuelMap values and TCHP values over the patch.

These metrics quantify how well the model's attention aligns with actual ocean heat content.

## Limitations

- The FuelMap is a **proxy**; it does not directly measure energy fluxes.
- TCHP validation is only possible when data are available (≥1993, Atlantic basin).
- The model is trained on Atlantic hurricanes only; performance in other basins may differ.
- The model does not account for vertical wind shear or other atmospheric factors beyond the selected ERA5 fields.