# FuelMap × TCHP Co-location Validation

**Date:** 2026-07-11 · **Status: PRELIMINARY** · Protocol and results produced by
`analysis/fuelmap_tchp_validation.py` (seed 42, 10,000 Monte Carlo replicates);
machine-readable report at `outputs/results/spatial/fuelmap_tchp_validation.json`.

## Hypothesis under test

The coordinate predicted by the model's FuelMap head ("target lock",
`pred_lat`/`pred_lon` in `test_predictions.csv`) co-locates with an
**independent** physical reference: the audited TCHP (Tropical Cyclone Heat
Potential) peak extracted by `run.py preprocess-tchp` from NOAA/AOML gridded
fields. This is a **plausible-correlate hypothesis** — a small p-value would
mean the FuelMap tends to sit nearer the subsurface ocean-heat maximum than a
null reference. It would **not** establish that the model causally uses
subsurface ocean heat (the model never sees TCHP as input).

## Availability audit (reported before any conclusion)

| Quantity | Count |
|---|---:|
| Test predictions | 155 |
| With audited TCHP peak (public gridded TCHP exists 2022+ only) | 62 |
| **Eligible RI-positive events** | **4** |

With **n = 4** RI events, the RI-only analysis is **severely underpowered and
strictly illustrative**. The all-events analysis (n = 62) has moderate power
for coarse effects only.

## Results — all eligible events (n = 62)

| Metric | Value |
|---|---|
| Median FuelMap → TCHP-peak distance | 456.6 km |
| Fraction within 100 km | 3.2% |
| Fraction within 2° (~222 km) | 21.0% |

### Null 1 — random point inside each event's window (Monte Carlo, B = 10,000)

Null median distance: 583.2 km (95% band 489.0–676.9 km).
Observed 456.6 km → **p = 0.0043** (one-sided). The FuelMap is closer to the
TCHP peak than a uniformly random point in the same window.

### Null 2 — storm-center baseline (sign-flip permutation, B = 10,000)

Median center → TCHP distance: 512.3 km. Mean paired difference (FuelMap −
center): −10.6 km; FuelMap closer in 54.8% of events → **p = 0.349**
(one-sided). **The FuelMap is NOT demonstrably better than simply predicting
the storm center.**

### Three-way reference (reused from `spatial_metrics.skill_comparison`, n = 62)

| Reference | Mean distance to TCHP peak |
|---|---:|
| Learned FuelMap | 463.5 km |
| Storm center (naive baseline) | 474.1 km |
| Physics-only enthalpy-flux peak (no NN) | 560.9 km |

## Results — RI-only subset (n = 4, illustrative only)

Median 894.8 km; worse than both nulls (random-point p = 0.985; vs center
p = 0.937, FuelMap closer in 1/4 events). No conclusion should be drawn from
four events beyond: the current data cannot validate FuelMap co-location for
RI cases.

## Honest interpretation

1. The FuelMap carries **some** spatial information (beats the random-point
   null, p ≈ 0.004): it is not placing its peak arbitrarily inside the window.
2. That information is **not distinguishable from storm location**: it does
   not beat the storm-center baseline (p ≈ 0.35, median improvement ≈ 56 km,
   mean ≈ 11 km). A model that simply pointed at the storm center would do
   statistically as well against TCHP.
3. For the RI events that motivate the FuelMap narrative, the sample (n = 4)
   is far too small to validate anything.

**Bottom line:** the co-location hypothesis is *not supported beyond storm
position* at current sample sizes. This is consistent with, and extends, the
correction record in `ERRATA.md`. Any stronger claim about FuelMap↔OHC
correspondence requires more eligible events (more TCHP years, or derived OHC
from reanalysis for pre-2022 seasons).

## Limitations

- TCHP reference restricted to 2022+ public gridded data (93/155 test events
  ineligible); eligibility itself may correlate with season/basin.
- Random-point null samples uniformly in the lat/lon box (cos-latitude area
  distortion over a ±5° window is small but nonzero).
- Single checkpoint, trained before the physics-loss reconnection (see
  `docs/diagnostic.md`); results may differ after retraining.
- Distances use great-circle (haversine) on the audited peak location only;
  field-level agreement (rank correlation, top-k overlap) is not evaluated
  here.
