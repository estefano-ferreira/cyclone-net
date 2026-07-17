# FuelMap × TCHP Co-location Validation

**Dates:** 2026-07-11 (preliminary run, n = 62) · 2026-07-12 (full re-run,
n = 226) · **Status: CLOSED — validated negative (ERRATA item 4)** · Protocol
and results produced by `analysis/fuelmap_tchp_validation.py` (seed 42, 10,000
Monte Carlo replicates); machine-readable report at
`outputs/results/spatial/fuelmap_tchp_validation.json` (carries the full
re-run; the preliminary run's numbers are preserved only in the dated section
at the end of this document).

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
| Test predictions | 2,679 |
| With audited TCHP peak (public gridded TCHP exists 2022+ only) | 226 |
| **Eligible RI-positive events** | **8** |

With **n = 8** RI events, the RI-only analysis remains **underpowered and
strictly illustrative**. The all-events analysis (n = 226) is adequately
powered for the effects tested here.

## Results — all eligible events (n = 226)

| Metric | Value |
|---|---|
| Median FuelMap → TCHP-peak distance | 539.3 km |
| Fraction within 100 km | 0.4% |
| Fraction within 2° (~222 km) | 5.8% |

### Null 1 — random point inside each event's window (Monte Carlo, B = 10,000)

Null median distance: 634.9 km (95% band 581.6–687.4 km).
Observed 539.3 km → **p = 0.0003** (one-sided). The FuelMap is closer to the
TCHP peak than a uniformly random point in the same window.

### Null 2 — storm-center baseline (sign-flip permutation, B = 10,000)

Median center → TCHP distance: 561.1 km. Mean paired difference (FuelMap −
center): −6.5 km; FuelMap closer in 46.0% of events → **p = 0.302**
(one-sided). **The FuelMap is NOT demonstrably better than simply predicting
the storm center.**

### Three-way reference (`three_way_skill.skill_comparison`, n = 226)

| Reference | Mean distance to TCHP peak |
|---|---:|
| Learned FuelMap | 527.1 km |
| Storm center (naive baseline) | 533.6 km |
| Physics-only enthalpy-flux peak (no NN) | 586.1 km |

## Results — RI-only subset (n = 8, illustrative only)

Median 610.2 km; beats neither null (random-point p = 0.395; vs center
p = 0.873, FuelMap closer in 3/8 events). No conclusion should be drawn from
eight events beyond: the current data cannot validate FuelMap co-location for
RI cases.

## Honest interpretation

1. The FuelMap carries **some** spatial information (beats the random-point
   null, p = 0.0003): it is not placing its peak arbitrarily inside the
   window.
2. That information is **not distinguishable from storm location**: it does
   not beat the storm-center baseline (p ≈ 0.30, median improvement ≈ 22 km,
   mean ≈ 6.5 km). A model that simply pointed at the storm center would do
   statistically as well against TCHP.
3. For the RI events that motivate the FuelMap narrative, the sample (n = 8)
   remains far too small to validate anything.

**Bottom line:** the co-location hypothesis is *not supported beyond storm
position*. This is the static angle of the three-angle validated negative
recorded in `ERRATA.md` item 4 (with the dynamic displacement control and the
physics-prior control). Any stronger claim about FuelMap↔OHC correspondence
requires more eligible events (more TCHP years, or derived OHC from reanalysis
for pre-2022 seasons).

## Preliminary run (2026-07-11) — superseded, preserved as record

The first pass ran on the 2020–2023 public-release test split: 155 test
predictions, **n = 62** eligible, **n = 4** RI-positive. Results: median
FuelMap → TCHP distance 456.6 km (within 100 km 3.2%, within ~222 km 21.0%);
random-point null median 583.2 km (95% band 489.0–676.9 km), p = 0.0043; vs
storm center: center median 512.3 km, mean paired difference −10.6 km, FuelMap
closer in 54.8% of events, p = 0.349; three-way means (n = 62) FuelMap
463.5 km / center 474.1 km / physics-only 560.9 km; RI-only (n = 4): median
894.8 km, random-point p = 0.985, vs center p = 0.937, closer in 1/4 events.
Same conclusion as the full re-run, at lower power. The JSON at the artifact
path was overwritten by the 2026-07-12 re-run; the preliminary numbers are
preserved only in this section.

## Limitations

- TCHP reference restricted to 2022+ public gridded data (2,453 of 2,679 test
  events have no audited TCHP peak); eligibility itself may correlate with
  season/basin.
- Random-point null samples uniformly in the lat/lon box (cos-latitude area
  distortion over a ±5° window is small but nonzero).
- Single checkpoint: predictions are the released production model's one
  authorized test-set read (`test_predictions.csv`, 2,679 rows). The
  preliminary run used an earlier checkpoint trained before the physics-loss
  reconnection (see `docs/diagnostic.md`).
- Distances use great-circle (haversine) on the audited peak location only;
  field-level agreement (rank correlation, top-k overlap) is not evaluated
  here.
