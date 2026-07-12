# Case Studies

> ⚠️ **SUPERSEDED — do not cite.** The examples below were produced by an earlier
> pipeline revision and are **not reproducible** from the current model or dataset.
> In particular, the FuelMap-vs-TCHP distances shown (tens of km) are inconsistent
> with the validated result on the current test set (median 539 km, no skill beyond
> the storm-centre baseline; see `docs/fuelmap_validation.md` and `ERRATA.md` item 4),
> and gridded TCHP is not publicly available before 2022, so the pre-2022 comparisons
> below cannot be independently verified. Real, reproducible test-set examples are in
> [`BENCHMARK.md`](../BENCHMARK.md). This page is retained only as a historical record
> pending replacement.

This page presents selected examples of CycloneNet predictions, including both successes and failures, to illustrate model behavior and limitations.

## Success Cases

### Hurricane Katrina (2005-08-27 06:00 UTC)

- **RI observed**: Yes (Δ24 = 45 kt)
- **Predicted probability**: 0.95
- **FuelMap peak**: 25.3°N, 87.5°W
- **TCHP maximum**: 25.1°N, 87.2°W
- **Distance**: 34 km

![Katrina FuelMap](figures/katrina_20050827_fuelmap.png)

*Interpretation*: The model correctly identifies RI and places the FuelMap peak very close to the actual TCHP maximum in the Gulf of Mexico's Loop Current.

### Hurricane Michael (2018-10-10 12:00 UTC)

- **RI observed**: Yes (Δ24 = 50 kt)
- **Predicted probability**: 0.86
- **FuelMap peak**: 29.8°N, 85.6°W
- **TCHP maximum**: 29.5°N, 85.9°W
- **Distance**: 41 km

![Michael FuelMap](figures/michael_20181010_fuelmap.png)

*Interpretation*: The model captures the region of high ocean heat content ahead of the storm.

## Failure Cases

### Hurricane Isaac (2012-08-27 12:00 UTC)

- **RI observed**: No
- **Predicted probability**: 0.32 (false positive)
- **FuelMap peak**: 26.1°N, 89.2°W
- **TCHP maximum**: 26.0°N, 89.0°W
- **Distance**: 18 km

*Interpretation*: Although RI did not occur, the model identified a region of high TCHP. This is an example of the forensic design: the model flags potential energy sources even when intensification does not materialize due to other factors (e.g., wind shear).

### Hurricane Gonzalo (2014-10-16 00:00 UTC)

- **RI observed**: Yes (Δ24 = 35 kt)
- **Predicted probability**: 0.12 (false negative)
- **FuelMap peak**: 27.5°N, 65.3°W
- **TCHP maximum**: 26.8°N, 64.7°W
- **Distance**: 112 km

*Interpretation*: The model missed this RI event. The FuelMap peak is relatively far from the TCHP maximum, suggesting that the model failed to associate the ocean heat with the storm's intensification. This may be due to unusual atmospheric conditions not captured by the input fields.