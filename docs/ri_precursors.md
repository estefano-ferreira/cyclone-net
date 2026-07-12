# RI Precursors — Exploratory, Hypothesis-Generating Analysis

**Date:** 2026-07-12 · **Status: HYPOTHESIS GENERATION, not confirmed discovery.**
Protocol and numbers: `analysis/ri_precursors.py` (seed 42, pre-registered
hypotheses, sign-flip permutation nulls, Bonferroni ×4 on primary tests);
machine-readable results in `outputs/results/ri_precursors.json`.

## Question

Do environmental conditions in the 24 h **before** rapid-intensification
onset differ from intensity/basin-matched non-RI moments? Exactly four
physically motivated hypotheses were tested — no variable scanning.

## Design

- **t = 0 — RI onset:** first 6-hourly point of an RI episode (dv24 crosses
  the pipeline's +30 kt/24 h criterion; Kaplan & DeMaria 2003). The event
  cube's five time slices provide the window (0, −6, −12, −18, −24 h).
- **Controls:** non-RI points matched 1:1 on initial intensity (±10 kt,
  widened to ±15 kt when needed) and basin (when known), excluding points
  within 24 h before an onset of their own storm. Seeded, no replacement.
- **Statistics:** paired differences, two-sided sign-flip permutation
  (10,000×) as the explicit exchangeability null; Cliff's δ as effect size;
  Bonferroni ×4 on each hypothesis's primary test. Per-lag rows are
  descriptive (not multiplicity-corrected).

## Availability audit (before any conclusion)

| Quantity | N |
|---|---:|
| RI-positive points (1980–2023) | 1,875 |
| **RI onsets** (episode starts) | **583** |
| Matched pairs with complete artifacts | **394** |
| Pairs usable for H2 (shear) / H4 (rh_mid) | **5** — pressure-level channels exist only for 2020–2023 events |

H2 and H4 are therefore **severely underpowered and untested in practice**;
they become testable if the historical pressure-level archive is downloaded.

## Results — hypothesis × lag × effect × p

**H1 — Pressure fall (min MSLP, hPa; onset − control):** primary = 24 h fall
difference: **−1.61 hPa, p = 0.0001, p·4 = 0.0004 → DETECTABLE.**

| Lag | Δ pareado (hPa) | Cliff's δ | p (descritivo) |
|---|---:|---:|---:|
| −24 h | +2.34 | +0.21 | 0.0001 |
| −18 h | +2.30 | +0.20 | 0.0001 |
| −12 h | +1.56 | +0.12 | 0.0003 |
| −6 h | +1.42 | +0.11 | 0.0008 |
| 0 h | +0.74 | +0.02 | 0.076 |

Reading: at equal wind intensity, RI-bound storms have **higher** central
pressure 24 h out and **converge to parity by onset** — i.e., they are
already deepening ~1.6 hPa/24 h faster, a pressure-wind "catch-down"
signature visible from at least 24 h before onset.

**H2 — Deep-layer shear: UNDERPOWERED (n = 5).** No verdict possible.

**H3 — SST (patch-mean, K):** primary = level at −24 h: **+1.44 K,
Cliff's δ = +0.53 (large), p = 0.0001, p·4 = 0.0004 → DETECTABLE.**
The difference is flat across all lags (+1.43 to +1.45 K, δ ≈ 0.53) — a
**standing environmental condition**, not a time-evolving precursor:
RI-bound storms sit over substantially warmer water throughout the window.

**H4 — Mid-level humidity: UNDERPOWERED (n = 5).** No verdict possible.

## Honest verdicts

1. **H1 detectable from ≥24 h before onset** (small-to-moderate effect,
   δ ≈ 0.2). Caveat: partially explainable by intensification persistence
   (storms already deepening tend to continue) — still useful as a
   precursor signal, but not evidence of a distinct mechanism.
2. **H3 detectable at all lags with a large effect (δ ≈ 0.53)** — consistent
   with established RI science (warm SST as a necessary ingredient), which
   is a reassuring sanity check rather than novelty. Caveat: matching did
   not control latitude/season, and SST covaries with both; part of the
   effect may be positional rather than causal.
3. **H2/H4 remain open** — testable only after historical pressure-level
   coverage (currently 2020–2023 only).

## H1 — control test: independent precursor or circularity?

Pressure and wind are anticorrelated, so H1 could merely re-detect the
intensification already in progress. Control analysis
(`analysis/ri_precursors_h1_control.py`, same 394 matched pairs; 245 usable
after requiring best-track wind history at t−24/−6 h — 149 dropped, audited):

**Test 1 — conditional (paired) logistic, trends measured strictly
pre-onset (t−24 h → t−6 h):**

| Covariate (scaled Δ, onset − control) | β | p (permutation, within-pair swap) |
|---|---:|---:|
| Wind trend | **+0.97** | 0.0005 |
| Pressure trend | **−0.32** | **0.023** |

The wind trend dominates — RI-bound storms were already intensifying
**+8.2 kt more** than controls before onset, so most of H1 is indeed the
intensification itself. But the pressure trend **retains a modest
independent association** (β ≈ −0.32, ~⅓ of the wind coefficient,
p = 0.023) — pressure falling faster than the concurrent wind trend implies
carries some additional signal. Direction is consistent across wind-trend
strata (robustness check).

**Test 2 — temporal lead:** the per-lag profiles of wind level (δ = −0.23
at t−24 h) and pressure level (δ = +0.19 at t−24 h) grow **simultaneously**
— pressure does not lead wind. In the t−24 h conditional model (pressure
level given wind level + wind trend), the pressure coefficient is **not
significant** (β = +0.23, p = 0.12).

**Revised verdict for H1:** **mostly circular, with a modest independent
residual.** The detectable "pressure precursor" is primarily the ongoing
intensification visible in the wind record; after controlling for it, a
small independent pressure-trend signal survives (p = 0.023, n = 245 —
fragile; treat as a weak generated hypothesis, possibly reflecting
pressure-wind imbalance during spin-up), and there is **no evidence of
temporal precedence** of pressure over wind.

## Corrections and limitations (disclosed)

- The stored `sst_anom_K` channel is a **spatial** anomaly (SST − patch
  mean), whose patch average is ~0 by construction; H3's first
  operationalization aggregated it and tested nothing. H3 was re-run on raw
  patch-mean SST — this is an operationalization fix of a pre-registered
  hypothesis, disclosed here.
- Controls come from the same best-track population; no reanalysis-wide
  climatological null was used.
- In the H1 control, the first implementation mean-centered the within-pair
  differences before the conditional logistic fit — which removes the paired
  signal by construction (all betas identically zero). Caught by a sanity
  inspection of the exact-zero coefficients and fixed to scale-only
  normalization; disclosed here.
- **This is observational, hypothesis-generating analysis:** detected
  signals require prospective validation and do not imply causality.
