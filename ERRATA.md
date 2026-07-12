# Erratum & Correction Note — CycloneNet V2 preprint

**Paper:** *CycloneNet V2: A Forensic Physics-Guided Deep Learning Framework for Atmospheric Singularity Mapping* (Feb 23, 2026).

This note corrects discrepancies between the preprint and the publicly released
code/data, found during an audit of the pipeline. It is published in the interest of
scientific integrity and reproducibility. None of the corrections below indicate
fabrication; they are release/reproducibility errors and an over-broad framing that
this note makes precise.

---

## 1. The "physics-guided" losses were inactive in the released code

**What the paper says:** the model is physics-guided via a KL FuelMap-alignment loss
and an equation-consistency loss, "added when the required tensors are present."

**What the released code actually did:** the training configuration did not define the
`training.physics.*` loss weights, so every physics-guided term defaulted to weight
`0.0`. The released model was, in practice, a plain 3D-CNN classifier with **no active
physics-guided constraint**. The configuration flags that appeared to enable them
(`use_prior_loss`, `use_consistency_loss`, `use_tv_l1`) were not read by the trainer.

**Correction:** the loss terms are now correctly wired and active by default
(`training.physics` block). Additionally, we note that the equation-consistency term is
**near-degenerate by construction** — it compares vorticity/divergence recomputed from
the wind field against stored channels derived from the *same* wind field — and it is
therefore documented as a weak representational regularizer, not a physical constraint.
It is disabled by default.

## 2. The operating threshold did not match the described methodology

**What the paper says:** a validation threshold targeting recall ≥ 90% (value 0.0666),
applied unchanged to the test set.

**What the released code did:** the threshold was selected by maximizing F1, not by
the recall target. The reported value (0.0666) is not reproducible from the released
threshold-selection routine.

**Correction:** threshold selection now follows a configurable policy
(`precision_at_recall`, honoring `training.eval_target_recall`), consistent with the
paper's stated forensic high-recall intent.

## 3. The reported metrics were NOT reproducible from the public repository — **RESOLVED**

**What the paper reports:** a held-out test set of **2,193 samples (211 RI positives)**,
with **ROC-AUC = 0.83, recall = 0.905**.

**What the public repository contained at audit time:** a dataset covering **2020–2023
only** (972 valid events; test split = **155 samples, 9 RI positives**). The headline
metrics **could not be reproduced** from the released code and data. On that public
dataset, with the corrected (physics-active) pipeline, test ROC-AUC was on the order of
**~0.6**, on a test set with too few positives (9) to support a stable estimate.

**Cause:** the headline numbers derived from a larger dataset that was not included in
the public release.

**Resolution (option (a) — full data release):** the repository now builds and releases
the **full 1980–2023** dataset (16,780 valid events / 802 RI positives / 992 storms) via
a windowed, checksummed, provenance-manifested pipeline, with hash-deterministic
storm-level splits. The model was retrained deterministically on this dataset (the
committed checkpoint was reproduced digit-for-digit from scratch), and the headline
metrics are **replaced** by numbers reproducible from the public artifacts:
**ROC-AUC 0.796 [95% CI 0.753–0.837], PR-AUC 0.251 [0.179–0.331], recall 0.852** on a
test split of 2,679 events (115 RI positives). Both confidence intervals sit entirely
above chance — the first release of this project for which that is true. The original
0.83/0.905 figures remain non-reproducible and are superseded; they should not be cited.

## 4. FuelMap localization does not beat a trivial baseline (validated negative, three angles)

The paper correctly **does not claim** externally validated localization. Validations
added during and after the audit make the negative result robust from three independent
angles:

1. **Static TCHP validation** (n = 226 eligible test events, TCHP publicly gridded
   2022+): the FuelMap peak beats a random-point null (p = 0.0003) but does **not**
   locate the TCHP peak better than a naive "storm-centre" baseline — median 539 km vs
   561 km, closer in only 46% of events (p = 0.30, sign-flip permutation).
2. **Dynamic displacement test:** the FuelMap's apparent collapse-toward-centre during
   RI, initially a candidate signal, was tested against a control.
3. **Physics-prior control:** re-running the displacement analysis on the pure
   enthalpy-flux prior (no learned weights) reproduces the dynamic behavior — it is
   arithmetic of the physics prior, **not** learned skill.

The localization claim is therefore **unsupported**, not merely "pending." A
counterfactual ablation (occluding the FuelMap region vs. an equal-size control) does
show the model's RI prediction depends on the identified region (paired test, p ≈ 1e-6) —
but this demonstrates *model-internal* dependence, **not** that the region is the true
physical energy source. Full protocols: `docs/fuelmap_validation.md`.

## 5. Novelty positioning

A literature check (operational and research) indicates that the components and the
forensic/diagnostic framing are **established**, not new:
- Deep learning for RI, including CNNs over the North Atlantic / East Pacific.
- Altimetry-derived ocean heat content (SSHA/ADT → TCHP) as an RI predictor
  (foundational AOML work; TCHP is already a SHIPS-RII predictor).
- Physics-informed / interpretable ML for TC intensity (spatial attention, PINNs,
  transformer interpretability).
- Identifying/quantifying the ocean energy source is operational via TCHP/OHC products
  and coupled ocean–atmosphere models (HWRF, HAFS) that compute the air–sea enthalpy
  flux directly.

**Correction to framing:** the contribution is **engineering and reproducibility**
(an auditable, configuration-driven, tested hybrid physics+ML pipeline), **not** a
scientific discovery, a novel architecture, nor a new capability to "identify the energy
source feeding hurricanes." The title term "Atmospheric Singularity Mapping" overstates
the scope and **has been removed** from the project's documentation and citation
metadata; the revised manuscript carries an honest title.

---

## Summary

The preprint's *intent* was honest (it explicitly disclaimed discovery, novel
architecture, and validated localization). The errors were: (1) the released code did
not implement the physics-guided method it described; (2) the threshold method differed;
(3) the headline metrics were not reproducible from the released artifacts; and (4) the
framing/title overstated the scope relative to existing literature and operational tools.

**Status:** all four items are now addressed. (1) and (2) are corrected in the current
code. (3) is resolved via option (a) — the full 1980–2023 dataset is released and the
superseded headline numbers are replaced by reproducible ones (ROC-AUC 0.796, PR-AUC
0.251, both CIs above chance). (4) is addressed by revising the title and framing: the
validated contribution is the auditable pipeline and the RI classification skill;
spatial energy-source attribution is documented as a validated-negative hypothesis
(item 4 above, three independent angles).
