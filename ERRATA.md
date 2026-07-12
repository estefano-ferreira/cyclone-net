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

## 3. The reported metrics are NOT reproducible from the public repository

**What the paper reports:** a held-out test set of **2,193 samples (211 RI positives)**,
with **ROC-AUC = 0.83, recall = 0.905**.

**What the public repository contains:** a dataset covering **2020–2023 only**
(972 valid events; test split = **155 samples, 9 RI positives**). The headline metrics
**cannot be reproduced** from the released code and data. On the public dataset, with the
corrected (physics-active) pipeline, test ROC-AUC is on the order of **~0.6**, on a test
set with too few positives (9) to support a stable estimate.

**Cause:** the headline numbers derive from a larger dataset that was not included in the
public release.

**Required action (choose one):**
- (a) release the full dataset used for the reported metrics, so 0.83 is reproducible; or
- (b) replace the headline metrics with the modest numbers reproducible from the public
  data, clearly stating the reduced sample size; or
- (c) clearly label the original metrics as derived from a non-released dataset.

Until one of these is done, the reported performance numbers should be treated as
**not independently reproducible**.

## 4. FuelMap localization does not beat a trivial baseline (new evidence)

The paper correctly **does not claim** externally validated localization. A validation
added during the audit (predicted FuelMap peak vs. TCHP peak) confirms and strengthens
this caveat with a concrete result: the FuelMap peak does **not** locate the TCHP peak
better than a naive "storm-centre" baseline. The localization claim should therefore be
stated as **unsupported**, not merely "pending."

A counterfactual ablation (occluding the FuelMap region vs. an equal-size control) does
show the model's RI prediction depends on the identified region (paired test, p ≈ 1e-6) —
but this demonstrates *model-internal* dependence, **not** that the region is the true
physical energy source.

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
the scope and should be revised.

---

## Summary

The preprint's *intent* was honest (it explicitly disclaimed discovery, novel
architecture, and validated localization). The errors are: (1) the released code did not
implement the physics-guided method it described; (2) the threshold method differed;
(3) the headline metrics are not reproducible from the released artifacts; and (4) the
framing/title overstate the scope relative to existing literature and operational tools.

Items (1) and (2) are corrected in the current code. Items (3) and (4) require the author
to decide between releasing the full dataset or revising the reported numbers and framing.
