# MANUSCRIPT V3 — PREPRINT skeleton (definition, pre-drafting)

**Status:** REVISED FOR AUTHOR REVIEW — 2026-07-16 (genre change). Supersedes
the Data-Descriptor skeleton of earlier today. Drafting redirection starts
only after the author approves this revision.

**GENRE (author decision, 2026-07-16):** `MANUSCRIPT_V3.md` is a **PREPRINT
for the Zenodo v3 record**, not a Data Descriptor. Factual reason: the
record lineage (18571958 v1.0.0 → 18577056 v1.0.1 → 18751255 v2.0.0) is
uniformly "Preprint" with a full paper PDF; a 3-page note as v3 would break
the line's own format — it would read as a notice, not a version.
Consequences:
- The supersession note does NOT go up alone; its content becomes §9
  (Correction record) of this paper. Reuse
  `docs/release/zenodo_v3_supersession_note.tex` §3–§4 content — reuse, do
  not rewrite.
- The v2.0.0 PDF is NOT attached to v3 — the version-DOI preserves the old
  record permanently; that is what versioning does.
- A **Data Descriptor remains a SEPARATE, REDUCED version of this paper**,
  to be derived later for journal submission (ESSD vs Scientific Data —
  decision still open, APC/institution-dependent). Not the same document.
  The Data-Descriptor draft assembled earlier today (current content of
  `MANUSCRIPT_V3.md`) is the raw material for both: its sections are reused
  in the preprint per the mapping below, and its original form seeds the
  reduced journal version later.

**Title (FINALIZED, do not touch):** "CycloneNet: A Reproducible Pipeline
and Leakage-Safe Two-Basin Dataset for Tropical-Cyclone
Rapid-Intensification Analysis"

**Order (author decision, maintained):** current state first, correction
record after (§9), conclusion last.

---

## Canonical numbers (single source of truth — NOTHING is recomputed)

Any number in V3 must match this table; drafting agents must not recompute.

| Quantity | Value | Source |
|---|---|---|
| Valid events | 16,780 | package manifest / DATA_DICTIONARY §2 |
| Storms | 992 (578 EP / 414 NA by genesis) | idem |
| Labels (v2) | 799 RI positive / 15,962 negative / 19 NULL | idem |
| Event list rows | 32,989 (byte-reproducible from raw IBTrACS) | TECHNICAL_VALIDATION §1 |
| Per-point basin | 8,888 EP / 7,892 NA (valid set) | basin repair manifest |
| Coverage | 1980–2023, Jun–Nov seasons, bbox [60N,140W,0N,20W] | DATA_DICTIONARY §1 |
| Cube shape | 40×40×5×14 float32, 448 KB/event | DATA_DICTIONARY §3–4 |
| Channels | 14 stored: 9 model-used, 3 stored-only (heat fluxes, anti-leakage), 2 tested (H6 NULL) | DATA_DICTIONARY §4 |
| Splits | 70/15/15 by SID, SHA256-deterministic + frozen override; test = 2,679 events / 112 v2-positives / 6 NULL | DATA_DICTIONARY §7 |
| Dev (PL-gated) | 14,101 events / 687 positives / 839 storms | PROJECT_STATE §7 |
| v1→v2 label correction | 148/32,989 rows (0.45%) misaligned; ZERO valid-set flips; 19 events → NULL; positives 802→799 | ERRATA item 6 |
| Historical CNN (retired) | test ROC-AUC 0.796 [0.753–0.837], PR-AUC 0.251 [0.179–0.331]; read under v1 labels (115 positives; 112 under v2 recount) | BENCHMARK.md |
| **Campaign ladder (dev pooled OOF PR-AUC, mean of seeds 42/123/456)** | GBM_S 0.202 / GBM_F 0.170 / CNN 0.171 / GBM_SF 0.249 | compare_20260716T121803Z.json / BENCHMARK.md |
| **H6 verdict (NULL)** | cross-seed ΔPR-AUC +0.0185, 95% CI [−0.0070, +0.0431] ∋ 0 | aggregate_20260716T120517Z.json |
| **H9 V1 verdict (NEGATIVE)** | Δ₁(CNN−GBM_SF) = −0.0781, CI [−0.1162, −0.0422] < 0 | compare_20260716T121803Z.json |
| **H9 V2 verdict (NULL)** | Δ₂(CNN−GBM_F) = +0.0005, CI [−0.0285, +0.0316] ∋ 0 | idem |
| Base rate | 4.8% (valid set) | DATA_DICTIONARY §9 |
| Licenses | code MIT, dataset CC BY 4.0; v1/v2 records CC BY-NC 4.0 (historical) | LICENSE / LICENSE-DATA / Zenodo records |
| DOI slots | dataset concept DOI + software record v3.0.0 — ⟦pending Zenodo mint⟧ | zenodo_v3_metadata.md |

**Superseded-claim inventory (quoted ONLY inside §9, as claims under
correction — verified against Zenodo + git history 2026-07-16):**

| Claim (verbatim era) | Where it lived | Repo forensics |
|---|---|---|
| ROC-AUC 0.97, Recall 0.92 | v1.0.0/v1.0.1 descriptions + v1-era README.md/BENCHMARK.md | text claim only; NO metrics artifact in any versioned tree ever recorded it; removed from repo at "v2 start" (`e41c0f5`) |
| "sub-pixel spatial accuracy" | idem | idem |
| mean spatial error "~26 km" | v1.0.0 description | NEVER in git history at any commit; origin not determinable from the repository |
| "18 hurricanes (1989–2024)" | v1.0.0/v1.0.1 descriptions | NEVER in git history; origin not determinable from the repository |
| "Target Lock" (branding) | v1-era README/BENCHMARK | removed at v2 start; survives lowercase as technical descriptor of `pred_lat/pred_lon` in validation docs |
| "Atmospheric Singularity Mapping" | v1 titles + v2.0.0 title | removed from repo identity (ERRATA item 5) |
| ROC-AUC 0.83 (2,193 test samples) | v2.0.0 | already covered by ERRATA item 3 (RESOLVED: superseded by 0.796/0.251) |

Lineage verified: `/records/18571958/latest` resolves to 18751255 — the
three records are ONE concept line; v3 supersedes the whole line.
**Pending author decision:** new ERRATA item (would be item 9) for the
v1.0.0/v1.0.1 claims, which item 3 does not cover.

## Global rules (inherited without exception — every drafting task carries these)

1. **CANONICAL TABLE** is the single source of truth. No agent recomputes.
2. **FABRICATION RULE (reinforced — the lesson of the first drafting
   round):** any number, technical name, or parameter NOT in the canonical
   table is a FABRICATION until verified against the repository. Do not
   fill a gap with a plausible value — leave `⟦VERIFY: …⟧` and report it.
   The first round invented "±3 h tolerance", "40 storms in the override
   map", "136 of 155 reassigned", `PRES_MEAN_FLUX`/`PRES_DIV_MEAN` — all
   plausible, all false, all in gaps the table did not cover.
   Exception: the CNN's frozen test-set metrics (0.796/0.251) are a read
   ALREADY MADE — citable per the table. The rule forbids NEW reads.
3. **"physics-guided" appears NOWHERE**, except in Code/Data Availability
   as "legacy naming kept for reproducibility" (class/module names only).
4. **FuelMap is not a results topic.** It appears in the project history
   (§1/§2) and in the correction record (§9). No figures, no
   re-argumentation.
5. **No model is the product.** CNN = historical record + evidence the
   labels support learning. GBM_SF = dev-fold evidence, NEVER test.
6. **Claim discipline:** lower bound is not a ceiling; claims qualified by
   architecture (GAP); NULL is "undefined", never coerced to negative;
   two-basin, never "North Atlantic sector".
7. **Corrections stated in full, no softening.**
8. **TONE: no narrative arc, no redemption, no "the rigor led to something
   better", no writing aimed at making the reader conclude the project is
   good. Describe. The facts are verifiable and speak for themselves.**

---

## Section skeleton (preprint)

Reuse mapping refers to the assembled Data-Descriptor draft (current
`MANUSCRIPT_V3.md`, reviewed + corrected today) — reuse means restructure
that text, not re-draft from sources.

### Abstract
New. Product-first (dataset + pipeline), the campaign verdicts in one
sentence each, the supersession of the record line stated plainly. Reuse
material from the current Abstract + zenodo_v3_metadata description.

### 1. Introduction
Reuse current §1 (Background & Summary) paragraphs 1–2 (forensic framing,
leakage-safe definition with forward-reference to §5). Add: what the paper
delivers (dataset, pipeline, campaign results, correction record).

### 2. Positioning
Sources: `docs/literature_review.md`, current §1 paragraph 4, the
SHIPS-RII boundary sentence, and
`.claude/research/2026-07-15_posicionamento_RI/VEREDITO_FINAL.md` — the
four dead novelty claims are the guardrail: NO pioneering claim survives.
What is established vs what this work adds (engineering + reproducibility
only).

### 3. Data
Reuse current §3 (Data Records) nearly whole — the dataset is the product.
Population, files, channels, sidecar schema, basin semantics + reader
warning, label semantics v2.

### 4. Methods
Reuse current §2 (Methods): sources with verbatim NOTICE citations, event
definition, labeling v2, QC, normalization, splits, windowed provenance.

### 5. Technical validation
Reuse current §4 whole (7 evidences + retraction discipline). The
byte-exact replication is the strongest evidence — it leads. §5.3 carries
the proof of the title's "Leakage-Safe" claim (connect to §1 definition).

### 6. The pre-registered campaign
NEW — real results section (the preprint has no Data-Descriptor scope
restriction). Content: the pre-registration discipline (registered before
running, CIs read once); the ladder (GBM_S 0.202 / GBM_F 0.170 / CNN 0.171
/ GBM_SF 0.249, dev pooled OOF); the four verdicts with CIs from the
canonical table (H6 NULL; H9 V1 negative; H9 V2 NULL; H8 cancelled as
consequence — ablating a component of a retired model decides nothing);
the ex-ante qualifications (GAP architecture, intensity-blind, basin
one-hot, 15-epoch budget); the scope guard verbatim from the
supersession note / registry. Sources: `docs/hypothesis_registry.md`,
`BENCHMARK.md`, the three pre-registration docs, zenodo_v3_supersession
note §2.

### 7. Discussion
NEW. What the verdicts mean and do not mean: the architecture question is
closed for THIS architecture only; tabular-vs-CNN at fixed information
diet licenses no claim about spatial signal in the data (GAP qualifier);
what a redesign would require (new pre-registration); why the project
designates no reference model; what the dataset supports next. No
forward-looking promises beyond registered agenda pointers.

### 8. Limitations
Reuse current §5 Known Limitations (7 items + two-basin heterogeneity) and
MANUSCRIPT_honest §7 items still applicable (surface-only fields, 0.25°
inner-core, no SHIPS-RII comparison). One list, no duplication.

### 9. Correction record
Reuse the supersession-note content (§3 of the .tex, no-softening rule) +
EXTEND to the whole record line per the superseded-claim inventory above:
- v2.0.0 claims: 0.83 non-reproducible → replaced by 0.796/0.251;
  FuelMap localization refuted (3 angles); physics-guided retired;
  Singularity Mapping removed. (ERRATA items 1–5.)
- v1.0.0/v1.0.1 claims (NEW coverage): ROC-AUC 0.97 / Recall 0.92,
  "sub-pixel spatial accuracy", "~26 km mean spatial error",
  "18 hurricanes (1989–2024)", "Target Lock". For 0.97/0.92/sub-pixel:
  text claims in v1-era README/BENCHMARK, no supporting artifact ever
  versioned. For 26 km / 18 hurricanes: **"not reproducible from any
  released artifact; origin not determinable from the repository or its
  history"** — exact wording, no invented explanation.
- dv24 v1→v2 label correction + basin relabel + Defect-0 retraction
  (ERRATA items 6–8).
Acceptance test (author, verbatim): a reader who cited ANY previous
version (v1.0.0, v1.0.1, or v2.0.0) and opens this paper discovers,
without ambiguity, that (a) they must not cite 0.97, 0.83, or the 26 km;
(b) FuelMap was refuted, not pending; (c) the CNN was retired;
(d) "sub-pixel accuracy" and "Atmospheric Singularity Mapping" were
withdrawn.

### 10. Conclusion
New, short. What is released, under what licenses, what was corrected,
what is registered as open agenda. Tone rule 8 applies doubly here.

### Data/Code availability + Acknowledgements
Reuse current §6 + §7 (two Zenodo records with DOI slots, platform URL,
replication-gate script, verbatim NOTICE attributions, author block slot,
coauthor slot, no competing interests).

---

## Drafting plan (after author approves this skeleton)

Delegation as before (one agent per new/restructured section, canonical
table + 8 global rules + named sources in every spec; Fable reviews each
against sources; assembly last). Reused sections need editing passes, not
re-drafting. Fixed cross-references are assigned at assembly.

Incidental repo finding to queue (outside this manuscript):
`docs/DATASET.md` still says ERA5/IBTrACS "1989–2024" (stale v1-era range;
current coverage is 1980–2023) — belongs to the relabel/hygiene pass.
