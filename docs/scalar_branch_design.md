# Design: scalar state branch (V4 candidate, future H10)

Status: **DESIGN ONLY — implementation is POST-V3 work.**
Blocked until (a) H6 and H8 complete (`src/`, `config.yaml` and
`analysis/feature_ablation_cnn.py` are frozen; any trainer/model change
before H8 closes invalidates the reused-arm comparison), AND (b) V3 is
published. The V3 measurement characterizes the intensity-blind CNN;
adding state changes the artifact under measurement and belongs to a
subsequent cycle. Written 2026-07-14, before any H10 run, so the design
itself is pre-registrable.

## 1. Motivation and scientific dependency

The production CNN sees only spatial fields; it never receives the storm's
own state (Vmax, persistence, latitude, season). The tabular baseline (H9)
tests whether those scalars alone — or scalars + field aggregates — match
the CNN. The interpretation of this design depends on H9's verdict:

- If GBM(S) ≥ CNN: the scalar branch is a **necessity** — the CNN is not
  even capturing what a 5-feature table captures.
- If CNN > GBM(SF): the scalar branch is an **enhancement** — testing
  whether state + spatial signal combine additively.

Either way the experiment is the same; only the framing of the
pre-registration changes. Do not write the H10 pre-registration until H9's
verdict is recorded in `docs/hypothesis_registry.md`.

## 2. Scalar feature set

Reuse the H9 "state_only" (S) set exactly as built by
`analysis/tabular_baseline_kfold.py::build_feature_table` — same
derivation, same guards:

| Feature | Source | Notes |
|---|---|---|
| `wind_kt` | event JSON | Vmax at t0 |
| `pressure_mb` | event JSON | |
| `center_lat`, `abs_lat` | event JSON | |
| `basin_*` one-hot | event JSON | category list FIXED globally (all basins in dev set), not per-fold — keeps input width stable |
| `doy_sin`, `doy_cos` | timestamp | |
| `dv_past_12h`, `dv_past_24h` | derived, same-SID past events (10–14 h / 22–26 h windows) | **past only**; `dv12_kt`/`dv24_kt` from metadata are TARGETS, never features |
| `dv_past_12h_missing`, `dv_past_24h_missing` | derived | 0/1 flags, NaN→0 fill |

No cube aggregates in the branch (the CNN already sees the fields; feeding
`cube_*` stats would blur the S vs F attribution).

Delivery mechanism: **precomputed feature table** (parquet keyed by
`event_id`, same builder/cache as H9, gitignored). The dataset reads the
table; it does NOT re-derive persistence at `__getitem__` time (the SID
join needs the whole event index — wrong layer for it).

## 3. Model change (`src/models/cyclone_net_physics_guided.py`)

Current path: `stem → pool → fc → emb (32-d) → head_ri/dv12/dv24`
(`hidden_channels: 32` from config; the class default 64 is unused in the
production path). FuelMap head consumes pre-pool `feat` and the forward
head consumes the energy score — **both untouched** by this design.

Change, additive and default-off:

```
__init__(..., n_scalar_features: int = 0, scalar_hidden: int = 16)
  if n_scalar_features > 0:
      self.scalar_proj = Sequential(Linear(n → 16), ReLU, Dropout(dropout))
      head_in = hidden_channels + 16
  else:
      head_in = hidden_channels          # exactly current behavior
  self.head_ri/dv12/dv24 = Linear(head_in, 1)

forward(x, scalars=None, prior_map_t0=None)
  emb = fc(pool(stem(x)))                 # (B, 32)
  if scalars is not None: emb = cat([emb, scalar_proj(scalars)], dim=-1)
  ... heads as today
```

Rationale for a 16-d projection instead of raw concat: ~13 raw scalars vs
32-d embedding — the projection gives the branch its own nonlinearity and
a controlled width, and one hyperparameter (`scalar_hidden`) instead of
per-feature scaling debates. All three scalar heads (ri/dv12/dv24) share
the concatenated embedding, mirroring today's shared-`emb` design.

Backward compatibility: with `n_scalar_features=0` the module graph is
IDENTICAL to today's (same parameter names/shapes) — old checkpoints load
unchanged; arm A of H10 is bit-for-bit today's architecture.

## 4. Dataset change (`src/data/dataset.py`)

- New optional ctor arg: `scalar_table_path` (parquet). When set,
  `__getitem__` adds `"scalars": (n,) float32` to the returned dict.
- Scalar normalization: z-score with **training-fold-only** stats, same
  discipline as the field norm stats (computed per fold in the ablation
  harness, stored alongside `normalization_stats`). One-hot/flag columns
  are not standardized.
- Missing event in table → hard error, not silent zeros (the table must
  cover the dev set 100%; the H9 census gate already guarantees this).

## 5. Trainer change (`src/training/trainer.py`)

Minimal: in `run_epoch`, `scalars = batch.get("scalars")`; pass through to
`model(batch["x"], scalars=scalars, prior_map_t0=prior)`. Losses,
lambdas, optimizer, physics terms: unchanged.

## 6. Config

```yaml
model:
  use_scalar_branch: false        # default off — parity with today
  scalar_hidden: 16
  scalar_features: [...]          # explicit list, mirrors H9 S-set
```

## 7. Ablation protocol (H10 — to pre-register after H9 verdict)

Same instrument as H6/H8: 3-fold SID-grouped StratifiedGroupKFold on the
PL-gated dev set (14,101/687), seeds 42/123/456, 15 epochs, phased one
seed per night, detached execution (PROJECT_STATE §6), pooled-OOF PR-AUC
with cluster bootstrap by SID, CI read once after 3 seeds.

- Arm A: winner configuration of H6 (channels per H6 verdict), no scalars.
- Arm B: same channels + scalar branch.
- `--reuse-arm-a`: if H10's arm A is architecturally identical to a
  completed H6/H8 arm **and the trainer/model code is unchanged since**,
  reuse those runs (H8 pattern). Any code change to the shared files
  between H8 and H10 forfeits reuse — arm A must be retrained.
- Decision branches, deltas of interest and abort criteria: fixed in the
  H10 pre-registration BEFORE the first run, referencing H9's recorded
  numbers.

## 8. Acceptance criteria (for the implementation spec, when unblocked)

1. **Parity test**: with `use_scalar_branch: false`, model state dict keys
   and all forward outputs are identical to the pre-change code (same
   seed → same weights → same logits on a fixed batch).
2. Unit test: scalar branch changes head input width and forward accepts
   `scalars`; missing `scalars` with branch enabled raises.
3. Feature table build is reproducible from `data/interim` metadata alone
   and never touches `dv12_kt`/`dv24_kt` as features (assert in builder).
4. Fold-scoped scalar stats: no statistic computed outside the training
   fold; no read of the frozen test split anywhere.
5. Paths via `rel_to_root`; secret_guard clean; no AI attribution.

## 9. Effort estimate

Model + dataset + trainer + config: ~0.5 day (delegable with this doc as
spec). H10 compute: ~11 h/seed × 3 nights, same as H6.
