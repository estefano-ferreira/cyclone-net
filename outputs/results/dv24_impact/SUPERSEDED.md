# SUPERSEDED — reports v1 through v4 are retracted (2026-07-16)

`report_20260716_144350.*`, `report_v2_20260716_145238.*`,
`report_v3_20260716_145921.*`, `report_v4_20260716_151027.*` computed their
"correct" reference by re-running grouped shifts on
`data/event_list_augmented.csv` — the builder's OUTPUT, from which
`build_event_list` (src/processors/ibtracs.py) had already dropped every row
without both targets (`dropna(subset=["dv12_kt","dv24_kt"])`). On that file it
is impossible to distinguish "temporal partner never existed" from "partner
row was dropped after being used", which fabricated:

- **"DEFECT 0 / cross-SID bleed / 84 phantom positives" — FALSE.** The
  trailing-row values were computed from real same-storm rows later removed
  by the dropna. Verified directly against raw IBTrACS (shipped dv24 equals
  wind at exactly t0+24h, same SID).
- **"EFFECT 2 / 3,062 undefined valid events" — FALSE** at the dataset level.
  The `NaN >= 30 → 0` coercion in `ri_labeling.py` is dead code w.r.t. the
  shipped list because the builder drops undefined rows.

The authoritative assessment is `report_v5_*` produced by
`analysis/dv24_impact_assessment_v5_raw_reference.py`, which rebuilds the
pre-dropna series from `data/raw/ibtracs.ALL.list.v04r00.csv`, proves exact
replication of the shipped file (32,989/32,989 rows, dv24 and ri_label equal
everywhere), and only then measures positional-vs-strict-temporal differences.

Kept on disk as an audit trail of the review process; do not cite v1–v4
numbers.
