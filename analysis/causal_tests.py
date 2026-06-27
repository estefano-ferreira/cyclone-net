from __future__ import annotations

"""CycloneNet — counterfactual (ablation) causal test (CLI wrapper).

Thin command-line front-end over src.evaluation.causal_ablation. It loads the
project config and the released checkpoint, then runs the FuelMap ablation vs a
low-fuel control region and reports whether the model's RI prediction causally
depends on the identified energy source.

Example:
  python analysis/causal_tests.py --split test --k 0.05 --factor 0.5 \
      --channels sst_anom_K wind_mps
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.causal_ablation import run_and_save
from src.utils.config import cfg_get, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="FuelMap counterfactual causal test")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--k", type=float, default=0.05, help="Top/bottom-k fraction for masks")
    ap.add_argument("--factor", type=float, default=0.5, help="Ablation strength in [0,1]")
    ap.add_argument("--channels", nargs="+", default=["sst_anom_K", "wind_mps"],
                    help="Input channel names to ablate inside the masks")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    out_path = results_dir / "causal" / f"causal_ablation_{args.split}.json"

    report = run_and_save(
        cfg, split=args.split, k=args.k, factor=args.factor,
        channels=args.channels, out_path=out_path, max_events=args.max_events,
    )
    print(json.dumps(report.get("causal_evidence", report), indent=2))


if __name__ == "__main__":
    main()
