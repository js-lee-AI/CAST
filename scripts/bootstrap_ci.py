"""Paired bootstrap interval of the gain over DFlash from a bench.py output.

gain = mean over prompts of (DFlash ms/token / CAST ms/token) - 1, with prompts
resampled with replacement (same indices for both methods).

Example (200 prompts, three repeats, predicted width 63):
  python scripts/bench.py --n-prompts 200 --seeds 0,1,2 --budgets 63 --skip-ar --out results/ci_8b.json
  python scripts/bootstrap_ci.py results/ci_8b.json --budget 63
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cast_trees.stats import paired_bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    res = json.load(open(args.result))
    for dom, rec in res["domains"].items():
        cast = rec["cast"][str(args.budget)]
        g, lo, hi = paired_bootstrap(rec["chain_ms_per_prompt"], cast["ms_per_prompt"], args.n_boot)
        print(f"{dom:10s} n={rec['n_prompts']:3d}  gain {100 * g:+.1f}%  "
              f"95% CI [{100 * lo:+.1f}, {100 * hi:+.1f}]")


if __name__ == "__main__":
    main()
