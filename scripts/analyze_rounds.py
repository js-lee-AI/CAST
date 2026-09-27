"""Replay diagnostics from collect_rounds.py logs (CPU only).

Prints and saves, by domain: the sorted prefix masses rho_(N), the rank of the
target correction at the chain's first rejection, depth-one reliability, the
fixed-shape ablation at a matched budget and the rank-cap gap.

Example:
  python scripts/analyze_rounds.py --rounds results/rounds --out results/replay.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cast_trees import replay  # noqa: E402
from cast_trees.utils import dump  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True, help="directory with <domain>.pt round logs")
    ap.add_argument("--domains", default="gsm8k,mtbench,humaneval")
    ap.add_argument("--n-max", type=int, default=128)
    ap.add_argument("--shape-budget", type=int, default=47)
    ap.add_argument("--rank-cap-budgets", default="32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}
    for dom in args.domains.split(","):
        rounds = replay.load_rounds(os.path.join(args.rounds, f"{dom}.pt"))
        rho = replay.mass_curve(rounds, args.n_max)
        rec = {
            "rounds": len(rounds),
            "rho": rho,
            "first_rejection_rank": replay.rejection_ranks(rounds),
            "depth1_reliability": replay.reliability(rounds, depth=1),
            "shape": replay.shape_ablation(rounds, args.shape_budget),
            "rank_cap": {int(b): replay.rank_cap(rounds, int(b))
                         for b in args.rank_cap_budgets.split(",")},
        }
        out[dom] = rec
        s, fr = rec["shape"], rec["first_rejection_rank"]
        print(f"[{dom}] {len(rounds)} rounds")
        print(f"  rho_(N): N=1 {rho[0]:.3f}  N=48 {rho[47]:.4f}  N=64 {rho[63]:.4f}")
        print(f"  first rejection: rank 2-4 {fr['rank2_4']:.1%}, rank 5-8 {fr['rank5_8']:.1%}")
        print(f"  shape at N={args.shape_budget}: " + "  ".join(f"{k} {v:.2f}" for k, v in s.items()))
        for b, rc in rec["rank_cap"].items():
            print(f"  rank cap at N={b}: R {rc['R_cap']:.3f} vs full {rc['R_full']:.3f} ({rc['gap_pct']:.2f}%)")
    dump(out, args.out)


if __name__ == "__main__":
    main()
