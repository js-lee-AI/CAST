"""Offline replay of shared-budget allocation over logged rounds (CPU only).

Random batches of B logged rounds share M = B * w_base verified nodes. We
compare total committed tokens under a uniform split, online global
best-first by path score and hindsight water-filling on the realized gains.

Example:
  python scripts/alloc_replay.py --rounds results/rounds --out results/alloc.json
"""
import argparse
import heapq
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cast_trees.replay import acceptance_curve, load_rounds  # noqa: E402
from cast_trees.utils import dump  # noqa: E402


def water_fill(curves, M, w_max):
    """Hindsight allocation: each unit goes to the largest realized gain."""
    B = len(curves)
    w, left, heap = [1] * B, M - B, []
    for i, (_, acc) in enumerate(curves):
        heapq.heappush(heap, (-(acc[2] - acc[1]), i, 1))
    while left > 0 and heap:
        _, i, cw = heapq.heappop(heap)
        w[i] = cw + 1
        left -= 1
        if cw + 1 < w_max:
            acc = curves[i][1]
            heapq.heappush(heap, (-(acc[cw + 2] - acc[cw + 1]), i, cw + 1))
    return sum(curves[i][1][min(w[i], w_max)] for i in range(B))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", required=True)
    ap.add_argument("--domains", default="gsm8k,mtbench,humaneval")
    ap.add_argument("--w-max", type=int, default=48)
    ap.add_argument("--sample", type=int, default=900, help="rounds drawn from each domain")
    ap.add_argument("--n-batches", type=int, default=400)
    ap.add_argument("--batches", default="8,16,32")
    ap.add_argument("--w-base", default="4,8,16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    W = args.w_max
    out = {}
    for dom in args.domains.split(","):
        rounds = load_rounds(os.path.join(args.rounds, f"{dom}.pt"))
        pick = rng.choice(len(rounds), size=min(args.sample, len(rounds)), replace=False)
        curves = []
        for i in pick:
            prio, acc = acceptance_curve(rounds[i], W)
            curves.append((prio, acc + [acc[-1]] * (W + 1 - len(acc))))
        rows = {}
        for B in [int(x) for x in args.batches.split(",")]:
            for wb in [int(x) for x in args.w_base.split(",")]:
                M = B * wb
                tot = np.zeros(3)
                for _ in range(args.n_batches):
                    batch = [curves[j] for j in rng.choice(len(curves), size=B, replace=False)]
                    uni = sum(c[1][min(max(1, M // B), W)] for c in batch)
                    pool = sorted(((p, i) for i, c in enumerate(batch) for p in c[0]), reverse=True)
                    w = [0] * B
                    for _, i in pool[:M]:
                        w[i] += 1
                    glob = sum(batch[i][1][min(w[i], W)] for i in range(B))
                    tot += (uni, glob, water_fill(batch, M, W))
                u, g, o = tot / args.n_batches
                rows[f"B{B}_W{wb}"] = {"uniform": u, "best_first": g, "water_fill": o,
                                       "best_first_gain_pct": 100 * (g - u) / u,
                                       "water_fill_gain_pct": 100 * (o - u) / u}
                print(f"{dom} B={B:2d} w_base={wb:2d}: best-first {100 * (g - u) / u:+.1f}%  "
                      f"water-fill {100 * (o - u) / u:+.1f}% over uniform")
        out[dom] = rows
    dump(out, args.out)


if __name__ == "__main__":
    main()
