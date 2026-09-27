"""Table 3: mean gain over DFlash on Blackwell at N*, with Table 10 behind its last row.

Greedy rows: mean over prompts of t_DFlash / t_CAST - 1, with a paired bootstrap
95% interval (10,000 resamples of prompts, seed 12345). Per-prompt times average
three repeats. Sampling row: T=1 with the stochastic verifier, gain of the
three-seed mean times, with the range of the per-seed gains.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DOMAIN_NAMES, SWEEP_DOMAINS, load  # noqa: E402

from cast_trees.stats import paired_bootstrap  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    pl, sm = load("prompt_level.json"), load("sampling.json")

    print("Table 3. Mean gain over DFlash on Blackwell at N* (63 for Qwen3-8B, 95 for Qwen3-4B)")
    print(f"{'':28s}" + "".join(f"{DOMAIN_NAMES[d]:>22s}" for d in SWEEP_DOMAINS))
    for key, label in [("blackwell_qwen3-8b", "Qwen3-8B, greedy decoding"),
                       ("blackwell_qwen3-4b", "Qwen3-4B, greedy decoding")]:
        cells, ns = [], []
        for d in SWEEP_DOMAINS:
            rec = pl[key]["domains"][d]
            g, lo, hi = paired_bootstrap(rec["dflash_ms"], rec["cast_ms"], args.n_boot)
            cells.append(f"{100 * g:+.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]")
            ns.append(len(rec["dflash_ms"]))
        print(f"{label:28s}" + "".join(f"{c:>22s}" for c in cells)
              + f"   prompts {'/'.join(map(str, ns))}")

    s = sm["blackwell_qwen3-8b"]
    cells, rows = [], []
    for d in SWEEP_DOMAINS:
        rec = s["domains"][d]
        seed_gains = [a / b - 1 for a, b in zip(rec["dflash_ms"], rec["cast_ms"])]
        df, ca = statistics.mean(rec["dflash_ms"]), statistics.mean(rec["cast_ms"])
        g = df / ca - 1
        cells.append(f"{100 * g:+.1f}% [{100 * min(seed_gains):.1f}, {100 * max(seed_gains):.1f}]")
        rows.append((d, rec["dflash_R"], rec["cast_R"], df, ca, g, min(seed_gains), max(seed_gains)))
    print(f"{'Qwen3-8B, sampling at T=1':28s}" + "".join(f"{c:>22s}" for c in cells))

    print("\nTable 10. Stochastic verification for Qwen3-8B on Blackwell at T=1 and N*=63")
    print(f"{'domain':10s} {'DFlash R':>8s} {'CAST R':>7s} {'DFlash ms':>9s} {'CAST ms':>8s} "
          f"{'mean gain':>9s}  seed range")
    for d, dr, cr, df, ca, g, lo, hi in rows:
        print(f"{DOMAIN_NAMES[d]:10s} {dr:8.2f} {cr:7.2f} {df:9.2f} {ca:8.2f} {100 * g:+8.1f}%  "
              f"{100 * lo:+.1f} to {100 * hi:+.1f}%")


if __name__ == "__main__":
    main()
