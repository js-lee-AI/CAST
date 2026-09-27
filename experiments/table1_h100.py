"""Table 1: Qwen3 decoding on H100 SXM, and the full width sweep of Table 12.

Cells are speedup over AR (t_AR / t) and committed round length R. The average
column averages the domain ratios. The EAGLE-3 rows of the paper come from the
official EAGLE code and are not rebuilt here.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAIN_NAMES, H100_DOMAINS, load  # noqa: E402

BLOCKS = [("Temperature = 0", "h100sxm_qwen3-8b", "Q3-8B"), ("", "h100sxm_qwen3-4b", "Q3-4B"),
          ("Temperature = 1", "h100sxm_qwen3-8b_t1", "Q3-8B"), ("", "h100sxm_qwen3-4b_t1", "Q3-4B")]


def row(run, pick):
    cells, spd, rs = [], [], []
    for d in H100_DOMAINS:
        ms, r = pick(run[d])
        s = run[d]["ar_ms"] / ms
        cells.append(f"{s:.2f}x {r:.2f}")
        spd.append(s)
        rs.append(r)
    cells.append(f"{statistics.mean(spd):.2f}x {statistics.mean(rs):.2f}")
    return " | ".join(f"{c:>11s}" for c in cells)


def table1(dec):
    head = " | ".join(f"{name:>11s}" for name in [DOMAIN_NAMES[d] for d in H100_DOMAINS] + ["Avg."])
    print("Table 1. Speedup over AR and committed round length R on H100 SXM")
    print(f"{'':25s} | {head}")
    for title, key, model in BLOCKS:
        if title:
            print(title)
        s = dec[key]
        run = s["runs"][0]
        n = str(s["n_star"])
        print(f"  {model} DFlash (16)".ljust(25) + " | " + row(run, lambda r: (r["dflash"]["ms"], r["dflash"]["R"])))
        print(f"  {model} CAST (N*={n})".ljust(25) + " | " + row(run, lambda r: (r["cast"][n]["ms"], r["cast"][n]["R"])))


def table12(dec):
    print("\nTable 12. Greedy width sweeps on H100 SXM, mean decode-only ms/token")
    print(f"{'domain':10s} {'max new':>7s} {'AR':>6s} {'chain':>6s} {'N=47':>6s} {'N=63':>6s} "
          f"{'N=95':>6s} {'N=127':>6s} {'R chain':>7s} {'R 127':>6s}")
    for key, model in [("h100sxm_qwen3-8b", "Qwen3-8B"), ("h100sxm_qwen3-4b", "Qwen3-4B")]:
        print(model)
        run = dec[key]["runs"][0]
        for d in ["gsm8k", "mtbench", "humaneval", "math500", "mbpp"]:
            r = run[d]
            c = r["cast"]
            print(f"{DOMAIN_NAMES[d]:10s} {r['max_new']:7d} {r['ar_ms']:6.2f} {r['dflash']['ms']:6.2f} "
                  + " ".join(f"{c[n]['ms']:6.2f}" for n in ("47", "63", "95", "127"))
                  + f" {r['dflash']['R']:7.2f} {c['127']['R']:6.2f}")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    dec = load("decoding.json")
    table1(dec)
    table12(dec)


if __name__ == "__main__":
    main()
