"""Figure 3: Qwen3-8B width sweeps on H100 SXM, speedup over standard DFlash.

Speedup is t_DFlash / t_CAST with decode-only latency. Prints the plotted values
and saves results/figure3.png when matplotlib is installed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAIN_NAMES, H100_DOMAINS, ROOT, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figure3.png"))
    args = ap.parse_args()
    run = load("decoding.json")["h100sxm_qwen3-8b"]["runs"][0]
    widths = sorted(run["gsm8k"]["cast"], key=int)
    print(f"{'domain':>10s}" + "".join(f"{'N=' + n:>8s}" for n in widths))
    curves = {}
    for d in H100_DOMAINS:
        r = run[d]
        curves[d] = [r["dflash"]["ms"] / r["cast"][n]["ms"] for n in widths]
        print(f"{DOMAIN_NAMES[d]:>10s}" + "".join(f"{x:8.3f}" for x in curves[d]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping the plot")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    x = [int(n) for n in widths]
    for d, y in curves.items():
        ax.plot(x, y, marker="o", ms=3, label=DOMAIN_NAMES[d])
    ax.set_xlabel("Nonroot budget N")
    ax.set_ylabel("Speedup over DFlash")
    ax.set_xticks(x)
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
