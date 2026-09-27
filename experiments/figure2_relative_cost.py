"""Figure 2: batch-one target-forward latency relative to 16 packed tokens.

Prints the plotted values and saves results/figure2.png when matplotlib is installed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load  # noqa: E402

LINES = [("Blackwell", "Qwen/Qwen3-8B", "Blackwell 8B"), ("Blackwell", "Qwen/Qwen3-4B", "Blackwell 4B"),
         ("A6000", "Qwen/Qwen3-8B", "A6000 8B"), ("H100 SXM", "Qwen/Qwen3-8B", "H100 8B"),
         ("H100 SXM", "Qwen/Qwen3-4B", "H100 4B")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figure2.png"))
    args = ap.parse_args()
    probes = [p for p in load("cost_probes.json")["probes"] if p["table"] == 5]
    curves = {}
    for gpu, target, label in LINES:
        p = next(q for q in probes if q["gpu"] == gpu and q["target"] == target)
        m = dict(zip(p["n"], p["ms"]))
        ns = [n for n in p["n"] if n >= 16]
        curves[label] = (ns, [m[n] / m[16] for n in ns])
    ns = curves["Blackwell 8B"][0]
    print(f"{'n':>14s}" + "".join(f"{n:>7d}" for n in ns))
    for label, (_, rel) in curves.items():
        print(f"{label:>14s}" + "".join(f"{r:7.3f}" for r in rel))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping the plot")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    for label, (x, y) in curves.items():
        ax.plot(x, y, marker="o", ms=3, label=label)
    ax.set_xlabel("Packed tokens n = N + 1")
    ax.set_ylabel("Relative verify latency")
    ax.set_xticks([16, 32, 64, 96, 128])
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
