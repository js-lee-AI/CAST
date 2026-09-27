"""Figure 4(c): sorted candidate prefix masses from offline greedy replay (Qwen3-8B).

rho_(N) is the N-th largest best-first prefix mass, averaged over the first 1,500
logged rounds of each domain. The rule's threshold for GSM8K on Blackwell is
about 0.015 (Section 3.3). Prints the curve at a few ranks and saves
results/figure4c.png when matplotlib is installed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAIN_NAMES, ROOT, SWEEP_DOMAINS, load  # noqa: E402

RANKS = [1, 4, 8, 16, 48, 64, 96, 128]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figure4c.png"))
    args = ap.parse_args()
    masses = load("masses.json")["domains"]
    print(f"{'rank r':>10s}" + "".join(f"{r:>8d}" for r in RANKS))
    for d in SWEEP_DOMAINS:
        print(f"{DOMAIN_NAMES[d]:>10s}" + "".join(f"{masses[d][r - 1]:8.4f}" for r in RANKS))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skipping the plot")
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    for d in SWEEP_DOMAINS:
        ax.plot(range(1, len(masses[d]) + 1), masses[d], label=DOMAIN_NAMES[d])
    ax.axhline(0.015, color="black", lw=0.8, ls="--", label="Blackwell GSM8K threshold")
    ax.set_yscale("log")
    ax.set_xlabel("Candidate index r")
    ax.set_ylabel("Prefix mass")
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
