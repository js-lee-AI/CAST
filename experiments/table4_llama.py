"""Table 4 and Table 16: LLaMA-3.1-8B with the block-10 DFlash head on H100 SXM.

Table 4 gives speedup over AR at N*=127. Table 16 adds every width and the
committed round length R. The EAGLE-3 rows of the paper come from the official
EAGLE code and are not rebuilt here.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAIN_NAMES, H100_DOMAINS, load  # noqa: E402
from sweep_table import print_sweep  # noqa: E402


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    s = load("decoding.json")["h100sxm_llama3.1-8b"]
    run, n = s["runs"][0], str(s["n_star"])
    print("Table 4. LLaMA-3.1-8B greedy decoding on H100 SXM, speedup over AR")
    print(f"{'method':16s}" + "".join(f"{DOMAIN_NAMES[d]:>11s}" for d in H100_DOMAINS) + f"{'Avg.':>11s}")
    for label, pick in [("DFlash (10)", lambda r: r["dflash"]["ms"]),
                        (f"CAST (N*={n})", lambda r: r["cast"][n]["ms"])]:
        spd = [run[d]["ar_ms"] / pick(run[d]) for d in H100_DOMAINS]
        print(f"{label:16s}" + "".join(f"{x:>10.2f}x" for x in spd) + f"{statistics.mean(spd):>10.2f}x")
    print("\nTable 16. Every width, speedup over AR and R")
    print_sweep(s, H100_DOMAINS, "DFlash (10)")


if __name__ == "__main__":
    main()
