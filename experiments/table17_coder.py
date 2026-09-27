"""Table 17: Qwen3-Coder-30B-A3B with the block-16 DFlash head on H100 SXM."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load  # noqa: E402
from sweep_table import print_sweep  # noqa: E402


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    s = load("decoding.json")["h100sxm_qwen3-coder-30b-a3b"]
    print("Table 17. Qwen3-Coder-30B-A3B greedy decoding on H100 SXM, speedup over AR and R")
    print_sweep(s, ["gsm8k", "humaneval", "mbpp"], "DFlash (16)", ar_digits=1)  # the paper prints one decimal here


if __name__ == "__main__":
    main()
