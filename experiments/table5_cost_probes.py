"""Tables 5 and 6: target-forward latency by packed-token count.

Every probe is batch one in bf16 over a random KV prefix (8 warmups, mean of 30
timed iterations). Table 6 prints the endpoint slope from n=16 to 128.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load  # noqa: E402


def short(target):
    return target.split("/")[1].replace("-Instruct", "")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    probes = load("cost_probes.json")["probes"]

    cols = [1, 16, 24, 32, 48, 64, 96, 128]
    print("Table 5. Target-forward latency (ms) by packed-token count n = N + 1, 1k-token KV prefix")
    print(f"{'GPU, target':30s}" + "".join(f"{'n=' + str(c) if c == 1 else c:>7}" for c in cols))
    for p in probes:
        if p["table"] == 5:
            m = dict(zip(p["n"], p["ms"]))
            print(f"{p['gpu'] + ', ' + short(p['target']):30s}" + "".join(f"{m[c]:7.1f}" for c in cols))

    cols = [1, 16, 32, 64, 128]
    print("\nTable 6. H100 SXM target-forward latency (ms) across KV-prefix lengths, slope from n=16 to 128")
    print(f"{'target':14s}{'prefix':>7s}" + "".join(f"{'n=' + str(c) if c == 1 else c:>7}" for c in cols) + f"{'slope':>8s}")
    for p in probes:
        if p["table"] == 6:
            m = dict(zip(p["n"], p["ms"]))
            print(f"{short(p['target']):14s}{str(p['prefix'] // 1024) + 'k':>7s}"
                  + "".join(f"{m[c]:7.1f}" for c in cols) + f"{(m[128] - m[16]) / 112:8.3f}")


if __name__ == "__main__":
    main()
