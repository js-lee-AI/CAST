"""Tables 7, 8 and 9 (and Figure 5): Blackwell and A6000 width sweeps over three domains.

Each Blackwell cell comes from the run with the median gain among the repeats
(three runs, except Qwen3-8B at N=127, measured once, and Qwen3-4B at N=31,
whose gain averages two runs). A6000 rows are single runs, and their N=127
cells come from a second A6000 host. Table 9 and Figure 5 are the N=15 cells,
where CAST verifies the same 16 packed tokens as DFlash.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DOMAIN_NAMES, SWEEP_DOMAINS, gain, load, median_run  # noqa: E402

GRID = [15, 31, 47, 63, 95, 127]
SETTINGS = [("Blackwell", "Qwen3-8B", "blackwell_qwen3-8b", None),
            ("Blackwell", "Qwen3-4B", "blackwell_qwen3-4b", None),
            ("A6000", "Qwen3-8B", "a6000_qwen3-8b", "a6000_second_host_qwen3-8b"),
            ("A6000", "Qwen3-4B", "a6000_qwen3-4b", "a6000_second_host_qwen3-4b")]


def table7(dec):
    print("Table 7. Decode-only ms/token, gain over DFlash at N* and committed round length R")
    print(f"{'setting':22s}{'domain':>10s}{'AR':>7s}{'DFlash':>8s}{'CAST':>7s}{'gain':>8s}{'R':>6s}")
    for gpu, tgt, key, _ in SETTINGS:
        s = dec[key]
        n = s["n_star"]
        for d in SWEEP_DOMAINS:
            g, rec = median_run(s, d, n)
            c = rec["cast"][str(n)]
            print(f"{gpu + ' ' + tgt[-2:] + ', N*=' + str(n):22s}{DOMAIN_NAMES[d]:>10s}{rec['ar_ms']:7.1f}"
                  f"{rec['dflash']['ms']:8.2f}{c['ms']:7.2f}{100 * g:+7.1f}%{c['R']:6.2f}")


def table8(dec):
    print("\nTable 8. Greedy width-sweep gains over DFlash (A6000 N=127 from a second host)")
    print(f"{'GPU':10s}{'target':10s}{'domain':>10s}" + "".join(f"{'N=' + str(n):>9s}" for n in GRID))
    for gpu, tgt, key, extra in SETTINGS:
        s = dec[key]
        for d in SWEEP_DOMAINS:
            cells = []
            for n in GRID:
                g, _ = median_run(s, d, n)
                if g is None and extra:
                    r = dec[extra]["runs"][0][d]
                    cells.append(f"{100 * gain(r['dflash']['ms'], r['cast'][str(n)]['ms']):+.1f}%‡")
                else:
                    cells.append(f"{100 * g:+.1f}%")
            print(f"{gpu:10s}{tgt:10s}{DOMAIN_NAMES[d]:>10s}" + "".join(f"{c:>9s}" for c in cells))


def table9(dec):
    print("\nTable 9 and Figure 5. Tree shape at 16 packed tokens, DFlash against CAST at N=15")
    print(f"{'GPU':10s}{'target':10s}{'domain':>10s}{'DFlash R':>9s}{'N=15 R':>8s}"
          f"{'DFlash ms':>10s}{'N=15 ms':>9s}{'shape gain':>11s}")
    for gpu, tgt, key, _ in SETTINGS:
        s = dec[key]
        for d in SWEEP_DOMAINS:
            g, rec = median_run(s, d, 15)
            c = rec["cast"]["15"]
            print(f"{gpu:10s}{tgt:10s}{DOMAIN_NAMES[d]:>10s}{rec['dflash']['R']:9.2f}{c['R']:8.2f}"
                  f"{rec['dflash']['ms']:10.2f}{c['ms']:9.2f}{100 * g:+10.1f}%")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    dec = load("decoding.json")
    table7(dec)
    table8(dec)
    table9(dec)


if __name__ == "__main__":
    main()
