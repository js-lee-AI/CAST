"""Table 2: the deployed width N* against the sweep oracle in all eight settings.

Gain over DFlash is t_DFlash / t_CAST - 1, averaged over domains. For Blackwell
and A6000 the per-domain gains are the Table 8 cells, one decimal as printed
(the run with the median gain among the Blackwell repeats). The A6000 sweeps end
at N=95, and their N=127 column comes from a second A6000 host.
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import H100_DOMAINS, gain, load, median_run  # noqa: E402

GRID = [15, 31, 47, 63, 95, 127]
ROWS = [
    ("H100 SXM, Qwen3-8B (b16)", "h100sxm_qwen3-8b", None),
    ("H100 SXM, Qwen3-4B (b16)", "h100sxm_qwen3-4b", None),
    ("Blackwell, Qwen3-8B (b16)", "blackwell_qwen3-8b", None),
    ("Blackwell, Qwen3-4B (b16)", "blackwell_qwen3-4b", None),
    ("A6000, Qwen3-8B (b16)", "a6000_qwen3-8b", "a6000_second_host_qwen3-8b"),
    ("A6000, Qwen3-4B (b16)", "a6000_qwen3-4b", "a6000_second_host_qwen3-4b"),
    ("H100 SXM, LLaMA-3.1-8B (b10)", "h100sxm_llama3.1-8b", None),
    ("H100 SXM, Qwen3-Coder-30B-A3B (b16)", "h100sxm_qwen3-coder-30b-a3b", None),
]


def domain_gains(setting, n):
    """Per-domain gains at width n, one decimal of percent as the paper prints them."""
    doms = [d for d in H100_DOMAINS if d in setting["runs"][0]]
    out = []
    for d in doms:
        g, _ = median_run(setting, d, n)
        if g is None:
            return None
        out.append(round(100 * g, 1))
    return out


def mean_gain(setting, n):
    g = domain_gains(setting, n)
    return None if g is None else statistics.mean(g)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    dec = load("decoding.json")
    print("Table 2. Deployed width N* against the sweep oracle, gain over DFlash averaged over domains")
    print(f"{'setting (drafter block)':38s} {'N*':>4s} {'gain at N*':>10s} {'oracle':>7s} {'(gain)':>8s} "
          f"{'gap':>5s} {'gain at 127':>11s}")
    for label, key, extra in ROWS:
        s = dec[key]
        curve = {n: mean_gain(s, n) for n in GRID}
        curve = {n: g for n, g in curve.items() if g is not None}
        n_star = s["n_star"]
        oracle = max(curve, key=curve.get)
        gap = curve[n_star] - curve[oracle]
        if 127 in curve:
            at127 = f"{curve[127]:+.1f}%"
        else:
            # second A6000 host, a separate session, so gains come straight from its own DFlash runs
            run = dec[extra]["runs"][0]
            g = statistics.mean(round(100 * gain(run[d]["dflash"]["ms"], run[d]["cast"]["127"]["ms"]), 1)
                                for d in run)
            at127 = f"{g:+.1f}%‡"
        print(f"{label:38s} {n_star:4d} {curve[n_star]:+9.1f}% {oracle:7d} {curve[oracle]:+7.1f}% "
              f"{gap:+5.1f} {at127:>11s}")
    print("‡ measured on a second A6000 host")


if __name__ == "__main__":
    main()
