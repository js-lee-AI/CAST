"""Speedup over AR and committed round length R across widths, for one H100 SXM setting."""
import statistics

from common import DOMAIN_NAMES


def print_sweep(setting, domains, dflash_label, ar_digits=2):
    run = setting["runs"][0]
    print(f"{'method (width)':18s}" + "".join(f"{DOMAIN_NAMES[d]:>13s}" for d in domains) + f"{'Avg.':>13s}")
    print(f"{'AR ms/token':18s}" + "".join(f"{run[d]['ar_ms']:>13.{ar_digits}f}" for d in domains))

    def line(label, pick):
        spd, rs = [], []
        for d in domains:
            ms, r = pick(run[d])
            spd.append(run[d]["ar_ms"] / ms)
            rs.append(r)
        cells = [f"{s:.2f} {r:.2f}" for s, r in zip(spd, rs)]
        cells.append(f"{statistics.mean(spd):.2f} {statistics.mean(rs):.2f}")
        print(f"{label:18s}" + "".join(f"{c:>13s}" for c in cells))

    line(dflash_label, lambda r: (r["dflash"]["ms"], r["dflash"]["R"]))
    for n in sorted(run[domains[0]]["cast"], key=int):
        star = "*" if int(n) == setting["n_star"] else ""
        line(f"CAST (N{star}={n})", lambda r, n=n: (r["cast"][n]["ms"], r["cast"][n]["R"]))
