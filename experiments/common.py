"""Shared helpers for the scripts that rebuild the paper tables from data/."""
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DOMAIN_NAMES = {"gsm8k": "GSM8K", "math500": "MATH-500", "humaneval": "HumanEval",
                "mbpp": "MBPP", "mtbench": "MT-Bench"}
H100_DOMAINS = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench"]
SWEEP_DOMAINS = ["gsm8k", "mtbench", "humaneval"]


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def gain(dflash_ms, cast_ms):
    """Gain over DFlash, t_DFlash / t_CAST - 1."""
    return dflash_ms / cast_ms - 1.0


def median_run(setting, dom, n):
    """(gain, record) of the run with the median gain at width n.

    With two runs the gain is their mean and the record is None.
    """
    rows = []
    for run in setting["runs"]:
        rec = run.get(dom)
        if rec and str(n) in rec["cast"]:
            rows.append((gain(rec["dflash"]["ms"], rec["cast"][str(n)]["ms"]), rec))
    if not rows:
        return None, None
    rows.sort(key=lambda t: t[0])
    if len(rows) % 2 == 0:
        return statistics.mean(g for g, _ in rows), None
    return rows[len(rows) // 2]


def pct(x, digits=1):
    return f"{100 * x:+.{digits}f}%"
