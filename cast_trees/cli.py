"""The `cast-trees` command. `python -m cast_trees` runs the same thing."""

from __future__ import annotations

import argparse
import json

from . import __version__

# Target-forward latency (ms) by packed tokens, H100 SXM, Qwen3-8B, paper Table 5
H100_N = [16, 24, 32, 48, 64, 96, 128]
H100_MS = [30.6, 30.2, 30.6, 31.2, 31.5, 32.2, 32.6]


def _ints(s):
    return [int(x) for x in s.split(",")]


def _floats(s):
    return [float(x) for x in s.split(",")]


def _demo(args):
    import numpy as np

    from . import block_topk, build_chain, build_tree, expected_accepted_length, predict_width

    width = predict_width(H100_N, H100_MS)
    print(width)
    logits = np.random.default_rng(args.seed).normal(scale=5, size=(15, 64))
    top_idx, top_lp = block_topk(logits)
    for label, nodes in [("DFlash chain", build_chain(top_idx, top_lp, 15)),
                         ("CAST, N=15", build_tree(top_idx, top_lp, 15)),
                         (f"CAST, N={width.n_star}", build_tree(top_idx, top_lp, width.n_star))]:
        print(f"{label:13s} {len(nodes):3d} nodes, expected accepted tokens "
              f"{expected_accepted_length(nodes):.2f}")
    return 0


def _width(args):
    from . import GSM8K_MASSES, predict_width

    draft_ms = args.draft_ms
    if args.probe:
        with open(args.probe) as f:
            probe = json.load(f)
        curve = probe["verify_ms"][args.prefix]
        n = sorted(int(k) for k in curve)
        ms = [curve[str(k)] for k in n]
        draft_ms = probe.get("draft_ms", 0.0) if draft_ms is None else draft_ms
    elif args.n and args.ms:
        n, ms = _ints(args.n), _floats(args.ms)
    else:
        raise SystemExit("give --probe, or --n and --ms")
    rho = GSM8K_MASSES
    if args.masses:
        with open(args.masses) as f:
            m = json.load(f)
        # data/masses.json or the output of scripts/analyze_rounds.py
        rho = m["domains"][args.domain] if "domains" in m else m[args.domain]["rho"]
    grid = _ints(args.grid)
    w = predict_width(n, ms, rho=rho, draft_ms=draft_ms or 0.0, grid=grid)
    print(f"c0 = {w.c0:.2f} ms, c1 = {w.c1:.4f} ms per packed token, c_draft = {draft_ms or 0.0:.2f} ms")
    for N, r, thr in w.trace:
        if N in grid or N == w.n_stop:
            print(f"  N={N:3d}  rho_(N+1)={r:.4f}  threshold={thr:.4f}")
    print(w)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cast-trees",
                                     description="Cost-aware speculative trees from one-pass block drafters")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the CPU quickstart")
    demo.add_argument("--seed", type=int, default=0)
    demo.set_defaults(func=_demo)

    width = sub.add_parser("width", help="predict the tree width N* from a latency probe")
    width.add_argument("--probe", help="output of scripts/probe_cost.py single")
    width.add_argument("--prefix", default="1024", help="KV-prefix length to read from --probe")
    width.add_argument("--n", help="packed-token counts, e.g. 16,32,64,128")
    width.add_argument("--ms", help="target-forward latency in ms at each count")
    width.add_argument("--draft-ms", type=float, default=None, help="drafter forward in ms")
    width.add_argument("--masses", help="JSON with a prefix-mass curve (default is the packaged GSM8K curve)")
    width.add_argument("--domain", default="gsm8k")
    width.add_argument("--grid", default="15,31,47,63,95,127")
    width.set_defaults(func=_width)

    args = parser.parse_args(argv)
    return args.func(args)
