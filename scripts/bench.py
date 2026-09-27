"""Batch-one decoding panel: AR, DFlash and CAST over a width sweep.

Example:
  python scripts/bench.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --domains gsm8k,mtbench,humaneval --budgets 15,31,47,63,95,127 --out results/qwen3_8b.json
"""
import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dflash.model import dflash_generate  # noqa: E402

from cast_trees.data import encode, load_prompts, stop_ids  # noqa: E402
from cast_trees.decode import ar_generate, cast_generate  # noqa: E402
from cast_trees.utils import dump, env_info, load_models  # noqa: E402


def mean(x):
    return statistics.mean(x) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--domains", default="gsm8k,mtbench,humaneval")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--budgets", default="15,31,47,63,95,127")
    ap.add_argument("--branch-k", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=None,
                    help="run the drafter on a shorter block (default: its own block size)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--draft-temp", type=float, default=1.0)
    ap.add_argument("--seeds", default="0", help="comma list; at T=0 these are plain repeats")
    ap.add_argument("--skip-ar", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)
    dev = "cuda"
    target, draft, tok = load_models(args.target, args.draft, dev)
    stops = stop_ids(tok)
    T = args.temperature
    budgets = [int(b) for b in args.budgets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    domains = args.domains.split(",")
    blk = args.block_size
    gen = lambda s: torch.Generator(device=dev).manual_seed(s) if T > 0 else None

    # warm up kernels once
    w = encode(tok, load_prompts(domains[0], 1)[0], dev)
    cast_generate(draft, target, w, 16, stops, budget=max(budgets), branch_k=args.branch_k,
                  temperature=T, generator=gen(0), block_size=blk)
    dflash_generate(draft, target, w, 16, stops, T, block_size=blk)
    if not args.skip_ar:
        ar_generate(target, w, 16, stops, T, gen(0))

    res = {"env": env_info(), "config": vars(args), "domains": {}}
    for dom in domains:
        prompts = [encode(tok, p, dev) for p in load_prompts(dom, args.n_prompts)]
        n = len(prompts)
        # per prompt, summed over seeds
        ar, ch, ch_R = [0.0] * n, [0.0] * n, [0.0] * n
        ca = {b: {"ms": [0.0] * n, "R": [0.0] * n, "packed": [0.0] * n} for b in budgets}
        for seed in seeds:
            for i, ids in enumerate(prompts):
                s = seed * 100000 + i
                if not args.skip_ar:
                    _, st = ar_generate(target, ids, args.max_new, stops, T, gen(s), return_stats=True)
                    ar[i] += st["ms_per_token"]
                torch.manual_seed(s)       # DFlash samples from the global generator
                r = dflash_generate(draft, target, ids, args.max_new, stops, T, block_size=blk,
                                    return_stats=True)
                ch[i] += r.time_per_output_token * 1000.0
                ch_R[i] += sum(r.acceptance_lengths) / max(len(r.acceptance_lengths), 1)
                for b in budgets:
                    _, st = cast_generate(draft, target, ids, args.max_new, stops, budget=b,
                                          branch_k=args.branch_k, temperature=T,
                                          draft_temp=args.draft_temp, generator=gen(s),
                                          block_size=blk, return_stats=True)
                    for key in ("ms", "R", "packed"):
                        ca[b][key][i] += st["ms_per_token" if key == "ms" else key]
                if (i + 1) % 10 == 0:
                    log(f"{dom} seed {seed}: {i + 1}/{n}")
        k = len(seeds)
        avg = lambda xs: [x / k for x in xs]
        rec = {"n_prompts": n, "chain_ms": mean(avg(ch)), "chain_R": mean(avg(ch_R)),
               "chain_ms_per_prompt": avg(ch), "cast": {}}
        if not args.skip_ar:
            rec["ar_ms"] = mean(avg(ar))
            rec["ar_ms_per_prompt"] = avg(ar)
        for b in budgets:
            ms = mean(avg(ca[b]["ms"]))
            rec["cast"][b] = {
                "ms": ms, "R": mean(avg(ca[b]["R"])), "packed": mean(avg(ca[b]["packed"])),
                "gain_vs_dflash": rec["chain_ms"] / ms - 1.0,
                "speedup_vs_ar": rec["ar_ms"] / ms if "ar_ms" in rec else None,
                "ms_per_prompt": avg(ca[b]["ms"]),
            }
        res["domains"][dom] = rec
        best = max(budgets, key=lambda b: rec["cast"][b]["gain_vs_dflash"])
        log(f"{dom}: DFlash {rec['chain_ms']:.2f} ms/tok (R={rec['chain_R']:.2f}), "
            f"best N={best} {rec['cast'][best]['ms']:.2f} ms/tok "
            f"({100 * rec['cast'][best]['gain_vs_dflash']:+.1f}%)")
        dump(res, args.out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
