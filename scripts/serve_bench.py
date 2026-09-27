"""Goodput of the batched harness under each allocation policy.

For every policy and batch size B we decode the first B GSM8K prompts
(cycled when B is larger), start the clock after two warm-up rounds and keep
the best of `--repeats` runs. With --check, every policy is first compared
with autoregressive greedy decoding at B = --check-batch.

Example:
  python scripts/serve_bench.py --batches 1,4,8,16,32 --check --out results/serving.json
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dflash.model import dflash_generate  # noqa: E402

from cast_trees.data import encode, load_prompts, stop_ids  # noqa: E402
from cast_trees.decode import ar_generate  # noqa: E402
from cast_trees.serving import POLICIES, BatchedServer  # noqa: E402
from cast_trees.utils import dump, env_info, load_models  # noqa: E402


@torch.inference_mode()
def top2_margin(target, prefix):
    v = torch.topk(target(prefix.unsqueeze(0).cuda()).logits[0, -1].float(), 2).values
    return float(v[0] - v[1])


def first_diff(a, b):
    n = min(a.numel(), b.numel())
    d = (a[:n] != b[:n]).nonzero(as_tuple=True)[0]
    if d.numel():
        return int(d[0])
    return None if a.numel() == b.numel() else n


def check(server, target, draft, prompts, stops, max_new, B, policies, kw, log):
    """Compare every policy with AR greedy; report the top-2 logit margin at each divergence."""
    ar = [ar_generate(target, p[None], max_new, stops)[0].cpu() for p in prompts]
    ref = [dflash_generate(draft, target, p[None], max_new, stops, 0.0)[0].cpu() for p in prompts]
    divs = [first_diff(a, g) for a, g in zip(ar, ref)]
    margins = [top2_margin(target, a[:d]) for a, d in zip(ar, divs) if d is not None]
    out = {"DFLASH_BATCH1": {"diverged": len(margins), "n": len(prompts), "margins": margins}}
    log(f"  {'DFLASH_BATCH1':18s} {len(margins)}/{len(prompts)} differ from AR, "
        f"max top-2 margin at divergence {max(margins, default=0.0):.3f}")
    for pol in policies:
        got = []
        for s in range(0, len(prompts), B):
            seqs, _ = server.generate(prompts[s:s + B], max_new, stops, pol, **kw)
            got += [x.cpu() for x in seqs]
        divs = [first_diff(a, g) for a, g in zip(ar, got)]
        margins = [top2_margin(target, a[:d]) for a, d in zip(ar, divs) if d is not None]
        out[pol] = {"diverged": len(margins), "n": len(prompts), "margins": margins}
        log(f"  {pol:18s} {len(margins)}/{len(prompts)} differ from AR, "
            f"max top-2 margin at divergence {max(margins, default=0.0):.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--w-wide", type=int, default=48)
    ap.add_argument("--w-base", type=int, default=8)
    ap.add_argument("--w-max", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-batch", type=int, default=4)
    ap.add_argument("--check-per-domain", type=int, default=6)
    ap.add_argument("--check-max-new", type=int, default=160)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)
    target, draft, tok = load_models(args.target, args.draft)
    stops = stop_ids(tok)
    server = BatchedServer(target, draft)
    policies = args.policies.split(",")
    kw = dict(w_wide=args.w_wide, w_base=args.w_base, w_max=args.w_max)
    res = {"env": env_info(), "config": vars(args)}

    if args.check:
        prompts = [encode(tok, p)[0] for d in ("gsm8k", "mtbench", "humaneval")
                   for p in load_prompts(d, args.check_per_domain)]
        log(f"greedy check on {len(prompts)} prompts at B={args.check_batch}")
        res["check"] = check(server, target, draft, prompts, stops, args.check_max_new,
                             args.check_batch, policies, dict(kw, warmup_rounds=0), log)

    pool = [encode(tok, p)[0] for p in load_prompts("gsm8k", args.n_prompts)]
    res["goodput"] = {}
    for pol in policies:
        res["goodput"][pol] = {}
        for B in [int(x) for x in args.batches.split(",")]:
            batch = [pool[i % len(pool)] for i in range(B)]
            runs = [server.generate(batch, args.max_new, stops, pol, measure=True, **kw)[1]
                    for _ in range(args.repeats)]
            best = max(runs, key=lambda s: s.get("goodput_tok_s", 0.0))
            res["goodput"][pol][B] = best
            log(f"{pol:18s} B={B:2d}: {best.get('goodput_tok_s', 0.0):8.1f} tok/s  "
                f"R={best['R']:.2f}  nodes/round={best.get('nodes_per_round', 0.0):.1f}")
        dump(res, args.out)


if __name__ == "__main__":
    main()
