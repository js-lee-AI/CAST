"""Target-forward and drafter cost probes.

single : batch-one target forward of n packed tokens over a random KV prefix
         (8 warmups, mean of 30 timed iterations) and the drafter forward.
batched: median latency of B requests verifying n tokens each over an L-token
         prefix, and of one batched drafter pass.

Examples:
  python scripts/probe_cost.py single --target Qwen/Qwen3-8B --prefix 1024 --out results/cost_8b.json
  python scripts/probe_cost.py single --target Qwen/Qwen3-8B --prefix 1024,4096,8192,16384 --out results/cost_prefix.json
  python scripts/probe_cost.py batched --target Qwen/Qwen3-8B --batches 1,8,16,32 --out results/cost_load.json
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dflash.model import DFlashDraftModel  # noqa: E402

from cast_trees.utils import dump, env_info  # noqa: E402

N_GRID = "1,4,8,16,24,32,48,64,96,128"


def mean_ms(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def median_ms(fn, reset=None, reps=25, warmup=8):
    ts = []
    for r in range(warmup + reps):
        if reset is not None:
            reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        if r >= warmup:
            ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1000.0


@torch.inference_mode()
def verify_curve(target, B, prefix, ns, batched=False):
    vocab = target.config.vocab_size
    ids = torch.randint(0, vocab, (B, prefix), device="cuda")
    pos = torch.arange(prefix + max(ns), device="cuda").unsqueeze(0).expand(B, -1)
    kv = DynamicCache()
    target(ids, position_ids=pos[:, :prefix], past_key_values=kv, use_cache=True)
    curve = {}
    for n in ns:
        toks = torch.randint(0, vocab, (B, n), device="cuda")
        p = pos[:, prefix:prefix + n]
        fwd = lambda: target(toks, position_ids=p, past_key_values=kv, use_cache=True)
        if batched:
            curve[n] = median_ms(fwd, reset=lambda: kv.crop(prefix))
        else:
            def step():
                fwd()
                kv.crop(prefix)
            curve[n] = mean_ms(step)
        print(f"B={B} prefix={prefix} n={n}: {curve[n]:.2f} ms", flush=True)
    return curve


@torch.inference_mode()
def draft_cost(draft, target, B, ctx_len, timer):
    dt = next(draft.parameters()).dtype
    H = target.config.hidden_size
    th = torch.randn(B, ctx_len, H * len(draft.target_layer_ids), device="cuda", dtype=dt)
    noise = torch.randn(B, draft.block_size, H, device="cuda", dtype=dt)
    pos = torch.arange(ctx_len + draft.block_size, device="cuda").unsqueeze(0).expand(B, -1)
    return timer(lambda: draft(target_hidden=th, noise_embedding=noise, position_ids=pos,
                               past_key_values=None, use_cache=False, is_causal=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "batched"])
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--prefix", default=None, help="comma list of prefix lengths")
    ap.add_argument("--n", default=None, help="comma list of packed-token counts")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").to("cuda").eval()
    draft = DFlashDraftModel.from_pretrained(args.draft, dtype=torch.bfloat16).to("cuda").eval()
    res = {"env": env_info(), "config": vars(args)}

    if args.mode == "single":
        ns = [int(x) for x in (args.n or N_GRID).split(",")]
        res["verify_ms"] = {}
        for L in [int(x) for x in (args.prefix or "1024").split(",")]:
            res["verify_ms"][L] = verify_curve(target, 1, L, ns)
            torch.cuda.empty_cache()
        res["draft_ms"] = draft_cost(draft, target, 1, 64, mean_ms)
    else:
        # widths W are nonroot nodes, each request verifies W + 1 tokens
        ws = [int(x) for x in (args.n or "1,2,4,8,12,15,16,24,31,48,64").split(",")]
        L = int(args.prefix or 512)
        res["verify_ms"], res["draft_ms"] = {}, {}
        for B in [int(x) for x in args.batches.split(",")]:
            res["draft_ms"][B] = draft_cost(draft, target, B, L, median_ms)
            curve = verify_curve(target, B, L, [w + 1 for w in ws], batched=True)
            res["verify_ms"][B] = {w: curve[w + 1] for w in ws}
            print(f"B={B}: draft {res['draft_ms'][B]:.2f} ms", flush=True)
            torch.cuda.empty_cache()
    dump(res, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
