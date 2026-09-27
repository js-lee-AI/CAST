"""Greedy equivalence of CAST and DFlash against AR decoding (batch one).

For each prompt we report the first position where the output departs from
AR greedy decoding and the target's top-2 logit margin there. In bf16 such
departures happen only at near-ties; in fp32 the outputs should match.

Example:
  python scripts/check_equivalence.py --budgets 31,63 --n-prompts 4 --dtype float32
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dflash.model import dflash_generate  # noqa: E402

from cast_trees.data import encode, load_prompts, stop_ids  # noqa: E402
from cast_trees.decode import ar_generate, cast_generate  # noqa: E402
from cast_trees.utils import load_models  # noqa: E402


def first_diff(a, b):
    n = min(a.shape[1], b.shape[1])
    d = (a[0, :n] != b[0, :n]).nonzero(as_tuple=True)[0]
    if d.numel():
        return int(d[0])
    return None if a.shape[1] == b.shape[1] else n


@torch.inference_mode()
def top2_margin(target, prefix):
    v = torch.topk(target(prefix).logits[0, -1].float(), 2).values
    return float(v[0] - v[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--domains", default="gsm8k,mtbench,humaneval")
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--budgets", default="31,63")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = ap.parse_args()

    target, draft, tok = load_models(args.target, args.draft, dtype=getattr(torch, args.dtype))
    stops = stop_ids(tok)
    budgets = [int(b) for b in args.budgets.split(",")]
    count = {"dflash": 0, **{f"cast@{b}": 0 for b in budgets}}
    total = 0
    for dom in args.domains.split(","):
        for p in load_prompts(dom, args.n_prompts):
            ids = encode(tok, p)
            ref = ar_generate(target, ids, args.max_new, stops)
            outs = {"dflash": dflash_generate(draft, target, ids, args.max_new, stops, 0.0)}
            for b in budgets:
                outs[f"cast@{b}"] = cast_generate(draft, target, ids, args.max_new, stops, budget=b)
            total += 1
            for name, o in outs.items():
                d = first_diff(ref, o)
                if d is not None:
                    count[name] += 1
                    print(f"[{dom}] {name} departs at token {d - ids.shape[1]}, "
                          f"top-2 margin {top2_margin(target, ref[:, :d]):.3f}")
    for name, c in count.items():
        print(f"{name}: {total - c}/{total} outputs identical to AR greedy")


if __name__ == "__main__":
    main()
