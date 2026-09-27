"""Greedy DFlash decoding that logs every round for offline replay.

Each round stores the drafter's top-K tokens and logits at every future
position, the chain's accepted count and, once the decode ends, the realized
greedy continuation after the round's root.

Example:
  python scripts/collect_rounds.py --target Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 \
      --n-prompts 40 --max-new 384 --topk 32 --out-dir results/rounds
"""
import argparse
import os
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dflash.model import extract_context_feature, sample  # noqa: E402

from cast_trees.data import encode, load_prompts, stop_ids  # noqa: E402
from cast_trees.utils import load_models  # noqa: E402


@torch.inference_mode()
def logged_decode(draft, target, input_ids, max_new_tokens, stop_token_ids, topk):
    dev = input_ids.device
    B, mask_id = draft.block_size, draft.mask_token_id
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    out_ids = torch.full((1, max_len + B), mask_id, dtype=torch.long, device=dev)
    pos_ids = torch.arange(out_ids.shape[1], device=dev).unsqueeze(0)
    kv_t, kv_d = DynamicCache(), DynamicCache()

    out = target(input_ids, position_ids=pos_ids[:, :n_in], past_key_values=kv_t, use_cache=True,
                 logits_to_keep=1, output_hidden_states=True)
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in:n_in + 1] = sample(out.logits, 0.0)
    ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)

    start, logs = n_in, []
    while start < max_len:
        blk = out_ids[:, start:start + B].clone()
        h = draft(target_hidden=ctx, noise_embedding=target.model.embed_tokens(blk),
                  position_ids=pos_ids[:, kv_d.get_seq_length():start + B],
                  past_key_values=kv_d, use_cache=True, is_causal=False)
        d_logits = target.lm_head(h[:, 1 - B:, :])[0]
        kv_d.crop(start)
        blk[0, 1:] = d_logits.argmax(dim=-1)

        out = target(blk, position_ids=pos_ids[:, start:start + B], past_key_values=kv_t,
                     use_cache=True, output_hidden_states=True)
        post = sample(out.logits, 0.0)
        a = (blk[:, 1:] == post[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        top = torch.topk(d_logits.float(), topk, dim=-1)
        logs.append({"start": start, "accepted": a,
                     "top_idx": top.indices.cpu(), "top_val": top.values.cpu()})

        out_ids[:, start:start + a + 1] = blk[:, :a + 1]
        out_ids[:, start + a + 1] = post[:, a]
        start += a + 1
        kv_t.crop(start)
        ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)[:, :a + 1, :]
        if stop_token_ids is not None and any(t in out_ids[0, n_in:start].tolist() for t in stop_token_ids):
            break

    realized = out_ids[0, :min(start + 1, max_len)].cpu()
    for lg in logs:
        s = lg.pop("start")
        lg["realized"] = realized[s + 1:s + lg["top_idx"].shape[0] + 1].clone()
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--domains", default="gsm8k,mtbench,humaneval")
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    target, draft, tok = load_models(args.target, args.draft)
    stops = stop_ids(tok)
    os.makedirs(args.out_dir, exist_ok=True)
    for dom in args.domains.split(","):
        rounds = []
        for i, p in enumerate(load_prompts(dom, args.n_prompts)):
            rounds += logged_decode(draft, target, encode(tok, p), args.max_new, stops, args.topk)
            if (i + 1) % 10 == 0:
                print(f"{dom}: {i + 1} prompts, {len(rounds)} rounds", flush=True)
        torch.save(rounds, os.path.join(args.out_dir, f"{dom}.pt"))


if __name__ == "__main__":
    main()
