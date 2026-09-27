"""Batch-one CAST decoding over an unmodified DFlash drafter."""
import time

import torch
import torch.nn.functional as F
from dflash.model import extract_context_feature
from transformers import DynamicCache

from .tree import build_tree
from .verify import gather_cache, topk_marginals, tree_attention_mask


def pick(logits, temperature, generator=None):
    if temperature < 1e-5:
        return int(logits.argmax())
    probs = F.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


def target_probs(logits, temperature):
    """Target sampling distribution; a point mass on the argmax at T=0."""
    if temperature < 1e-5:
        p = torch.zeros_like(logits, dtype=torch.float32)
        p[int(logits.argmax())] = 1.0
        return p
    return F.softmax(logits.float() / temperature, dim=-1)


def spec_sample_node(p, child_tokens, child_q, generator=None):
    """Recursive speculative sampling over the children of one node.

    Children are tried in q-weighted order without replacement, and after a
    rejection p becomes the normalized residual (p - q)_+. Returns (token, i)
    if child i is accepted, else (token, None). The token is distributed as p.
    """
    n = int(child_tokens.numel())
    if n == 0:
        return int(torch.multinomial(p, 1, generator=generator)), None

    # residual mass on the candidates (rc) and off them (om), kept normalized
    pc = p[child_tokens].double().tolist()
    qc = child_q[child_tokens].double().tolist()
    rc = list(pc)
    om = max(0.0, 1.0 - sum(pc))
    avail = [True] * n
    coins = torch.rand(2 * n, device=p.device, generator=generator).tolist()

    for step in range(n):
        idxs = [i for i in range(n) if avail[i]]
        zq = sum(qc[i] for i in idxs)
        if zq <= 0:
            break
        # draw the next candidate from the renormalized proposal
        acc, chosen = 0.0, idxs[-1]
        for i in idxs:
            acc += qc[i] / zq
            if coins[2 * step] <= acc:
                chosen = i
                break
        qt = qc[chosen] / zq
        a = min(1.0, rc[chosen] / qt) if qt > 0 else 0.0
        if a >= 1.0 or coins[2 * step + 1] <= a:
            return int(child_tokens[chosen]), chosen
        for i in idxs:
            rc[i] = max(0.0, rc[i] - qc[i] / zq)
        z = om + sum(rc)
        if z <= 0:
            return int(torch.multinomial(p, 1, generator=generator)), None
        om /= z
        rc = [x / z for x in rc]
        avail[chosen] = False

    # every child rejected: sample the correction from the full residual
    cur = p.double().clone()
    pcs = sum(pc)
    cur.mul_(om / (1.0 - pcs) if (1.0 - pcs) > 0 else 0.0)
    cur[child_tokens] = torch.tensor(rc, dtype=torch.float64, device=p.device)
    cur.clamp_(min=0.0)
    s = cur.sum()
    if s <= 0:
        return int(torch.multinomial(p, 1, generator=generator)), None
    return int(torch.multinomial(cur / s, 1, generator=generator)), None


def walk_tree(nodes, logits, temperature=0.0, d_logits=None, draft_temp=1.0, generator=None):
    """Walk the verified tree and return (accepted path, next root token).

    Greedy at T=0. At T>0 each visited node runs spec_sample_node with the
    drafter marginal of its depth as the proposal.
    """
    children = {i: [] for i in range(-1, len(nodes))}
    for i, nd in enumerate(nodes):
        children[nd.parent].append(i)
    path, cur = [], -1

    if temperature < 1e-5:
        while True:
            tok = pick(logits[cur + 1], temperature, generator)
            hit = next((c for c in children[cur] if nodes[c].token == tok), None)
            if hit is None:
                return path, tok
            path.append(hit)
            cur = hit

    q_cache = {}
    while True:
        p = target_probs(logits[cur + 1], temperature)
        kids = children[cur]
        if not kids:
            return path, int(torch.multinomial(p, 1, generator=generator))
        depth = nodes[kids[0]].depth           # siblings share a depth
        if depth not in q_cache:
            q_cache[depth] = F.softmax(d_logits[depth - 1].float() / draft_temp, dim=-1)
        toks = torch.tensor([nodes[c].token for c in kids], device=logits.device)
        tok, j = spec_sample_node(p, toks, q_cache[depth], generator)
        if j is None:
            return path, tok
        path.append(kids[j])
        cur = kids[j]


@torch.inference_mode()
def cast_generate(draft, target, input_ids, max_new_tokens, stop_token_ids,
                  budget=63, branch_k=8, temperature=0.0, draft_temp=1.0,
                  generator=None, block_size=None, return_stats=False):
    """CAST decoding with a best-first tree of `budget` nonroot nodes.

    One drafter pass per round, one ancestor-masked verify pass, then the KV of
    the accepted path is kept. The clock starts right before the first verify.
    """
    device = input_ids.device
    blk_len = block_size or draft.block_size        # root + L future positions
    mask_id = draft.mask_token_id
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    dtype = next(target.parameters()).dtype

    out_ids = torch.full((1, max_len + 2 * blk_len + 4), mask_id, dtype=torch.long, device=device)
    pos_ids = torch.arange(out_ids.shape[1], device=device).unsqueeze(0)
    kv_t, kv_d = DynamicCache(), DynamicCache()

    out = target(input_ids, position_ids=pos_ids[:, :n_in], past_key_values=kv_t,
                 use_cache=True, logits_to_keep=1, output_hidden_states=True)
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in] = pick(out.logits[0, -1], temperature, generator)
    ctx = extract_context_feature(out.hidden_states, draft.target_layer_ids)

    start, t0 = n_in, None
    lens, packed = [], []
    while start < max_len:
        blk = out_ids[:, start:start + blk_len].clone()
        blk[:, 1:] = mask_id
        h = draft(target_hidden=ctx, noise_embedding=target.model.embed_tokens(blk),
                  position_ids=pos_ids[:, kv_d.get_seq_length():start + blk_len],
                  past_key_values=kv_d, use_cache=True, is_causal=False)
        d_logits = target.lm_head(h[:, 1 - blk_len:, :])[0]         # (L, V)
        kv_d.crop(start)

        top_idx, top_lp = topk_marginals(d_logits, branch_k)
        nodes = build_tree(top_idx, top_lp, budget, branch_k)

        if t0 is None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        toks = torch.tensor([[int(out_ids[0, start])] + [nd.token for nd in nodes]], device=device)
        pos = torch.tensor([[start] + [start + nd.depth for nd in nodes]], device=device)
        n_prefix = kv_t.get_seq_length()
        out = target(toks, position_ids=pos, past_key_values=kv_t, use_cache=True,
                     attention_mask=tree_attention_mask(n_prefix, nodes, device, dtype),
                     output_hidden_states=True)

        path, bonus = walk_tree(nodes, out.logits[0], temperature, d_logits, draft_temp, generator)
        a = len(path)
        lens.append(a + 1)
        packed.append(1 + len(nodes))

        for d, i in enumerate(path):
            out_ids[0, start + 1 + d] = nodes[i].token
        out_ids[0, start + a + 1] = bonus
        keep = [0] + [i + 1 for i in path]
        gather_cache(kv_t, n_prefix, keep)
        feats = extract_context_feature(out.hidden_states, draft.target_layer_ids)[0]
        ctx = feats[keep, :][None]

        start += a + 1
        if stop_token_ids is not None and any(
                t in out_ids[0, n_in:start].tolist() for t in stop_token_ids):
            break

    final = out_ids[:, :min(start + 1, max_len)]
    if stop_token_ids is not None:
        hits = torch.isin(final[0, n_in:], torch.tensor(stop_token_ids, device=device))
        idx = hits.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            final = final[:, :n_in + idx[0] + 1]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0 if t0 is not None else 0.0
    if not return_stats:
        return final
    n_out = final.shape[1] - n_in
    return final, {
        "ms_per_token": elapsed * 1000.0 / max(n_out, 1),
        "R": sum(lens) / max(len(lens), 1),          # committed round length
        "packed": sum(packed) / max(len(packed), 1),
        "rounds": len(lens),
        "n_out": n_out,
    }


@torch.inference_mode()
def ar_generate(target, input_ids, max_new_tokens, stop_token_ids, temperature=0.0,
                generator=None, return_stats=False):
    """Plain autoregressive decoding with the same decode-only clock."""
    device = input_ids.device
    kv = DynamicCache()
    out = target(input_ids, past_key_values=kv, use_cache=True, logits_to_keep=1)
    toks = [pick(out.logits[0, -1], temperature, generator)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        out = target(torch.tensor([[toks[-1]]], device=device), past_key_values=kv, use_cache=True)
        toks.append(pick(out.logits[0, -1], temperature, generator))
        if stop_token_ids is not None and toks[-1] in stop_token_ids:
            break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    final = torch.cat([input_ids, torch.tensor([toks], device=device)], dim=1)
    if not return_stats:
        return final
    return final, {"ms_per_token": elapsed * 1000.0 / len(toks), "n_out": len(toks)}
