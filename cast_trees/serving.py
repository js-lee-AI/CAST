"""Batched decoding with a shared verification budget.

Requests decode in lockstep rounds: drafter pass, allocation, one target pass
over all trees, cache compaction.

Policies:
  FIXED_CHAIN         rank-0 chain of the full block (standard DFlash)
  FIXED_WIDE          w_wide nodes for every request
  UNIFORM             w_base nodes for every request
  PER_REQUEST         global best-first over B * w_base nodes, dense verify
  PER_REQUEST_VARLEN  same allocation, ragged verify (tree_attn.py)
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from dflash.model import extract_context_feature
from transformers import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .decode import pick, walk_tree
from .tree import build_chain, build_tree, select_global_topk
from .verify import cache_layers
from .tree_attn import varlen_tree_attention

POLICIES = ["FIXED_CHAIN", "FIXED_WIDE", "UNIFORM", "PER_REQUEST", "PER_REQUEST_VARLEN"]


class _VarlenPlan:
    """State of one ragged verify pass, read by the attention hook below."""
    active = False

    def set(self, **kw):
        self.__dict__.update(kw)
        self.layer = 0
        self.active = True

    def clear(self):
        self.active = False
        self.buf_k = self.buf_v = None


PLAN = _VarlenPlan()
PACK_KV = {"k": [], "v": []}    # post-RoPE pack K/V of every layer, filled during the pass


def _varlen_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    if not PLAN.active:
        # prefill and any other forward: plain SDPA
        mask = attention_mask if not isinstance(attention_mask, bool) else None
        o = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling,
                                           is_causal=mask is None and query.shape[2] > 1,
                                           enable_gqa=True)
        return o.transpose(1, 2).contiguous(), None

    li = PLAN.layer
    PLAN.layer += 1
    PACK_KV["k"][li], PACK_KV["v"][li] = key[0], value[0]
    Hkv, D = key.shape[1], query.shape[-1]
    # ragged k/v = concat(prefix store of b, pack of b) for every request
    k_rag = torch.empty((PLAN.total_kv, Hkv, D), dtype=key.dtype, device=key.device)
    v_rag = torch.empty_like(k_rag)
    k_rag[PLAN.pre_dst] = PLAN.buf_k[li][PLAN.pre_batch, :, PLAN.pre_pos, :]
    v_rag[PLAN.pre_dst] = PLAN.buf_v[li][PLAN.pre_batch, :, PLAN.pre_pos, :]
    k_rag[PLAN.pack_dst] = key[0].transpose(0, 1)
    v_rag[PLAN.pack_dst] = value[0].transpose(0, 1)
    # fp32 inside attention: large outlier channels in K make a bf16 dot lossy
    out = varlen_tree_attention(query[0].transpose(0, 1).contiguous().float(), k_rag.float(),
                                v_rag.float(), PLAN.cu_q, PLAN.cu_kv, PLAN.prefix_len,
                                PLAN.pack_len, PLAN.ancestor_mask, sm_scale=scaling)
    return out.to(query.dtype).unsqueeze(0), None


ALL_ATTENTION_FUNCTIONS.register("cast_varlen", _varlen_attention)


class BatchedServer:
    def __init__(self, target, draft, branch_k=8, temperature=0.0, draft_temp=1.0, generator=None):
        self.target, self.draft = target, draft
        self.branch_k = branch_k
        self.temperature, self.draft_temp, self.generator = temperature, draft_temp, generator
        self.blk_len = draft.block_size
        self.mask_id = draft.mask_token_id
        self.layer_ids = draft.target_layer_ids
        self.dtype = next(target.parameters()).dtype
        self.device = next(target.parameters()).device

    # ---- prefill into a left-padded batch cache ----
    @torch.inference_mode()
    def _prefill(self, prompts):
        B, lens = len(prompts), [int(p.numel()) for p in prompts]
        S = max(lens)
        inp = torch.full((B, S), self.mask_id, dtype=torch.long, device=self.device)
        valid = torch.zeros((B, S), dtype=torch.bool, device=self.device)
        pos = torch.zeros((B, S), dtype=torch.long, device=self.device)
        for b, ids in enumerate(prompts):
            inp[b, S - lens[b]:] = ids.to(self.device)
            valid[b, S - lens[b]:] = True
            pos[b, S - lens[b]:] = torch.arange(lens[b], device=self.device)
        causal = torch.arange(S, device=self.device)
        vis = (causal[None, :, None] >= causal[None, None, :]) & valid[:, None, :]
        amask = torch.zeros((B, 1, S, S), dtype=self.dtype, device=self.device)
        amask.masked_fill_(~vis[:, None], torch.finfo(self.dtype).min)
        kv = DynamicCache()
        out = self.target(inp, position_ids=pos, past_key_values=kv, use_cache=True,
                          attention_mask=amask, output_hidden_states=True)
        feats = extract_context_feature(out.hidden_states, self.layer_ids)
        roots = torch.tensor([pick(out.logits[b, S - 1], self.temperature, self.generator)
                              for b in range(B)], device=self.device)
        ctx = [feats[b:b + 1, S - lens[b]:].clone() for b in range(B)]
        return kv, valid, roots, ctx, list(lens)

    # ---- drafting and allocation ----
    @torch.inference_mode()
    def _draft(self, roots, ctx, seq_lens):
        """Drafter pass of every request; returns top-k marginals and raw logits."""
        tops, logits = [], []
        for b in range(roots.shape[0]):
            blk = torch.full((1, self.blk_len), self.mask_id, dtype=torch.long, device=self.device)
            blk[0, 0] = roots[b]
            pos = torch.arange(seq_lens[b] + self.blk_len, device=self.device).unsqueeze(0)
            h = self.draft(target_hidden=ctx[b], noise_embedding=self.target.model.embed_tokens(blk),
                           position_ids=pos, past_key_values=DynamicCache(), use_cache=True,
                           is_causal=False)
            d_logits = self.target.lm_head(h[:, 1 - self.blk_len:, :])[0]
            top = torch.topk(F.log_softmax(d_logits.float(), dim=-1), self.branch_k, dim=-1)
            tops.append((top.indices.cpu(), top.values.cpu()))
            logits.append(d_logits)
        return tops, logits

    def _allocate(self, tops, policy, w_wide, w_base, w_max):
        if policy == "FIXED_CHAIN":
            return [build_chain(ti, tl, self.blk_len - 1) for ti, tl in tops]
        if policy == "FIXED_WIDE":
            return [build_tree(ti, tl, w_wide, self.branch_k) for ti, tl in tops]
        if policy == "UNIFORM":
            return [build_tree(ti, tl, max(1, w_base), self.branch_k) for ti, tl in tops]
        if policy in ("PER_REQUEST", "PER_REQUEST_VARLEN"):
            full = [build_tree(ti, tl, w_max, self.branch_k) for ti, tl in tops]
            return select_global_topk(full, len(tops) * w_base)
        raise ValueError(policy)

    # ---- dense verify: packs right-padded to the largest tree ----
    @torch.inference_mode()
    def _verify_dense(self, kv, valid, roots, trees, seq_lens):
        B, S = valid.shape
        P = max(1 + len(t) for t in trees)
        tok = torch.full((B, P), self.mask_id, dtype=torch.long, device=self.device)
        pos = torch.zeros((B, P), dtype=torch.long, device=self.device)
        vis = np.zeros((B, P, P), dtype=bool)
        for b, nodes in enumerate(trees):
            tok[b, 0], pos[b, 0], vis[b, 0, 0] = roots[b], seq_lens[b], True
            for i, nd in enumerate(nodes):
                tok[b, i + 1] = nd.token
                pos[b, i + 1] = seq_lens[b] + nd.depth
                vis[b, i + 1] = vis[b, nd.parent + 1]
                vis[b, i + 1, i + 1] = True
            for i in range(1 + len(nodes), P):          # padding rows see themselves
                pos[b, i] = seq_lens[b]
                vis[b, i, i] = True
        neg = torch.finfo(self.dtype).min
        amask = torch.zeros((B, 1, P, S + P), dtype=self.dtype, device=self.device)
        amask[:, :, :, :S].masked_fill_(~valid[:, None, None, :].expand(B, 1, P, S), neg)
        amask[:, :, :, S:].masked_fill_(torch.from_numpy(~vis).to(self.device)[:, None], neg)
        out = self.target(tok, position_ids=pos, past_key_values=kv, use_cache=True,
                          attention_mask=amask, output_hidden_states=True)
        return out.logits, extract_context_feature(out.hidden_states, self.layer_ids)

    @staticmethod
    def _compact(cache, keep, rows, device):
        """Keep `keep[b]` positions of every surviving row, left-padded again."""
        lens = [int(keep[b].numel()) for b in rows]
        S = max(lens) if lens else 0
        n_layers, get, put = cache_layers(cache)
        for li in range(n_layers):
            k, v = get(li)
            nk = torch.zeros((len(rows), k.shape[1], S, k.shape[3]), dtype=k.dtype, device=device)
            nv = torch.zeros((len(rows), v.shape[1], S, v.shape[3]), dtype=v.dtype, device=device)
            for nb, b in enumerate(rows):
                if lens[nb]:
                    nk[nb, :, S - lens[nb]:] = k[b, :, keep[b]]
                    nv[nb, :, S - lens[nb]:] = v[b, :, keep[b]]
            put(li, nk, nv)
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = S
        return S

    # ---- ragged verify over a per-request KV store ----
    @torch.inference_mode()
    def _init_store(self, kv, valid, headroom=320):
        lens = valid.sum(dim=1).tolist()
        n_layers, get, _ = cache_layers(kv)
        k0 = get(0)[0]
        B, H, S, D = k0.shape
        cap = int(max(lens)) + headroom
        buf_k = [torch.zeros((B, H, cap, D), dtype=k0.dtype, device=k0.device) for _ in range(n_layers)]
        buf_v = [torch.zeros_like(x) for x in buf_k]
        for li in range(n_layers):
            k, v = get(li)
            for b in range(B):
                n = int(lens[b])
                buf_k[li][b, :, :n] = k[b, :, S - n:]
                buf_v[li][b, :, :n] = v[b, :, S - n:]
        return buf_k, buf_v, [int(x) for x in lens]

    @torch.inference_mode()
    def _verify_varlen(self, buf_k, buf_v, store_len, roots, trees, seq_lens):
        dev = self.device
        B = roots.shape[0]
        pack_len = [1 + len(t) for t in trees]
        assert max(pack_len) <= 64, "the ancestor bitmask holds at most 64 pack tokens"
        tok, pos, bits = [], [], []
        for b, nodes in enumerate(trees):
            base = len(bits)
            tok.append(int(roots[b]))
            pos.append(seq_lens[b])
            bits.append(1)
            for i, nd in enumerate(nodes):
                tok.append(nd.token)
                pos.append(seq_lens[b] + nd.depth)
                bits.append(bits[base + nd.parent + 1] | (1 << (i + 1)))
        cu_q, cu_kv = [0], [0]
        for b in range(B):
            cu_q.append(cu_q[-1] + pack_len[b])
            cu_kv.append(cu_kv[-1] + store_len[b] + pack_len[b])

        prefix_t = torch.tensor(store_len, dtype=torch.int32, device=dev)
        pack_t = torch.tensor(pack_len, dtype=torch.int32, device=dev)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=dev)
        cu_kv_t = torch.tensor(cu_kv, dtype=torch.int32, device=dev)
        # gather / scatter indices that assemble the ragged k/v in every layer
        pre_batch = torch.repeat_interleave(torch.arange(B, device=dev), prefix_t.long())
        cu_pre = torch.tensor([0] + list(np.cumsum(store_len)), device=dev)
        pre_pos = torch.arange(int(cu_pre[-1]), device=dev) - cu_pre[:-1].repeat_interleave(prefix_t.long())
        pre_dst = cu_kv_t.long()[:-1].repeat_interleave(prefix_t.long()) + pre_pos
        within = torch.arange(len(tok), device=dev) - cu_q_t.long()[:-1].repeat_interleave(pack_t.long())
        pack_dst = (cu_kv_t.long()[:-1] + prefix_t.long()).repeat_interleave(pack_t.long()) + within

        n_layers = len(buf_k)
        PACK_KV["k"], PACK_KV["v"] = [None] * n_layers, [None] * n_layers
        PLAN.set(buf_k=buf_k, buf_v=buf_v, prefix_len=prefix_t, pack_len=pack_t, cu_q=cu_q_t,
                 cu_kv=cu_kv_t, total_kv=sum(store_len) + len(tok),
                 ancestor_mask=torch.tensor(np.array(bits, dtype=np.uint64).view(np.int64), device=dev),
                 pre_batch=pre_batch, pre_pos=pre_pos, pre_dst=pre_dst, pack_dst=pack_dst)
        prev = self.target.config._attn_implementation
        self.target.config._attn_implementation = "cast_varlen"
        try:
            out = self.target(torch.tensor([tok], device=dev), position_ids=torch.tensor([pos], device=dev),
                              past_key_values=DynamicCache(), use_cache=True, output_hidden_states=True)
        finally:
            self.target.config._attn_implementation = prev
            PLAN.clear()
        hid = extract_context_feature(out.hidden_states, self.layer_ids)[0]
        # back to (B, P, .) so the walk is shared with the dense path
        P = max(pack_len)
        logits = torch.zeros((B, P, out.logits.shape[-1]), dtype=out.logits.dtype, device=dev)
        hidden = torch.zeros((B, P, hid.shape[-1]), dtype=hid.dtype, device=dev)
        for b in range(B):
            logits[b, :pack_len[b]] = out.logits[0, cu_q[b]:cu_q[b + 1]]
            hidden[b, :pack_len[b]] = hid[cu_q[b]:cu_q[b + 1]]
        return logits, hidden, cu_q

    def _append_store(self, buf_k, buf_v, store_len, cu_q, keep, accepted, rows):
        """Append the accepted pack K/V of surviving requests, drop the rest."""
        dev = self.device
        src, dst_b, dst_p, new_len = [], [], [], []
        for nb, b in enumerate(rows):
            for j, pk in enumerate(keep[b]):
                src.append(cu_q[b] + pk)
                dst_b.append(nb)
                dst_p.append(store_len[b] + j)
            new_len.append(store_len[b] + accepted[b])
        if len(rows) != len(store_len):
            idx = torch.tensor(rows, dtype=torch.long, device=dev)
            buf_k = [x[idx] for x in buf_k]
            buf_v = [x[idx] for x in buf_v]
        cap = buf_k[0].shape[2]
        if max(new_len) > cap:
            new_cap = max(max(new_len), 2 * cap)
            grow = lambda x: torch.cat([x, x.new_zeros(x.shape[0], x.shape[1], new_cap - cap, x.shape[3])], 2)
            buf_k, buf_v = [grow(x) for x in buf_k], [grow(x) for x in buf_v]
        src = torch.tensor(src, dtype=torch.long, device=dev)
        dst_b = torch.tensor(dst_b, dtype=torch.long, device=dev)
        dst_p = torch.tensor(dst_p, dtype=torch.long, device=dev)
        for li in range(len(buf_k)):
            buf_k[li][dst_b, :, dst_p] = PACK_KV["k"][li][:, src].permute(1, 0, 2)
            buf_v[li][dst_b, :, dst_p] = PACK_KV["v"][li][:, src].permute(1, 0, 2)
        return buf_k, buf_v, new_len

    # ---- main loop ----
    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens, stop_token_ids, policy,
                 w_wide=48, w_base=8, w_max=48, warmup_rounds=2, measure=False):
        """Decode a static batch. Returns (outputs with prompts, stats).

        The shared budget follows the requests still in flight. With `measure`
        the clock starts after `warmup_rounds` rounds and goodput counts the
        tokens committed inside the window.
        """
        varlen = policy == "PER_REQUEST_VARLEN"
        if varlen:
            w_max = min(w_max, 48)
        B0 = len(prompts)
        kv, valid, roots, ctx, seq_lens = self._prefill(prompts)
        if varlen:
            buf_k, buf_v, store_len = self._init_store(kv, valid)
            del kv
        gen = [[int(roots[b])] for b in range(B0)]
        alive = list(range(B0))
        lens_hist, rounds, t0 = [], 0, None
        n_timed = committed = nodes_sum = 0
        need_q = self.temperature >= 1e-5

        while alive:
            if measure and t0 is None and rounds >= warmup_rounds:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            tops, d_logits = self._draft(roots, ctx, seq_lens)
            trees = self._allocate(tops, policy, w_wide, w_base, w_max)
            if varlen:
                logits, hidden, cu_q = self._verify_varlen(buf_k, buf_v, store_len, roots, trees, seq_lens)
            else:
                logits, hidden = self._verify_dense(kv, valid, roots, trees, seq_lens)

            B, keep, accepted, bonus, rows = len(alive), [], [], [], []
            for b in range(B):
                path, nxt = walk_tree(trees[b], logits[b], self.temperature,
                                      d_logits[b] if need_q else None, self.draft_temp, self.generator)
                new = [trees[b][i].token for i in path] + [nxt]
                gen[alive[b]].extend(new)
                keep.append([0] + [i + 1 for i in path])
                accepted.append(len(path) + 1)
                bonus.append(nxt)
                lens_hist.append(len(path) + 1)
                if t0 is not None:
                    committed += len(new)
                g = gen[alive[b]]
                done = len(g) >= max_new_tokens or (
                    stop_token_ids is not None and any(s in g for s in stop_token_ids))
                if not done:
                    rows.append(b)
            if t0 is not None:
                n_timed += 1
                nodes_sum += sum(len(t) for t in trees)
            rounds += 1
            if not rows:
                break

            if varlen:
                buf_k, buf_v, store_len = self._append_store(buf_k, buf_v, store_len, cu_q, keep,
                                                             accepted, rows)
            else:
                S = valid.shape[1]
                keep_abs = [torch.cat([valid[b].nonzero(as_tuple=True)[0],
                                       torch.tensor([S + k for k in keep[b]], device=self.device)])
                            for b in range(B)]
                S_new = self._compact(kv, keep_abs, rows, self.device)
                valid = torch.zeros((len(rows), S_new), dtype=torch.bool, device=self.device)
                for nb, b in enumerate(rows):
                    valid[nb, S_new - int(keep_abs[b].numel()):] = True
            # the drafter context grows by the features of the committed positions
            ctx = [torch.cat([ctx[b], hidden[b, keep[b]][None]], dim=1) for b in rows]
            seq_lens = [seq_lens[b] + accepted[b] for b in rows]
            roots = torch.tensor([bonus[b] for b in rows], dtype=torch.long, device=self.device)
            alive = [alive[b] for b in rows]

        outs = []
        for b in range(B0):
            g = gen[b]
            cut = next((i + 1 for i, t in enumerate(g) if stop_token_ids and t in stop_token_ids), None)
            g = g[:cut] if cut is not None else g[:max_new_tokens]
            outs.append(torch.tensor(prompts[b].tolist() + g, dtype=torch.long))
        stats = {"rounds": rounds, "R": sum(lens_hist) / max(len(lens_hist), 1)}
        if measure and t0 is not None:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            stats.update(elapsed_s=elapsed, committed=committed,
                         goodput_tok_s=committed / max(elapsed, 1e-9),
                         nodes_per_round=nodes_sum / max(n_timed, 1))
        return outs, stats
