"""Ragged tree attention against a dense fp32 reference (needs a GPU)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu


def scenario(B=8, Hq=32, Hkv=8, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    qs, ks, vs, bits, reqs = [], [], [], [], []
    cu_q, cu_kv = [0], [0]
    for _ in range(B):
        S = int(torch.randint(200, 400, (1,), generator=g))
        P = int(torch.randint(2, 64, (1,), generator=g))
        parent = [-1] + [int(torch.randint(0, i, (1,), generator=g)) for i in range(1, P)]
        qs.append(torch.randn(P, Hq, D, generator=g))
        ks.append(torch.randn(S + P, Hkv, D, generator=g))
        vs.append(torch.randn(S + P, Hkv, D, generator=g))
        for i in range(P):
            w, j = 0, i
            while j != -1:
                w |= 1 << j
                j = parent[j]
            bits.append(w)
        cu_q.append(cu_q[-1] + P)
        cu_kv.append(cu_kv[-1] + S + P)
        reqs.append((S, P, parent))
    dev = "cuda"
    as_i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return dict(q=torch.cat(qs).to(dev), k=torch.cat(ks).to(dev), v=torch.cat(vs).to(dev),
                cu_q=as_i32(cu_q), cu_kv=as_i32(cu_kv),
                prefix=as_i32([r[0] for r in reqs]), pack=as_i32([r[1] for r in reqs]),
                bits=torch.tensor(np.array(bits, dtype=np.uint64).view(np.int64), device=dev),
                reqs=reqs)


def reference(sc):
    q, k, v = sc["q"], sc["k"], sc["v"]
    G = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    cq, ckv = sc["cu_q"].tolist(), sc["cu_kv"].tolist()
    for b, (S, P, parent) in enumerate(sc["reqs"]):
        mask = torch.zeros(P, S + P, dtype=torch.bool, device=q.device)
        mask[:, :S] = True
        for i in range(P):
            j = i
            while j != -1:
                mask[i, S + j] = True
                j = parent[j]
        qb = q[cq[b]:cq[b + 1]].transpose(0, 1)
        kb = k[ckv[b]:ckv[b + 1]].transpose(0, 1).repeat_interleave(G, 0)
        vb = v[ckv[b]:ckv[b + 1]].transpose(0, 1).repeat_interleave(G, 0)
        att = (qb @ kb.transpose(1, 2)) / math.sqrt(q.shape[-1])
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        out[cq[b]:cq[b + 1]] = (att @ vb).transpose(0, 1)
    return out


def test_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from cast_trees.tree_attn import varlen_tree_attention
    sc = scenario()
    for dtype in (torch.float32, torch.bfloat16):
        x = {n: sc[n].to(dtype) for n in ("q", "k", "v")}
        ref = reference(dict(sc, **{n: t.float() for n, t in x.items()}))
        got = varlen_tree_attention(x["q"], x["k"], x["v"], sc["cu_q"], sc["cu_kv"],
                                    sc["prefix"], sc["pack"], sc["bits"]).float()
        err = (got - ref).abs()
        print(f"{dtype}: max abs err {err.max().item():.2e}, mean {err.mean().item():.2e}")
        assert err.max().item() < 5e-2 and err.mean().item() < 5e-3
