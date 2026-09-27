"""Ragged tree-verification attention (Triton, forward only).

Each request b owns a key/value stream concat(prefix_b, pack_b) and a pack of
P_b = 1 + n_b query tokens. A query sees the whole prefix and, inside the
pack, the positions set in its int64 ancestor bitmask. Supports GQA and packs
of at most 64 tokens (one bitmask word).
"""
import math

import torch
import triton
import triton.language as tl

BLOCK_M = 32   # query tile height; tiles never cross a request boundary


@triton.jit
def _tree_attn_fwd(
    Q, K, V, Out, AncestorMask, TileReq, TileRow0, TileNRows, CuKV, PrefixLen, PackLen,
    sm_scale,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    H_q: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_h = pid_h // GROUP

    b = tl.load(TileReq + pid_t)
    row0 = tl.load(TileRow0 + pid_t)
    n_rows = tl.load(TileNRows + pid_t)
    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_valid = tl.arange(0, BLOCK_M) < n_rows

    kv_start = tl.load(CuKV + b)
    S_b = tl.load(PrefixLen + b)
    P_b = tl.load(PackLen + b)

    q = tl.load(Q + offs_m[:, None] * stride_qm + pid_h * stride_qh + offs_d[None, :] * stride_qd,
                mask=row_valid[:, None], other=0.0)
    qk_scale = sm_scale * 1.44269504
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    # prefix: dense attention over [0, S_b)
    for start_n in range(0, S_b, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kn = kv_start + offs_n
        n_valid = offs_n < S_b
        k = tl.load(K + kn[:, None] * stride_km + kv_h * stride_kh + offs_d[None, :] * stride_kd,
                    mask=n_valid[:, None], other=0.0)
        v = tl.load(V + kn[:, None] * stride_vm + kv_h * stride_vh + offs_d[None, :] * stride_vd,
                    mask=n_valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)).to(tl.float32)
        qk = tl.where(n_valid[None, :], qk * qk_scale, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v).to(tl.float32)
        m_i = m_ij

    # pack: one tile, ancestor-masked
    offs_p = tl.arange(0, BLOCK_N)
    kn = kv_start + S_b + offs_p
    pack_valid = offs_p < P_b
    bits = tl.load(AncestorMask + offs_m, mask=row_valid, other=0)
    allowed = ((bits[:, None] >> offs_p[None, :]) & 1) == 1
    allowed = allowed & pack_valid[None, :] & row_valid[:, None]
    k = tl.load(K + kn[:, None] * stride_km + kv_h * stride_kh + offs_d[None, :] * stride_kd,
                mask=pack_valid[:, None], other=0.0)
    v = tl.load(V + kn[:, None] * stride_vm + kv_h * stride_vh + offs_d[None, :] * stride_vd,
                mask=pack_valid[:, None], other=0.0)
    qk = tl.dot(q, tl.trans(k)).to(tl.float32)
    qk = tl.where(allowed, qk * qk_scale, -float("inf"))
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
    p = tl.math.exp2(qk - m_ij[:, None])
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v).to(tl.float32)

    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]
    tl.store(Out + offs_m[:, None] * stride_om + pid_h * stride_oh + offs_d[None, :] * stride_od,
             acc.to(Out.dtype.element_ty), mask=row_valid[:, None])


def _configs():
    # BLOCK_N >= 64 so a full pack fits in one tile; few stages to stay within smem
    return [triton.Config({"BLOCK_N": bn}, num_stages=s, num_warps=w)
            for bn in (64, 128) for s in (1, 2) for w in (2, 4, 8)]


_tree_attn = triton.autotune(configs=_configs(), key=["H_q", "D"])(_tree_attn_fwd)


def _tile_schedule(cu_q, device):
    req, row0, nrows = [], [], []
    cu = cu_q.tolist()
    for b in range(len(cu) - 1):
        for r in range(cu[b], cu[b + 1], BLOCK_M):
            req.append(b)
            row0.append(r)
            nrows.append(min(BLOCK_M, cu[b + 1] - r))
    as_t = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    return as_t(req), as_t(row0), as_t(nrows)


def varlen_tree_attention(q, k, v, cu_q, cu_kv, prefix_len, pack_len, ancestor_mask, sm_scale=None):
    """q: [total_q, H_q, D], k/v: [total_kv, H_kv, D] -> [total_q, H_q, D].

    cu_q / cu_kv: int32 [B+1] offsets of packs and of concat(prefix, pack).
    prefix_len / pack_len: int32 [B]. ancestor_mask: int64 [total_q].
    """
    total_q, H_q, D = q.shape
    H_kv = k.shape[1]
    assert H_q % H_kv == 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    tile_req, tile_row0, tile_nrows = _tile_schedule(cu_q.detach().cpu(), q.device)
    _tree_attn[(tile_req.numel(), H_q)](
        q, k, v, out, ancestor_mask.to(torch.int64), tile_req, tile_row0, tile_nrows,
        cu_kv.to(torch.int32), prefix_len.to(torch.int32), pack_len.to(torch.int32),
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H_q=H_q, GROUP=H_q // H_kv, D=D, BLOCK_M=BLOCK_M,
    )
    return out
