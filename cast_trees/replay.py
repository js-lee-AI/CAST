"""Offline replay of greedy round logs (see scripts/collect_rounds.py).

At T=0 the target continuation is fixed, so the accepted length of any tree is
the longest realized prefix that is a path of the tree. Trees are sets of rank
tuples, (r1, r2, ...) meaning rank r1 at depth 1, r2 at depth 2 and so on.
"""
import heapq
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def log_probs(lg):
    m = lg["top_idx"].shape[0]
    return F.log_softmax(lg["top_val"][:m].float(), dim=-1)


def accepted_len(tree, top_idx, realized):
    """Committed round length R = accepted nodes + 1."""
    m = min(top_idx.shape[0], len(realized))
    tau, path = 0, ()
    for d in range(m):
        hit = (top_idx[d] == realized[d]).nonzero(as_tuple=True)[0]
        if len(hit) == 0:
            break
        nxt = path + (int(hit[0]),)
        if nxt not in tree:
            break
        tau, path = tau + 1, nxt
    return tau + 1


def best_first(lp, budget, k=8):
    """Top-`budget` prefix-closed tree by path log-prob with rank cap k."""
    k = min(k, lp.shape[1])
    heap = [(-float(lp[0, r]), 1, (r,)) for r in range(k)]
    heapq.heapify(heap)
    tree = set()
    while heap and len(tree) < budget:
        neg, d, path = heapq.heappop(heap)
        tree.add(path)
        if d < lp.shape[0]:
            for r in range(k):
                heapq.heappush(heap, (neg - float(lp[d, r]), d + 1, path + (r,)))
    return tree


def chain(m):
    return {tuple([0] * (d + 1)) for d in range(m)}


def schedule_tree(widths, budget, depth_max):
    """Fixed branching widths by depth, then a rank-0 chain for the rest."""
    tree, frontier, d = set(), [()], 0
    while frontier and len(tree) < budget and d < depth_max:
        w = widths[d] if d < len(widths) else 1
        nxt = []
        for p in frontier:
            for r in range(w):
                if len(tree) >= budget:
                    break
                tree.add(p + (r,))
                nxt.append(p + (r,))
        frontier, d = nxt, d + 1
    p = tuple([0] * d)
    while len(tree) < budget and d < depth_max:
        p = p[:d] + (0,)
        tree.add(p)
        d += 1
    return tree


def usable(lg):
    return lg["top_idx"].shape[0] > 0 and len(lg["realized"]) > 0


def mass_curve(rounds, n_max=128, k=8, n_rounds=1500):
    """rho_(N): mean N-th largest path mass of the best-first tree, N = 1..n_max."""
    acc = [[] for _ in range(n_max)]
    for lg in rounds[:n_rounds]:
        lp = log_probs(lg)
        m = lp.shape[0]
        heap = [(-float(lp[0, r]), 1) for r in range(k)]
        heapq.heapify(heap)
        vals = []
        while heap and len(vals) < n_max:
            neg, d = heapq.heappop(heap)
            vals.append(float(np.exp(-neg)))
            if d < m:
                for r in range(k):
                    heapq.heappush(heap, (neg - float(lp[d, r]), d + 1))
        for i, v in enumerate(vals):
            acc[i].append(v)
    return [float(np.mean(a)) if a else 0.0 for a in acc]


def rejection_ranks(rounds):
    """Rank of the target correction at the chain's first rejected position."""
    K = rounds[0]["top_idx"].shape[1]
    ranks = []
    for lg in rounds:
        j, cont = lg["accepted"], lg["realized"]
        if j < lg["top_idx"].shape[0] and j < len(cont):
            hit = (lg["top_idx"][j] == cont[j]).nonzero(as_tuple=True)[0]
            ranks.append(int(hit[0]) if len(hit) else K)
    c, n = Counter(ranks), max(len(ranks), 1)
    frac = lambda lo, hi: sum(c.get(r, 0) for r in range(lo, hi)) / n
    return {"rank1": frac(0, 1), "rank2_4": frac(1, 4), "rank5_8": frac(4, 8),
            "rank9_16": frac(8, 16), "rank17_32": frac(16, 32), f"beyond{K}": c.get(K, 0) / n,
            "n": len(ranks)}


def reliability(rounds, depth=1, k=8, min_count=50):
    """Binned drafter probability against the empirical hit rate at one depth."""
    s, c, h = np.zeros(10), np.zeros(10), np.zeros(10)
    d = depth - 1
    for lg in rounds:
        cont = lg["realized"]
        if min(len(cont), lg["top_idx"].shape[0]) <= d:
            continue
        pr = F.softmax(lg["top_val"][d].float(), dim=-1)
        for r in range(k):
            p = float(pr[r])
            b = min(int(p * 10), 9)
            s[b] += p
            c[b] += 1
            h[b] += int(lg["top_idx"][d, r] == cont[d])
    return [{"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "mean_p": s[b] / c[b],
             "hit_rate": h[b] / c[b], "n": int(c[b])} for b in range(10) if c[b] >= min_count]


def shape_ablation(rounds, budget=47, k=8):
    """Chain, two fixed schedules and the top-N tree at a matched budget."""
    res = {"chain": [], "schedule_8_4_2_2": [], "binary": [], "top_n": []}
    for lg in rounds:
        if not usable(lg):
            continue
        ti, cont = lg["top_idx"], lg["realized"]
        m = ti.shape[0]
        res["chain"].append(accepted_len(chain(min(budget, m)), ti, cont))
        res["schedule_8_4_2_2"].append(accepted_len(schedule_tree([8, 4, 2, 2], budget, m), ti, cont))
        res["binary"].append(accepted_len(schedule_tree([2] * 5, budget, m), ti, cont))
        res["top_n"].append(accepted_len(best_first(log_probs(lg), budget, k), ti, cont))
    return {key: float(np.mean(v)) for key, v in res.items()}


def rank_cap(rounds, budget, k=8):
    """R of the rank-capped tree against the tree over every logged rank."""
    capped, full = [], []
    for lg in rounds:
        if not usable(lg):
            continue
        lp = log_probs(lg)
        capped.append(accepted_len(best_first(lp, budget, k), lg["top_idx"], lg["realized"]))
        full.append(accepted_len(best_first(lp, budget, lp.shape[1]), lg["top_idx"], lg["realized"]))
    a, b = float(np.mean(capped)), float(np.mean(full))
    return {"R_cap": a, "R_full": b, "gap_pct": 100.0 * (b - a) / b, "rounds": len(capped)}


def acceptance_curve(lg, w_max, k=8):
    """Node priorities and R after each best-first node, root included.

    Returns (prio, acc): prio[i] is the path log-prob of the i-th added node
    (0.0 for the root) and acc[w] the committed length with the first w nodes.
    """
    m = min(lg["top_idx"].shape[0], len(lg["realized"]))
    if m == 0:
        return [0.0], [0, 1]
    lp = F.log_softmax(lg["top_val"][:m].float(), dim=-1)
    heap, tree, prio, acc = [(0.0, 0, ())], set(), [], [0]
    while heap and len(tree) < w_max:
        neg, d, path = heapq.heappop(heap)
        tree.add(path)
        prio.append(-neg)
        acc.append(accepted_len(tree, lg["top_idx"], lg["realized"]) if path else 1)
        if d < m:
            for r in range(min(k, lp.shape[1])):
                heapq.heappush(heap, (neg - float(lp[d, r]), d + 1, path + (r,)))
    return prio, acc


def load_rounds(path):
    return torch.load(path, weights_only=True)
