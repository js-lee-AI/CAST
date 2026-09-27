"""Best-first candidate trees over one-pass block scores (numpy only).

The functions index plain arrays, so they take numpy arrays or CPU torch
tensors alike. The torch side of verification lives in verify.py.
"""
import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class TreeNode:
    token: int
    depth: int      # offset from the pending root, 1..L
    parent: int     # index into the node list, -1 = root
    score: float    # path log-probability under the drafter marginals


def block_topk(logits, k=8):
    """Top-k token ids and log-probs at each of the L future positions."""
    x = np.asarray(logits, float)
    lp = x - x.max(axis=-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(axis=-1, keepdims=True))
    idx = np.argsort(-lp, axis=-1, kind="stable")[:, :k]
    return idx, np.take_along_axis(lp, idx, axis=-1)


def build_tree(top_idx, top_lp, budget, k=8):
    """Best-first prefix-closed tree with `budget` nonroot nodes.

    top_idx / top_lp: (L, >=k) per-position candidates. Nodes are returned in
    pop order, so a parent always comes before its children.
    """
    L = top_idx.shape[0]
    k = min(k, top_idx.shape[1])
    heap, nodes, cnt = [], [], 0
    for r in range(k):
        heapq.heappush(heap, (-float(top_lp[0, r]), 1, -1, r, cnt))
        cnt += 1
    while heap and len(nodes) < budget:
        neg, d, parent, r, _ = heapq.heappop(heap)
        me = len(nodes)
        nodes.append(TreeNode(int(top_idx[d - 1, r]), d, parent, -neg))
        if d < L:
            for r2 in range(k):
                heapq.heappush(heap, (neg - float(top_lp[d, r2]), d + 1, me, r2, cnt))
                cnt += 1
    return nodes


def build_chain(top_idx, top_lp, length):
    """Rank-0 path of the given length, i.e. the standard DFlash block."""
    return [TreeNode(int(top_idx[d - 1, 0]), d, d - 2, float(top_lp[:d, 0].sum()))
            for d in range(1, length + 1)]


def expected_accepted_length(nodes):
    """Expected accepted draft tokens under the drafter's own scores, sum of path masses."""
    return float(sum(np.exp(nd.score) for nd in nodes))


def select_global_topk(trees, total):
    """Split a shared budget across requests by global best-first score.

    A child never scores above its parent, so taking the top `total` nodes of
    the pooled forest keeps every tree prefix-closed.
    """
    pool = [(nd.score, b, j) for b, nodes in enumerate(trees) for j, nd in enumerate(nodes)]
    pool.sort(key=lambda t: (-t[0], t[1], t[2]))
    keep = [set() for _ in trees]
    for _, b, j in pool[:total]:
        keep[b].add(j)
    out = []
    for b, nodes in enumerate(trees):
        kept = sorted(keep[b])
        remap = {-1: -1}
        for new, old in enumerate(kept):
            remap[old] = new
        out.append([TreeNode(nodes[i].token, nodes[i].depth, remap.get(nodes[i].parent, -1),
                             nodes[i].score) for i in kept])
    return out


def ancestor_matrix(nodes):
    """(1+N, 1+N) bool, row i marks the ancestors of packed token i (and i)."""
    n = 1 + len(nodes)
    vis = np.zeros((n, n), dtype=bool)
    vis[0, 0] = True
    for i, nd in enumerate(nodes):
        vis[i + 1] = vis[nd.parent + 1]
        vis[i + 1, i + 1] = True
    return vis
