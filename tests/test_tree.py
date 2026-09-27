import itertools

import numpy as np
import pytest

from cast_trees import (ancestor_matrix, block_topk, build_chain, build_tree, expected_accepted_length,
                        select_global_topk)


def random_marginals(L=15, V=100, k=8, seed=0):
    return block_topk(np.random.default_rng(seed).normal(size=(L, V)) * 3, k)


def test_tree_is_prefix_closed_top_n():
    idx, lp = random_marginals()
    nodes = build_tree(idx, lp, budget=63, k=8)
    assert len(nodes) == 63
    for i, nd in enumerate(nodes):
        assert nd.parent < i
        if nd.parent >= 0:
            assert nodes[nd.parent].depth == nd.depth - 1
            assert nodes[nd.parent].score >= nd.score
    # brute force over the 3-level universe: best-first gives the top-N scores
    small = build_tree(idx[:3], lp[:3], budget=20, k=4)
    scores = []
    for d in range(1, 4):
        for ranks in itertools.product(range(4), repeat=d):
            scores.append(sum(float(lp[j, r]) for j, r in enumerate(ranks)))
    top = sorted(scores, reverse=True)[:20]
    assert np.allclose(sorted(n.score for n in small), sorted(top), atol=1e-9)


def test_tree_beats_chain_at_the_same_budget():
    # Theorem 1: the top-N tree maximizes the expected accepted length for its size
    idx, lp = random_marginals()
    chain = build_chain(idx, lp, 15)
    tree = build_tree(idx, lp, 15)
    assert expected_accepted_length(tree) >= expected_accepted_length(chain)


def test_expected_length_of_a_certain_chain():
    lp = np.log(np.full((4, 2), [1.0, 1e-300]))
    idx = np.zeros((4, 2), dtype=int)
    assert expected_accepted_length(build_chain(idx, lp, 4)) == pytest.approx(4.0)


def test_global_allocation_keeps_trees_closed():
    trees = [build_tree(*random_marginals(seed=s), budget=48) for s in range(8)]
    split = select_global_topk(trees, 8 * 8)
    assert sum(len(t) for t in split) == 64
    for t in split:
        for i, nd in enumerate(t):
            assert -1 <= nd.parent < i
            if nd.parent >= 0:
                assert t[nd.parent].depth == nd.depth - 1


def test_ancestor_matrix():
    idx, lp = random_marginals()
    nodes = build_tree(idx, lp, budget=31)
    vis = ancestor_matrix(nodes)
    for i, nd in enumerate(nodes):
        chain, j = {0, i + 1}, nd.parent
        while j >= 0:
            chain.add(j + 1)
            j = nodes[j].parent
        assert set(vis[i + 1].nonzero()[0].tolist()) == chain


def test_numpy_topk_matches_the_torch_path():
    torch = pytest.importorskip("torch")
    from cast_trees.verify import topk_marginals
    logits = np.random.default_rng(1).normal(size=(15, 200)) * 3
    idx_np, lp_np = block_topk(logits, 8)
    idx_t, lp_t = topk_marginals(torch.tensor(logits), 8)
    assert np.array_equal(idx_np, idx_t.numpy())
    assert np.allclose(lp_np, lp_t.double().numpy(), atol=1e-5)
    # the decoder hands CPU tensors to build_tree, the library hands numpy arrays
    a = build_tree(idx_t, lp_t, 63)
    b = build_tree(idx_t.numpy(), lp_t.numpy(), 63)
    assert [(n.token, n.depth, n.parent) for n in a] == [(n.token, n.depth, n.parent) for n in b]
