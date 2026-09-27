"""Monte-Carlo check that recursive speculative sampling keeps the target law."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("dflash")


def test_node_sampler_matches_target(V=40, n_draws=200000, seed=0):
    from cast_trees.decode import spec_sample_node
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(V, generator=g, dtype=torch.float64) * 2, 0)
    q = torch.softmax(torch.randn(V, generator=g, dtype=torch.float64) * 2, 0)
    kids = torch.topk(q, 6).indices                       # the proposal's favorites
    counts = torch.zeros(V, dtype=torch.float64)
    for _ in range(n_draws):
        tok, _ = spec_sample_node(p, kids, q, generator=g)
        counts[tok] += 1
    tv = 0.5 * (counts / n_draws - p).abs().sum().item()
    # expected TV of an exact sampler at this draw count is about 0.01
    assert tv < 0.02, tv
