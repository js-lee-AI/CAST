"""CAST: cost-aware speculative trees from one-pass block drafters.

    import cast_trees
    width = cast_trees.predict_width(n, ms)              # N* from a latency probe
    nodes = cast_trees.build_tree(top_idx, top_lp, width.n_star)

The core needs neither a GPU nor torch. Torch is imported only when a
model-facing name is first used.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ._masses import GSM8K_MASSES
from .stats import paired_bootstrap
from .tree import (TreeNode, ancestor_matrix, block_topk, build_chain, build_tree,
                   expected_accepted_length, select_global_topk)
from .width import (GRID, WidthPrediction, deploy_width, fit_cost, mass_curve, predict_width,
                    prefix_masses, stopping_width)

# Heavy names, imported on first attribute access (PEP 562).
_LAZY = {
    "cast_generate": "decode",
    "ar_generate": "decode",
    "walk_tree": "decode",
    "spec_sample_node": "decode",
    "topk_marginals": "verify",
    "tree_attention_mask": "verify",
    "gather_cache": "verify",
    "BatchedServer": "serving",
    "POLICIES": "serving",
    "varlen_tree_attention": "tree_attn",
    "load_models": "utils",
    "load_prompts": "data",
}

__all__ = [
    "__version__", "GRID", "GSM8K_MASSES", "TreeNode", "WidthPrediction",
    "ancestor_matrix", "block_topk", "build_chain", "build_tree", "deploy_width",
    "expected_accepted_length", "fit_cost", "mass_curve", "paired_bootstrap",
    "predict_width", "prefix_masses", "select_global_topk", "stopping_width", *_LAZY,
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
