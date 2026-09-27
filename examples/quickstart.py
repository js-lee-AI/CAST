"""CAST on a measured cost curve and toy block scores. CPU only, no downloads."""
import numpy as np

import cast_trees

# Target-forward latency (ms) by packed tokens, H100 SXM, Qwen3-8B (paper Table 5)
n, ms = [16, 24, 32, 48, 64, 96, 128], [30.6, 30.2, 30.6, 31.2, 31.5, 32.2, 32.6]
width = cast_trees.predict_width(n, ms)
print(width)

# Toy one-pass block scores, 15 future positions over a 64-token vocabulary
top_idx, top_lp = cast_trees.block_topk(np.random.default_rng(0).normal(scale=5, size=(15, 64)))
for label, nodes in [("DFlash chain", cast_trees.build_chain(top_idx, top_lp, 15)),
                     ("CAST, N=15", cast_trees.build_tree(top_idx, top_lp, 15)),
                     (f"CAST, N={width.n_star}", cast_trees.build_tree(top_idx, top_lp, width.n_star))]:
    print(f"{label:13s} {len(nodes):3d} nodes, expected accepted tokens "
          f"{cast_trees.expected_accepted_length(nodes):.2f}")
