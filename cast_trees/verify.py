"""Torch side of tree verification: drafter top-k, tree attention mask, KV gather."""
import torch
import torch.nn.functional as F

from .tree import ancestor_matrix


def topk_marginals(d_logits, k):
    """Top-k token ids and log-probs at each future position, moved to CPU."""
    lp = F.log_softmax(d_logits.float(), dim=-1)
    top = torch.topk(lp, k, dim=-1)
    return top.indices.cpu(), top.values.cpu()


def tree_attention_mask(n_prefix, nodes, device, dtype):
    """Additive mask for the packed pass [root] + nodes.

    Every packed token sees the cached prefix and, inside the pack, only its
    own ancestor chain. Built on the host and copied once.
    """
    vis = ancestor_matrix(nodes)
    n = vis.shape[0]
    mask = torch.zeros((1, 1, n, n_prefix + n), device=device, dtype=dtype)
    mask[0, 0, :, n_prefix:] = torch.from_numpy(~vis).to(device) * torch.finfo(dtype).min
    return mask


def cache_layers(cache):
    """(n_layers, get, set) over a DynamicCache, for old and new layouts."""
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        layers = cache.layers

        def get(i):
            return layers[i].keys, layers[i].values

        def put(i, k, v):
            layers[i].keys, layers[i].values = k, v
        return len(layers), get, put

    def get(i):
        return cache.key_cache[i], cache.value_cache[i]

    def put(i, k, v):
        cache.key_cache[i], cache.value_cache[i] = k, v
    return len(cache.key_cache), get, put


def gather_cache(cache, n_prefix, keep):
    """Keep the prefix and the packed entries listed in `keep`, in order."""
    n_layers, get, put = cache_layers(cache)
    dev = get(0)[0].device
    sel = torch.arange(n_prefix, device=dev)
    if keep:
        sel = torch.cat([sel, torch.tensor([n_prefix + i for i in keep], device=dev)])
    for i in range(n_layers):
        k, v = get(i)
        put(i, k.index_select(2, sel), v.index_select(2, sel))
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = sel.numel()
    return cache
