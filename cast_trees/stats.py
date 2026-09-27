"""Paired prompt-level bootstrap of the gain over DFlash (standard library only)."""
import random
import statistics


def paired_bootstrap(chain_ms, cast_ms, n_boot=10000, seed=12345):
    """Mean gain and 95% interval of mean(DFlash ms / CAST ms) - 1 over prompts.

    Prompts are resampled with replacement, with the same indices for both methods.
    """
    ratio = [c / k for c, k in zip(chain_ms, cast_ms)]
    rng = random.Random(seed)
    idx = list(range(len(ratio)))
    reps = sorted(statistics.mean(ratio[rng.choice(idx)] for _ in idx) - 1.0 for _ in range(n_boot))
    return statistics.mean(ratio) - 1.0, reps[int(0.025 * n_boot)], reps[int(0.975 * n_boot)]
