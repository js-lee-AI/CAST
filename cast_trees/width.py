"""Marginal stopping rule for the tree width (numpy only).

Round time is modeled as l(N) = c_draft + c0 + c1 * (N + 1), where c1 is the
least-squares slope of the batch-one target forward over 16 <= n <= 128 packed
tokens. With g(N) the sum of the N largest prefix masses rho_(1..N), node N+1
is added while

    rho_(N+1) >= (g(N) + 1) / l(N) * c1.

The deployed width is the first grid point at or above the stopping point.
"""
import heapq
from dataclasses import dataclass, field

import numpy as np

from ._masses import GSM8K_MASSES

GRID = (15, 31, 47, 63, 95, 127)


def fit_cost(n, ms, lo=16, hi=128):
    """Least-squares intercept and slope (ms, ms per packed token) over lo <= n <= hi."""
    n, ms = np.asarray(n, float), np.asarray(ms, float)
    sel = (n >= lo) & (n <= hi)
    c1, c0 = np.polyfit(n[sel], ms[sel], 1)
    return float(c0), float(c1)


def stopping_width(rho, c_draft, c0, c1, n_max=None):
    """Largest N the marginal rule accepts, and the (N, rho_(N+1), threshold) trace."""
    rho = np.asarray(rho, float)
    n_max = len(rho) if n_max is None else min(n_max, len(rho))
    g, trace = 0.0, []
    for N in range(1, n_max):
        g += rho[N - 1]
        ell = c_draft + c0 + c1 * (N + 1)
        thr = (g + 1.0) / ell * c1
        trace.append((N, float(rho[N]), float(thr)))
        if rho[N] < thr:
            return N, trace
    return n_max, trace


def deploy_width(n_stop, grid=GRID):
    """First grid width at or above the stopping point (the largest one if none)."""
    for w in sorted(grid):
        if w >= n_stop:
            return w
    return max(grid)


@dataclass
class WidthPrediction:
    c0: float
    c1: float
    n_stop: int
    n_star: int
    crossed: bool
    trace: list = field(repr=False, default_factory=list)

    def __str__(self):
        where = f"stops at N={self.n_stop}" if self.crossed else "does not stop inside the mass curve"
        return f"c1 = {self.c1:.4f} ms per packed token, the rule {where}, N* = {self.n_star}"


def predict_width(n, ms, rho=None, draft_ms=0.0, grid=GRID, lo=16, hi=128):
    """Deployed width from a latency probe (packed tokens n, target-forward ms).

    rho defaults to the GSM8K prefix-mass curve of the paper, which it uses for
    every deployment. draft_ms is the drafter forward, 0 if it was not measured.
    """
    c0, c1 = fit_cost(n, ms, lo, hi)
    rho = GSM8K_MASSES if rho is None else rho
    n_stop, trace = stopping_width(rho, draft_ms, c0, c1)
    return WidthPrediction(c0, c1, n_stop, deploy_width(n_stop, grid), n_stop < len(rho), trace)


def prefix_masses(top_lp, n_max=128, k=8):
    """Path masses of one round in best-first order, from (L, >=k) log-probs."""
    top_lp = np.asarray(top_lp, float)
    m, k = top_lp.shape[0], min(k, top_lp.shape[1])
    heap = [(-top_lp[0, r], 1) for r in range(k)]
    heapq.heapify(heap)
    vals = []
    while heap and len(vals) < n_max:
        neg, d = heapq.heappop(heap)
        vals.append(float(np.exp(-neg)))
        if d < m:
            for r in range(k):
                heapq.heappush(heap, (neg - top_lp[d, r], d + 1))
    return vals


def mass_curve(rounds_lp, n_max=128, k=8, n_rounds=1500):
    """rho_(N) for N = 1..n_max, averaged over logged rounds of (L, K) log-probs."""
    acc = [[] for _ in range(n_max)]
    for lp in list(rounds_lp)[:n_rounds]:
        for i, v in enumerate(prefix_masses(lp, n_max, k)):
            acc[i].append(v)
    return [float(np.mean(a)) if a else 0.0 for a in acc]
