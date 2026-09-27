import numpy as np
import pytest

from cast_trees import (GSM8K_MASSES, deploy_width, fit_cost, mass_curve, predict_width, prefix_masses,
                        stopping_width)


def test_fit_cost_recovers_a_line():
    n = [1, 8, 16, 32, 64, 128, 256]
    c0, c1 = fit_cost(n, [10.0 + 0.05 * x for x in n])
    assert (c0, c1) == pytest.approx((10.0, 0.05))


def test_stopping_rule_on_a_hand_checked_case():
    # g(1)+1 = 1.5, l(1) = 1 + 2 * 0.25 = 1.5, threshold = 0.25, rho_(2) = 0.3 passes
    # g(2)+1 = 1.8, l(2) = 1.75, threshold = 0.257, rho_(3) = 0.2 fails
    n_stop, trace = stopping_width([0.5, 0.3, 0.2, 0.1], c_draft=0.0, c0=1.0, c1=0.25)
    assert n_stop == 2
    assert trace[1][2] == pytest.approx(1.8 / 1.75 * 0.25)


def test_flat_cost_never_stops():
    n_stop, _ = stopping_width(GSM8K_MASSES, 3.0, 30.0, 0.0)
    assert n_stop == len(GSM8K_MASSES)


def test_deploy_width_rounds_up_to_the_grid():
    assert deploy_width(47) == 47
    assert deploy_width(48) == 63
    assert deploy_width(200) == 127
    assert deploy_width(80, grid=(15, 31, 47, 63, 95)) == 95


def test_packaged_curve_matches_the_paper():
    # Appendix G: masses fall from 0.86 at the first candidate to 0.016 at the 48th and 0.011 at the 64th
    assert round(GSM8K_MASSES[0], 2) == 0.86
    assert round(GSM8K_MASSES[47], 3) == 0.016
    assert round(GSM8K_MASSES[63], 3) == 0.011
    assert all(a >= b for a, b in zip(GSM8K_MASSES, GSM8K_MASSES[1:]))


def test_h100_sxm_probe_gives_the_widest_grid_point():
    w = predict_width([16, 24, 32, 48, 64, 96, 128], [30.6, 30.2, 30.6, 31.2, 31.5, 32.2, 32.6], draft_ms=3.8)
    assert w.n_star == 127


def test_prefix_masses_are_best_first():
    lp = np.log(np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]))
    assert prefix_masses(lp, n_max=4, k=3) == pytest.approx([0.6, 0.3, 0.3, 0.24])


def test_numpy_mass_curve_matches_the_replay_module():
    torch = pytest.importorskip("torch")
    from cast_trees import replay
    g = torch.Generator().manual_seed(0)
    rounds = [{"top_idx": torch.arange(15 * 32).reshape(15, 32), "top_val": torch.randn(15, 32, generator=g) * 3,
               "realized": torch.zeros(15, dtype=torch.long), "accepted": 0} for _ in range(20)]
    ref = replay.mass_curve(rounds, 64)
    lps = [torch.log_softmax(r["top_val"].float(), -1).numpy() for r in rounds]
    assert np.allclose(mass_curve(lps, 64), ref, rtol=1e-5)
