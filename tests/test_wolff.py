"""Tests for the Wolff cluster algorithm and the autocorrelation-time estimator."""
from __future__ import annotations

import itertools

import numpy as np

from src.autocorr import n_blocks_for, tau_int
from src.fss import _chain, measure
from src.metropolis import _seed
from src.wolff import wolff_chain

TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


def test_tau_int_of_ar1_process():
    """AR(1) with coefficient ρ has τ_int = (1 + ρ) / (2 (1 − ρ)) = 9.5 for ρ = 0.9."""
    rng = np.random.default_rng(0)
    rho, n = 0.9, 400_000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    r = tau_int(x)
    assert abs(r["tau"] - 9.5) < 3 * r["err"] + 0.3


def test_tau_int_of_white_noise_is_one_half():
    r = tau_int(np.random.default_rng(1).normal(size=100_000))
    assert abs(r["tau"] - 0.5) < 0.05


def test_blocks_span_twenty_autocorrelation_times():
    assert n_blocks_for(200_000, 100.0) == 50              # capped
    assert n_blocks_for(200_000, 1000.0) == 10             # 20 000-sweep blocks
    assert n_blocks_for(1000, 1000.0) == 2                 # never below 2


def _exact_4x4(T: float) -> tuple[float, float]:
    """<e> and <|m|> per site on the 4 x 4 periodic lattice by enumerating 2^16 states."""
    states = np.array(list(itertools.product([-1, 1], repeat=16)), dtype=float).reshape(-1, 4, 4)
    e = -(states * np.roll(states, 1, axis=1) + states * np.roll(states, 1, axis=2)).sum(axis=(1, 2)) / 16
    m = np.abs(states.mean(axis=(1, 2)))
    w = np.exp(-16 * (e - e.min()) / T)
    return float((e * w).sum() / w.sum()), float((m * w).sum() / w.sum())


def test_wolff_matches_exact_enumeration_on_4x4():
    T = 2.5
    e_exact, m_exact = _exact_4x4(T)
    _seed(0)
    m, e, _ = wolff_chain(np.ones((4, 4), dtype=np.int8), T, 1000, 100_000)
    assert abs(e.mean() - e_exact) < 0.01
    assert abs(np.abs(m).mean() - m_exact) < 0.01


def test_wolff_decorrelates_much_faster_than_metropolis_at_tc():
    """At Tc on L = 16, τ_int of |m| in sweeps is several times smaller for Wolff."""
    _seed(0)
    m_met, _ = _chain(np.ones((16, 16), dtype=np.int8), TC, 2000, 100_000)
    _seed(0)
    m_wol, _, sweeps = wolff_chain(np.ones((16, 16), dtype=np.int8), TC, 200, 20_000)
    assert tau_int(np.abs(m_wol))["tau"] * sweeps < 0.3 * tau_int(np.abs(m_met))["tau"]


def test_measure_with_wolff_reports_autocorrelation_and_blocks():
    r = measure(8, 1.5, n_therm=200, n_measure=2000, seed=0, algorithm="wolff")
    assert 0.9 < r["M_abs"] <= 1.0
    assert r["tau_m"] >= 0.5 and r["block_over_tau"] >= 20 or r["n_blocks"] == 2
