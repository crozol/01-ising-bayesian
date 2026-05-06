"""Physics-driven tests for the Metropolis sampler.

Tests start from cold (fully aligned) configurations so the assertions only
rely on qualitative thermodynamic behaviour and do not depend on the random
seed or on whether numba is available on the host (numba uses its own RNG,
independent from ``numpy.random``).
"""
from __future__ import annotations

import numpy as np

from src.metropolis import _sweep, simulate_at_temperature


def test_cold_start_stays_ordered_at_low_T():
    """A fully aligned lattice should remain aligned at T well below Tc."""
    lattice = np.ones((8, 8), dtype=np.int8)
    for _ in range(200):
        _sweep(lattice, 0.5, 1.0)
    assert abs(lattice.mean()) > 0.9


def test_cold_start_disorders_at_high_T():
    """A fully aligned lattice should disorder at T well above Tc."""
    lattice = np.ones((16, 16), dtype=np.int8)
    for _ in range(3000):
        _sweep(lattice, 5.0, 1.0)
    assert abs(lattice.mean()) < 0.30


def test_simulate_returns_consistent_dict_keys():
    r = simulate_at_temperature(T=2.0, size=4, n_therm=100, n_measure=100, seed=0)
    assert set(r.keys()) == {"T", "M_mean", "M_std"}
    assert r["T"] == 2.0
    assert 0.0 <= r["M_mean"] <= 1.0
    assert r["M_std"] >= 0.0
