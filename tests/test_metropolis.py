"""Physics-driven tests for the Metropolis sampler.

The lattice is kept small (N=8) and the chain length short so the suite runs
in seconds even on the first call, when numba is compiling the inner sweep.
"""
from __future__ import annotations

import numpy as np

from src.metropolis import simulate_at_temperature


def test_low_temperature_aligns_spins():
    """At T well below Tc the lattice should saturate to |M| close to 1."""
    r = simulate_at_temperature(T=0.5, size=8, n_therm=400, n_measure=400, seed=0)
    assert r["M_mean"] > 0.95


def test_high_temperature_disorders_spins():
    """At T well above Tc, fluctuations should keep |M| small."""
    r = simulate_at_temperature(T=5.0, size=8, n_therm=400, n_measure=400, seed=0)
    assert r["M_mean"] < 0.30


def test_returns_consistent_dict_keys():
    r = simulate_at_temperature(T=2.0, size=4, n_therm=100, n_measure=100, seed=0)
    assert set(r.keys()) == {"T", "M_mean", "M_std"}
    assert r["T"] == 2.0
    assert 0.0 <= r["M_mean"] <= 1.0
    assert r["M_std"] >= 0.0
