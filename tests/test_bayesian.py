"""Tests for the Bayesian inference module.

The MH backend is exercised on a synthetic curve generated from a known
``(Tc, beta)`` pair, with a short chain that still recovers the truth within
a coarse tolerance.
"""
from __future__ import annotations

import numpy as np

from src.bayesian import (
    _log_prior,
    _log_likelihood,
    sample_mh_numpy,
)


def test_log_prior_finite_at_typical_values():
    lp = _log_prior(Tc=2.27, beta=0.125, sigma=0.05)
    assert np.isfinite(lp)


def test_log_prior_minus_infinity_outside_support():
    assert _log_prior(Tc=0.5, beta=0.125, sigma=0.05) == -np.inf
    assert _log_prior(Tc=2.27, beta=1.0, sigma=0.05) == -np.inf
    assert _log_prior(Tc=2.27, beta=0.125, sigma=-0.01) == -np.inf


def test_log_likelihood_peaks_near_truth():
    """Likelihood at the synthetic truth must beat distant parameter choices."""
    rng = np.random.default_rng(0)
    Tc_true, beta_true = 2.27, 0.125
    T = np.linspace(1.5, 3.5, 25)
    M = np.where(T < Tc_true, np.clip(1.0 - T / Tc_true, 1e-10, 1.0) ** beta_true, 0.0)
    M_obs = M + rng.normal(0, 0.02, M.shape)
    ll_truth = _log_likelihood(Tc_true, beta_true, 0.02, T, M_obs)
    ll_far = _log_likelihood(Tc_true + 0.5, beta_true + 0.2, 0.02, T, M_obs)
    assert ll_truth > ll_far


def test_mh_recovers_synthetic_truth():
    """A short MH run on a synthetic curve recovers Tc within a coarse tolerance."""
    rng = np.random.default_rng(0)
    Tc_true, beta_true = 2.27, 0.125
    T = np.linspace(1.5, 3.5, 25)
    M = np.where(T < Tc_true, np.clip(1.0 - T / Tc_true, 1e-10, 1.0) ** beta_true, 0.0)
    M_obs = M + rng.normal(0, 0.02, M.shape)
    samples = sample_mh_numpy(T, M_obs, draws=500, tune=300, chains=2, seed=0)
    Tc_post = samples["Tc"].mean()
    assert abs(Tc_post - Tc_true) < 0.10
