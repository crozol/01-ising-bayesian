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


def test_flat_beta_prior_is_flat_inside_and_closed_outside():
    a = _log_prior(Tc=2.3, beta=0.05, sigma=0.05, beta_prior="flat")
    b = _log_prior(Tc=2.3, beta=0.40, sigma=0.05, beta_prior="flat")
    assert np.isfinite(a) and a == b
    assert _log_prior(Tc=2.3, beta=0.55, sigma=0.05, beta_prior="flat") == -np.inf


def test_heteroscedastic_likelihood_reduces_to_homoscedastic_with_zero_errors():
    T = np.linspace(1.5, 3.5, 25)
    M = np.where(T < 2.3, (1 - T / 2.3).clip(1e-10) ** 0.12, 0.0) + 0.01
    homo = _log_likelihood(2.3, 0.12, 0.03, T, M)
    hetero = _log_likelihood(2.3, 0.12, 0.03, T, M, M_err=np.zeros_like(M))
    assert np.isclose(homo, hetero)


def test_heteroscedastic_likelihood_downweights_noisy_points():
    """A residual on a point with a large Monte Carlo error costs less."""
    T = np.array([1.5, 2.0])
    M_pred = (1 - T / 2.3) ** 0.12
    M_obs = M_pred + np.array([0.0, 0.05])
    quiet = _log_likelihood(2.3, 0.12, 0.01, T, M_obs, M_err=np.array([0.0, 0.0]))
    noisy = _log_likelihood(2.3, 0.12, 0.01, T, M_obs, M_err=np.array([0.0, 0.05]))
    assert noisy > quiet


def test_fixed_amplitude_scales_the_power_law():
    """With amplitude A the likelihood peaks on data generated as A (1 − T/Tc)^β."""
    T = np.linspace(1.5, 2.2, 10)
    M = 1.2224 * (1 - T / 2.27) ** 0.125
    assert _log_likelihood(2.27, 0.125, 0.01, T, M, amplitude=1.2224) > \
        _log_likelihood(2.27, 0.125, 0.01, T, M)


def test_onsager_amplitude_is_the_leading_coefficient():
    """M_Onsager / (1 − T/Tc)^(1/8) → A = (8√2/Tc)^(1/8) as T → Tc⁻."""
    from scripts.model_comparison import ONSAGER_AMPLITUDE, TC_EXACT, onsager_magnetization
    eps = 1e-8
    M = onsager_magnetization(np.array([TC_EXACT * (1 - eps)]))[0]
    assert abs(M / eps ** 0.125 - ONSAGER_AMPLITUDE) < 1e-5
    assert abs(ONSAGER_AMPLITUDE - 1.2224) < 1e-4


def test_correction_to_scaling_recovers_onsager():
    """On Onsager's exact curve the corrected model M = A t^β (1 + b t) gives
    β ≈ 1/8, and its truncation remainder is below 3e-4 on T < 2.25."""
    from scripts.model_comparison import (
        TC_EXACT, _log_post_correction, onsager_magnetization, truncation_remainder)
    T = np.linspace(1.5, 2.2, 9)
    M = onsager_magnetization(T)
    b, rem = truncation_remainder(T)
    assert abs(b + 0.225) < 0.005 and rem.max() < 3e-4
    err = np.full_like(T, 1e-3)
    truth = _log_post_correction(np.array([TC_EXACT, 0.125, b]), T, M, err)
    assert truth > _log_post_correction(np.array([TC_EXACT, 0.15, b]), T, M, err)
