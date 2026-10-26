"""Tests for the finite-size-scaling measurement and fit."""
from __future__ import annotations

import numpy as np

from src.fss import _binder, _jackknife, measure
from src.fss_fit import binder_crossings, chi2_fit, log_post, preliminary, sample, scaling_window


def test_binder_limits():
    """U = 2/3 for a fully ordered magnetisation and → 0 for a Gaussian one."""
    assert np.isclose(_binder(np.array([1.0, -1.0, 1.0, 1.0])), 2.0 / 3.0)
    rng = np.random.default_rng(0)
    assert abs(_binder(rng.normal(0, 0.1, 200_000))) < 0.01


def test_jackknife_error_of_mean_matches_standard_error_for_iid_data():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 50_000)
    _, err = _jackknife(x, 50, np.mean)
    assert abs(err / (1 / np.sqrt(len(x))) - 1) < 0.3


def test_measure_returns_physical_values():
    r = measure(8, 1.5, n_therm=200, n_measure=2000, seed=0, n_blocks=20)
    assert 0.9 < r["M_abs"] <= 1.0
    assert 0.6 < r["U"] <= 2.0 / 3.0 + 1e-9


def _synthetic(Tc=2.27, nu=1.0, b=0.125, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    L = np.repeat([16.0, 32.0, 64.0], 9)
    T = np.tile(np.linspace(2.23, 2.31, 9), 3)
    x = (T - Tc) * L ** (1 / nu)
    U = 0.61 - 0.05 * x - 0.005 * x ** 2
    M = L ** -b * (1.0 - 0.1 * x - 0.01 * x ** 2)
    e = np.full_like(T, noise)
    return {"L": L, "T": T, "U": U + rng.normal(0, noise, T.shape), "U_err": e,
            "M_abs": M + rng.normal(0, noise, T.shape), "M_abs_err": e}


def test_log_post_prefers_truth():
    d = _synthetic()
    assert log_post(np.array([2.27, 1.0, 0.125, 1e-4, 1e-4]), d) > \
        log_post(np.array([2.30, 1.3, 0.20, 1e-4, 1e-4]), d)


def test_fit_recovers_synthetic_parameters():
    d = _synthetic()
    s, c = sample(d, draws=1500, tune=3000, chains=2, seed=0)
    post = s.reshape(-1, s.shape[-1])
    assert abs(post[:, 0].mean() - 2.27) < 0.005
    assert abs(post[:, 1].mean() - 1.0) < 0.1
    assert abs(post[:, 2].mean() - 0.125) < 0.02
    assert abs(c.reshape(-1, 6)[:, 0].mean() - 0.61) < 0.01


def test_binder_crossing_of_synthetic_curves():
    d = _synthetic(noise=1e-6)
    cr = binder_crossings(d, [16, 32, 64], n_boot=50)
    for row in cr:
        assert abs(row["T_cross"] - 2.27) < 0.002


def test_preliminary_estimates_from_synthetic_curves():
    t_star, nu0 = preliminary(_synthetic(nu=0.9, noise=1e-6), [16, 32, 64])
    assert abs(t_star - 2.27) < 0.002
    assert abs(nu0 - 0.9) < 0.05


def test_scaling_window_keeps_the_same_x_range_for_every_size():
    d = scaling_window(_synthetic(), 1.0, 2.27, 1.0)
    x = (d["T"] - 2.27) * d["L"]
    assert np.all(np.abs(x) <= 1.0 + 1e-9)
    assert {int(L): int((d["L"] == L).sum()) for L in (16, 32, 64)} == {16: 9, 32: 7, 64: 3}
    for row in binder_crossings(d, [16, 32, 64], n_boot=20):     # on the shared temperatures
        assert abs(row["T_cross"] - 2.27) < 0.005


def test_chi2_fit_of_synthetic_data_is_acceptable_and_recovers_parameters():
    r = chi2_fit(_synthetic(seed=2))
    assert abs(r["Tc"] - 2.27) < 0.005
    assert abs(r["nu"] - 1.0) < 0.1
    assert r["dof"] == 2 * 27 - 9
    assert r["Q"] > 1e-3


def test_chi2_fit_errors_and_higher_order_scaling_functions():
    d = _synthetic(seed=2)
    r2, r3 = chi2_fit(d), chi2_fit(d, order=3)
    assert 0 < r2["Tc_err"] < 0.005 and 0 < r2["nu_err"] < 0.1 and 0 < r2["beta_err"] < 0.02
    assert r3["dof"] == r2["dof"] - 2
    assert r3["chi2"] <= r2["chi2"] + 1e-6                      # nested model fits at least as well
