"""Histogram reweighting (src/reweight.py) and the correlated fit (src/fss_reweighted.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.fss import measure
from src.reweight import Runs, jackknife_cov


@pytest.fixture(scope="module")
def series(tmp_path_factory):
    d = tmp_path_factory.mktemp("series")
    rows = {}
    for i, T in enumerate((2.24, 2.30, 2.36)):
        rows[T] = measure(8, T, n_therm=500, n_measure=20_000, seed=11 + i, n_blocks=20,
                          algorithm="wolff", series_out=str(d / f"L8_T{T:.4f}.npz"))
    return d, rows


def test_one_run_reweighted_to_its_own_temperature_is_its_direct_average(series):
    d, rows = series
    r = Runs([str(d / "L8_T2.3000.npz")], n_blocks=20)
    o = Runs.observables(r.solve(), [2.30])
    assert o["M_abs"][0] == pytest.approx(rows[2.30]["M_abs"], rel=1e-12)
    assert o["U"][0] == pytest.approx(rows[2.30]["U"], rel=1e-12)


def test_multiple_histograms_agree_with_the_direct_averages(series):
    d, rows = series
    r = Runs.from_dir(str(d), 8, n_blocks=20)
    temps = [2.24, 2.27, 2.30, 2.33, 2.36]
    c = r.curves(temps)
    err_U = np.sqrt(np.diag(jackknife_cov(c["jackknife"]["U"])))
    for k, T in enumerate(temps):
        if T in rows:
            assert abs(c["U"][k] - rows[T]["U"]) < 4 * np.hypot(err_U[k], rows[T]["U_err"])
    assert np.all(np.diff(c["M_abs"]) < 0)                   # <|m|> falls with T
    with pytest.raises(ValueError):
        r.curves([2.20])                                     # outside the simulated range


def test_generalised_chi2_with_a_diagonal_covariance_is_the_classical_one():
    from src.fss_fit import _chi2
    from src.fss_reweighted import chi2_corr

    rng = np.random.default_rng(0)
    pts, rows = [], {k: [] for k in ("L", "T", "U", "U_err", "M_abs", "M_abs_err")}
    for L in (32.0, 64.0):
        T = 2.269 + np.linspace(-1, 1, 5) / L
        U = 0.61 - 0.4 * (T - 2.269) * L + 0.001 * rng.normal(size=5)
        M = L ** -0.125 * (1.0 - 0.8 * (T - 2.269) * L) + 0.001 * rng.normal(size=5)
        eU, eM = np.full(5, 0.002), np.full(5, 0.003)
        pts.append({"L": L, "T": T, "U": U, "M_abs": M, "W": np.diag(np.r_[1 / eU, 1 / eM])})
        for k, v in zip(rows, (np.full(5, L), T, U, eU, M, eM)):
            rows[k].append(v)
    d = {k: np.concatenate(v) for k, v in rows.items()}
    th = (2.2695, 1.01, 0.124)
    assert chi2_corr(th, pts) == pytest.approx(_chi2(th, d), rel=1e-10)
