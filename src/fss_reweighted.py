"""Finite-size-scaling fit on reweighted curves (src/reweight.py).

The fit of src/fss_fit.py uses the simulated temperatures only: inside the scaling window
|T - T*| L^(1/nu0) <= x_max, L = 128 has 2 of them and L = 16 seven, so an amplitude of the
correction to scaling that depends on x cannot be resolved (README, section 12). Reweighting all 15
runs of a size (Ferrenberg & Swendsen 1989) gives <|m|> and U at any temperature in the simulated
range. Here every size gets G evenly spaced points of the scaling variable over |x| <= x_max, at
T_g = T* + x_g / L^(1/nu0), with T* and nu0 the same model-free values that place the window of
src/fss_fit.py, so the windows are the same and only the sampling of the curves changes. Where the
window reaches beyond the simulated temperatures (L = 16 at its low end: 2.199 < 2.20), the points
cover the part of it inside the simulated range.

Points reweighted from the same runs are correlated, U and <|m|> too. The first approach was a
generalised chi^2 with the full block-jackknife covariance of each size (chi2_corr, fit; Hartlap's
factor (n_b - p - 2) / (n_b - 1) on the inverse, Hartlap, Simon & Schneider 2007). It does not work
here, and the diagnostic is kept in the output: the eigenvalues of the correlation matrix of 5 points
of U and <|m|> at L = 64 span 7 orders of magnitude (8.8 to 4e-7), because the reweighted curves fix
their own curvature and higher derivatives far more precisely than the points themselves; the
generalised chi^2 then rejects every polynomial scaling function, even of order 5 (L = 32, 64:
827 for 5 degrees of freedom), and with 7 points per size the estimated covariance is not even
positive definite. A goodness-of-fit test on dense reweighted curves therefore cannot choose between
corrections to scaling.

What is reported instead (jackknife_fit) is least squares on the jackknife errors alone, with the
errors of (Tc, nu, beta) from refitting the whole curve of every jackknife sample, so the correlations
are accounted for in the errors without inverting the covariance; the diagonal chi^2 is only
descriptive. Fits: the default of src/fss_fit.py (L = 32-128, |x| <= 1, quadratic scaling
functions), the same with L = 16, and with a wider window, each without a correction to scaling, with
a constant-amplitude correction c L^(-omega), and with an amplitude linear in x, (c0 + c1 x)
L^(-omega), for omega = 1 and 2 (and 1.75, the leading correction of the magnetisation in the
literature, as in scripts/scaling_corrections.py). The spread of the estimates across these fits is
the systematic uncertainty that the statistical errors leave out.

    python -m src.fss_reweighted                       # data/series_wolff -> results/fss_reweighted.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .fss_fit import _hessian, load, preliminary
from .reweight import Runs, jackknife_cov


def reweighted_points(series_dir: str, sizes, t_star: float, nu0: float, x_max: float,
                      n_points: int = 5, runs: dict | None = None) -> list[dict]:
    """For every size: the G temperatures of the window, U and <|m|> there, and the whitening matrix
    W (W^T W = Hartlap-corrected inverse covariance of the 2G values)."""
    out = []
    for L in sizes:
        r = runs[L] if runs is not None else Runs.from_dir(series_dir, L)
        scale = L ** (1.0 / nu0)
        lo = max(-x_max, (r.T.min() - t_star) * scale)          # the window, inside the simulated range
        hi = min(x_max, (r.T.max() - t_star) * scale)
        T = t_star + np.linspace(lo, hi, n_points) / scale
        c = r.curves(T)
        jk = np.concatenate([c["jackknife"]["U"], c["jackknife"]["M_abs"]], axis=1)
        cov = jackknife_cov(jk)
        p, nb = cov.shape[0], r.n_blocks
        hartlap = (nb - p - 2) / (nb - 1)
        if hartlap <= 0:
            raise ValueError(f"L = {L}: {nb} blocks are too few for {p} correlated points")
        prec = hartlap * np.linalg.inv(cov)
        W = np.linalg.cholesky(prec).T              # W^T W = prec
        out.append({"L": float(L), "T": T, "U": c["U"], "M_abs": c["M_abs"], "W": W,
                    "U_err": np.sqrt(np.diag(cov)[:n_points]), "M_err": np.sqrt(np.diag(cov)[n_points:]),
                    "n_blocks": nb, "hartlap": hartlap, "cond": float(np.linalg.cond(cov))})
    return out


def _columns(T, L, Tc, nu, order, omega, omega_x):
    x = (T - Tc) * L ** (1.0 / nu)
    cols = [x ** k for k in range(order + 1)]
    if omega is not None:
        cols.append(np.full_like(x, L ** (-float(omega))))
        if omega_x:
            cols.append(x * L ** (-float(omega)))
    return np.stack(cols, 1)


def chi2_corr(th, pts, order=2, omega=None, omega_x=False, return_coef=False):
    """Generalised chi^2 of both scaling forms; U and <|m|> have their own linear coefficients."""
    Tc, nu, b = th
    rows_y, rows_A = [], []
    for d in pts:
        A = _columns(d["T"], d["L"], Tc, nu, order, omega, omega_x)
        k = A.shape[1]
        Z = np.zeros_like(A)
        A_full = np.block([[A, Z], [Z, A * d["L"] ** (-b)]])       # U rows, then <|m|> rows
        y = np.concatenate([d["U"], d["M_abs"]])
        rows_y.append(d["W"] @ y)
        rows_A.append(d["W"] @ A_full)
    y, A = np.concatenate(rows_y), np.concatenate(rows_A)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    chi2 = float(np.sum((y - A @ c) ** 2))
    return (chi2, c, k) if return_coef else chi2


def fit(pts, start=(2.27, 1.0, 0.12), order=2, omega=None, omega_x=False) -> dict:
    from scipy import optimize, stats

    f = lambda th: chi2_corr(th, pts, order, omega, omega_x)  # noqa: E731
    r = optimize.minimize(f, start, method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 40_000})
    Tc, nu, b = r.x
    cov = 2 * np.linalg.inv(_hessian(f, r.x, np.array([2e-5, 2e-3, 2e-4])))
    grad_beta = np.array([0.0, b, nu])
    chi2, c, k = chi2_corr(r.x, pts, order, omega, omega_x, return_coef=True)
    n = sum(2 * len(d["T"]) for d in pts)
    dof = n - 3 - 2 * k
    return {"order": order, "omega": omega, "omega_x": omega_x,
            "Tc": float(Tc), "Tc_err": float(np.sqrt(cov[0, 0])),
            "nu": float(nu), "nu_err": float(np.sqrt(cov[1, 1])),
            "beta": float(nu * b), "beta_err": float(np.sqrt(grad_beta @ cov @ grad_beta)),
            "U0": float(c[0]), "chi2": chi2, "dof": int(dof), "Q": float(stats.chi2.sf(chi2, dof))}


def curve_samples(runs: dict, sizes, t_star: float, nu0: float, x_max: float, n_points: int = 5) -> list[dict]:
    """Per size: the window's temperatures, U and <|m|> there, and their block-jackknife samples."""
    out = []
    for L in sizes:
        r = runs[L]
        scale = L ** (1.0 / nu0)
        lo = max(-x_max, (r.T.min() - t_star) * scale)
        hi = min(x_max, (r.T.max() - t_star) * scale)
        T = t_star + np.linspace(lo, hi, n_points) / scale
        c = r.curves(T)
        out.append({"L": float(L), "T": T, "U": c["U"], "M_abs": c["M_abs"],
                    "U_jk": c["jackknife"]["U"], "M_jk": c["jackknife"]["M_abs"], "n_blocks": r.n_blocks})
    return out


def _diagonal_points(samples: list[dict], b: int | None = None) -> list[dict]:
    """Points weighted by their jackknife errors alone; ``b`` selects jackknife sample b."""
    pts = []
    for s in samples:
        nb = s["n_blocks"]
        eU = np.sqrt((nb - 1) / nb * ((s["U_jk"] - s["U_jk"].mean(0)) ** 2).sum(0))
        eM = np.sqrt((nb - 1) / nb * ((s["M_jk"] - s["M_jk"].mean(0)) ** 2).sum(0))
        U, M = (s["U"], s["M_abs"]) if b is None else (s["U_jk"][b], s["M_jk"][b])
        pts.append({"L": s["L"], "T": s["T"], "U": U, "M_abs": M, "W": np.diag(np.r_[1 / eU, 1 / eM])})
    return pts


def jackknife_fit(samples: list[dict], order=2, omega=None, omega_x=False) -> dict:
    """Least squares on the jackknife errors alone (no inverse covariance), with the errors of
    (Tc, nu, beta) from refitting every jackknife sample: block b deleted from every run of every size
    (the sizes are independent simulations, so one joint deletion per sample is valid)."""
    import warnings

    nb = {s["n_blocks"] for s in samples}
    if len(nb) != 1:
        raise ValueError("every size needs the same number of jackknife blocks")
    nb = nb.pop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = fit(_diagonal_points(samples), order=order, omega=omega, omega_x=omega_x)
        start = (full["Tc"], full["nu"], full["beta"] / full["nu"])
        th = np.array([[r["Tc"], r["nu"], r["beta"]] for r in
                       (fit(_diagonal_points(samples, b), start=start, order=order, omega=omega,
                            omega_x=omega_x) for b in range(nb))])
    d = th - th.mean(0)
    cov = (nb - 1) / nb * d.T @ d
    err = np.sqrt(np.diag(cov))
    return {"order": order, "omega": omega, "omega_x": omega_x,
            "Tc": full["Tc"], "Tc_err": float(err[0]), "nu": full["nu"], "nu_err": float(err[1]),
            "beta": full["beta"], "beta_err": float(err[2]), "U0": full["U0"],
            "chi2_diagonal": full["chi2"], "dof": full["dof"], "n_blocks": nb}


def main(series_dir="data/series_wolff", csv="data/fss_wolff.csv", out="results/fss_reweighted.json",
         n_points=5) -> list:
    d = load(csv)
    t_star, nu0 = preliminary(d, [32, 64, 128])
    print(f"window: T* = {t_star:.5f}, nu0 = {nu0:.4f} (model-free, as in src/fss_fit.py)", flush=True)
    runs = {L: Runs.from_dir(series_dir, L) for L in (16, 32, 64, 128)}
    import warnings

    # diagnostic: the generalised chi^2 on the default window rejects every polynomial order
    diagnostic = []
    pts = reweighted_points(series_dir, (32, 64, 128), t_star, nu0, 1.0, n_points, runs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for order in (2, 3, 4, 5):
            r = fit(pts, order=order)
            diagnostic.append({"order": order, "chi2": r["chi2"], "dof": r["dof"], "Q": r["Q"]})
            print(f"generalised chi2, L=32-128, |x|<=1, order {order}: {r['chi2']:.1f}/{r['dof']} "
                  f"(Q {r['Q']:.2g}); correlation condition number up to {max(p['cond'] for p in pts):.1e}",
                  flush=True)
    results = []
    for sizes, x_max in (((32, 64, 128), 1.0), ((16, 32, 64, 128), 1.0), ((32, 64, 128), 1.5),
                         ((16, 32, 64, 128), 1.5)):
        samples = curve_samples(runs, sizes, t_star, nu0, x_max, n_points)
        for omega, omega_x in ((None, False), (1.0, False), (1.0, True), (1.75, True), (2.0, False),
                               (2.0, True)):
            r = jackknife_fit(samples, omega=omega, omega_x=omega_x)
            r.update({"sizes": list(sizes), "x_max": x_max, "n_points_per_size": n_points,
                      "t_star": t_star, "nu0": nu0})
            results.append(r)
            corr = "none" if omega is None else f"{'(c0 + c1 x)' if omega_x else 'c'} L^-{omega:g}"
            print(f"L={'-'.join(map(str, sizes)):11s} |x|<={x_max:<3g} corr {corr:20s} "
                  f"Tc {r['Tc']:.5f}({r['Tc_err'] * 1e5:.0f}) nu {r['nu']:.4f}({r['nu_err'] * 1e4:.0f}) "
                  f"beta {r['beta']:.4f}({r['beta_err'] * 1e4:.0f})  diagonal chi2 "
                  f"{r['chi2_diagonal']:.1f}/{r['dof']}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"t_star": t_star, "nu0": nu0, "generalised_chi2_diagnostic": diagnostic,
                                     "jackknife_fits": results}, indent=2))
    print(f"[ok] {out}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series-dir", default="data/series_wolff")
    p.add_argument("--csv", default="data/fss_wolff.csv")
    p.add_argument("--out", default="results/fss_reweighted.json")
    p.add_argument("--n-points", type=int, default=5)
    a = p.parse_args()
    main(a.series_dir, a.csv, a.out, a.n_points)
