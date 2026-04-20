"""Export simulation data + MCMC posterior samples to JSON for the portfolio site.

The output JSON is consumed by Plotly charts on the project detail page.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import arviz as az
import numpy as np
from scipy.optimize import curve_fit


ONSAGER_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))
EXACT_BETA = 0.125


def _asym_sigmoid(T, A, Tc, wL, wR, c):
    """Asymmetric sigmoid:

        M(T) = A/2 · [1 − tanh((T − Tc)/w)] + c
        w = wL  if T < Tc ,  wR  if T >= Tc

    Different widths above and below Tc capture the finite-size Ising curve
    much better than a symmetric tanh (the left side is broad because the
    ordered phase approaches 1 gradually; the right side is sharp because
    the transition collapses M toward the finite-size tail).
    """
    w = np.where(T < Tc, wL, wR)
    return A * 0.5 * (1.0 - np.tanh((T - Tc) / w)) + c


def _fit_smoothed(T: np.ndarray, M: np.ndarray, S: np.ndarray) -> dict:
    """Weighted nonlinear least squares fit of the asymmetric sigmoid."""
    popt, pcov = curve_fit(
        _asym_sigmoid, T, M,
        p0=[0.92, 2.42, 0.30, 0.08, 0.08],
        sigma=S,
        absolute_sigma=False,
        bounds=([0.5, 2.0, 0.01, 0.01, 0.0], [1.2, 2.8, 1.0, 1.0, 0.25]),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    M_pred = _asym_sigmoid(T, *popt)
    chi2 = float(np.sum(((M - M_pred) / S) ** 2))
    dof = int(len(M) - len(popt))
    return {
        "A":  round(float(popt[0]), 4),
        "Tc": round(float(popt[1]), 4),
        "wL": round(float(popt[2]), 4),
        "wR": round(float(popt[3]), 4),
        "c":  round(float(popt[4]), 4),
        "A_err":  round(float(perr[0]), 4),
        "Tc_err": round(float(perr[1]), 4),
        "wL_err": round(float(perr[2]), 4),
        "wR_err": round(float(perr[3]), 4),
        "c_err":  round(float(perr[4]), 4),
        "chi2": round(chi2, 3),
        "dof": dof,
        "chi2_reduced": round(chi2 / dof, 3) if dof > 0 else None,
    }


def _load_csv(path: str) -> dict:
    T, M, S = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            T.append(float(row["T"]))
            M.append(float(row["M_mean"]))
            S.append(float(row["M_std"]))
    return {"T": T, "M_mean": M, "M_std": S}


def _thin(array_2d: np.ndarray, max_draws: int = 1500) -> np.ndarray:
    """Subsample draws dimension to cap file size."""
    n_chains, n_draws = array_2d.shape
    if n_draws <= max_draws:
        return array_2d
    step = max(1, n_draws // max_draws)
    return array_2d[:, ::step]


def _posterior_to_dict(trace) -> dict:
    post = trace.posterior
    Tc = _thin(post["Tc"].values)
    beta = _thin(post["beta"].values)
    sigma = _thin(post["sigma"].values)
    return {
        "Tc": Tc.tolist(),
        "beta": beta.tolist(),
        "sigma": sigma.tolist(),
        "n_chains": int(Tc.shape[0]),
        "n_draws": int(Tc.shape[1]),
    }


def _summary_dict(trace) -> dict:
    summary = az.summary(trace, var_names=["Tc", "beta", "sigma"], hdi_prob=0.95)
    out = {}
    for param in ["Tc", "beta", "sigma"]:
        out[param] = {
            "mean": round(float(summary.loc[param, "mean"]), 4),
            "sd": round(float(summary.loc[param, "sd"]), 4),
            "hdi": [
                round(float(summary.loc[param, "hdi_2.5%"]), 4),
                round(float(summary.loc[param, "hdi_97.5%"]), 4),
            ],
            "ess_bulk": int(summary.loc[param, "ess_bulk"]),
            "ess_tail": int(summary.loc[param, "ess_tail"]),
            "r_hat": round(float(summary.loc[param, "r_hat"]), 3),
        }
    return out


def main(
    csv_in: str = "data/magnetization.csv",
    trace_in: str = "data/trace.nc",
    out_path: str = "../website/assets/data/01-ising.json",
) -> None:
    trace = az.from_netcdf(trace_in)
    mag = _load_csv(csv_in)

    fit = _fit_smoothed(
        np.array(mag["T"]),
        np.array(mag["M_mean"]),
        np.array(mag["M_std"]),
    )
    print(f"[fit] A  = {fit['A']} ± {fit['A_err']}")
    print(f"[fit] Tc = {fit['Tc']} ± {fit['Tc_err']}")
    print(f"[fit] wL = {fit['wL']} ± {fit['wL_err']}")
    print(f"[fit] wR = {fit['wR']} ± {fit['wR_err']}")
    print(f"[fit] c  = {fit['c']} ± {fit['c_err']}")
    print(f"[fit] chi2/dof = {fit['chi2_reduced']}")

    data = {
        "magnetization": mag,
        "posterior": _posterior_to_dict(trace),
        "summary": _summary_dict(trace),
        "fit": fit,
        "exact": {
            "Tc": round(ONSAGER_TC, 6),
            "beta": EXACT_BETA,
        },
        "metadata": {
            "lattice_size": 32,
            "n_temps": 25,
            "n_thermalization": 2000,
            "n_measure": 3000,
            "sampler": "random-walk Metropolis-Hastings (NumPy)",
            "mcmc_chains": 4,
            "mcmc_draws_post_burn": 5000,
            "mcmc_tune": 1000,
        },
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))  # minified

    size_kb = out.stat().st_size / 1024
    print(f"[ok] {out}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
