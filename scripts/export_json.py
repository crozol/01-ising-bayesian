"""Export simulation data + MCMC posterior samples to JSON for the portfolio site.

The output JSON is consumed by Plotly charts on the project detail page.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import arviz as az
import numpy as np


ONSAGER_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))
EXACT_BETA = 0.125


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
    data = {
        "magnetization": _load_csv(csv_in),
        "posterior": _posterior_to_dict(trace),
        "summary": _summary_dict(trace),
        "exact": {
            "Tc": round(ONSAGER_TC, 6),
            "beta": EXACT_BETA,
        },
        "metadata": {
            "lattice_size": 28,
            "n_temps": 25,
            "n_thermalization": 1500,
            "n_measure": 2000,
            "sampler": "random-walk Metropolis-Hastings (NumPy)",
            "mcmc_chains": 4,
            "mcmc_draws_post_burn": 3000,
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
