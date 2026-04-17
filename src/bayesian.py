"""Bayesian inference of (Tc, β) from the simulated M(T) curve.

Near the second-order phase transition of the 2D Ising model, the magnetization
follows a power law:

    M(T) ≈ (1 − T/Tc)^β   if T < Tc
    M(T) = 0             otherwise

A probabilistic model is built with physically motivated priors. Two backends
are provided:

- ``numpy`` (default): a random-walk Metropolis-Hastings sampler implemented
  directly in NumPy. Fast on any machine, no C compiler required.
- ``pymc``: the reference PyMC + NUTS implementation. More sophisticated, but
  requires a working PyTensor C compiler to be fast.

Both backends produce an ``arviz.InferenceData`` saved as NetCDF so downstream
tooling (``src/plots.py``) does not care which sampler was used.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# ----------------------------- log-posterior -----------------------------

def _log_prior(Tc: float, beta: float, sigma: float) -> float:
    """Priors: Tc ~ Normal(2.3, 0.3), β ~ Normal(0.12, 0.05), σ ~ HalfNormal(0.1)."""
    if not (1.0 < Tc < 4.0):
        return -np.inf
    if not (0.005 < beta < 0.6):
        return -np.inf
    if sigma <= 0.0:
        return -np.inf
    lp = -0.5 * ((Tc - 2.3) / 0.3) ** 2
    lp += -0.5 * ((beta - 0.12) / 0.05) ** 2
    # HalfNormal(σ_scale=0.1): log density ∝ −σ²/(2·0.1²)
    lp += -0.5 * (sigma / 0.1) ** 2
    return lp


def _log_likelihood(Tc: float, beta: float, sigma: float, T: np.ndarray, M: np.ndarray) -> float:
    """Gaussian likelihood with piecewise power-law mean."""
    arg = np.clip(1.0 - T / Tc, 1e-10, 1.0)
    M_pred = np.where(T < Tc, arg ** beta, 0.0)
    n = len(M)
    return -0.5 * np.sum((M - M_pred) ** 2) / (sigma ** 2) - n * np.log(sigma)


def _log_posterior(theta: np.ndarray, T: np.ndarray, M: np.ndarray) -> float:
    Tc, beta, sigma = theta
    lp = _log_prior(Tc, beta, sigma)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(Tc, beta, sigma, T, M)


# --------------------------- NumPy MH sampler ---------------------------

def sample_mh_numpy(
    T: np.ndarray,
    M: np.ndarray,
    *,
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    step: tuple[float, float, float] = (0.05, 0.01, 0.015),
    seed: int = 0,
) -> dict:
    """Random-walk Metropolis-Hastings.

    Returns a dict with arrays of shape (chains, draws) for each parameter,
    ready to wrap into an arviz.InferenceData object.
    """
    rng = np.random.default_rng(seed)
    samples_Tc = np.empty((chains, draws))
    samples_beta = np.empty((chains, draws))
    samples_sigma = np.empty((chains, draws))

    for c in range(chains):
        # random initialization within each prior's support
        theta = np.array([
            rng.normal(2.3, 0.2),
            np.abs(rng.normal(0.12, 0.04)) + 0.01,
            np.abs(rng.normal(0.05, 0.02)) + 1e-3,
        ])
        log_p = _log_posterior(theta, T, M)
        accept = 0
        total = 0

        for i in range(-tune, draws):
            prop = theta + rng.normal(0.0, step, 3)
            log_p_new = _log_posterior(prop, T, M)
            total += 1
            if np.log(rng.random() + 1e-300) < log_p_new - log_p:
                theta = prop
                log_p = log_p_new
                accept += 1
            if i >= 0:
                samples_Tc[c, i] = theta[0]
                samples_beta[c, i] = theta[1]
                samples_sigma[c, i] = theta[2]

        print(f"  [chain {c+1}/{chains}] acceptance rate: {accept/total:.1%}", flush=True)

    return {
        "Tc": samples_Tc,
        "beta": samples_beta,
        "sigma": samples_sigma,
    }


def _to_inferencedata(samples: dict):
    """Wrap NumPy samples into arviz.InferenceData for uniform downstream handling."""
    import arviz as az
    import xarray as xr

    coords = {
        "chain": np.arange(samples["Tc"].shape[0]),
        "draw": np.arange(samples["Tc"].shape[1]),
    }
    data_vars = {k: (("chain", "draw"), v) for k, v in samples.items()}
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return az.InferenceData(posterior=ds)


# --------------------------- PyMC backend (optional) ---------------------------

def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    T, M = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            T.append(float(row["T"]))
            M.append(float(row["M_mean"]))
    return np.array(T), np.array(M)


def build_model_pymc(T_obs: np.ndarray, M_obs: np.ndarray):
    import pymc as pm

    with pm.Model() as model:
        Tc = pm.Normal("Tc", mu=2.3, sigma=0.3)
        beta = pm.Normal("beta", mu=0.12, sigma=0.05)
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        arg = pm.math.clip(1.0 - T_obs / Tc, 1e-10, 1.0)
        M_pred = pm.math.switch(T_obs < Tc, arg ** beta, 0.0)
        pm.Normal("M_obs", mu=M_pred, sigma=sigma, observed=M_obs)
    return model


# ------------------------------ orchestration ------------------------------

def run_full_inference(
    csv_in: str = "data/magnetization.csv",
    trace_out: str = "data/trace.nc",
    *,
    backend: str = "numpy",
    draws: int = 3000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 0,
):
    import arviz as az

    T_obs, M_obs = load_data(csv_in)
    print(f"[bayesian] {len(T_obs)} points loaded from {csv_in}", flush=True)
    print(f"[bayesian] backend = {backend}", flush=True)

    if backend == "numpy":
        samples = sample_mh_numpy(
            T_obs, M_obs, draws=draws, tune=tune, chains=chains, seed=seed,
        )
        trace = _to_inferencedata(samples)
    elif backend == "pymc":
        import pymc as pm
        model = build_model_pymc(T_obs, M_obs)
        with model:
            trace = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=1,
                target_accept=target_accept, random_seed=seed,
                progressbar=False, compute_convergence_checks=True,
            )
    else:
        raise ValueError(f"unknown backend: {backend}")

    summary = az.summary(trace, var_names=["Tc", "beta", "sigma"], hdi_prob=0.95)
    print("\n[bayesian] posterior summary:\n", flush=True)
    print(summary, flush=True)

    out_path = Path(trace_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(trace, out_path)
    print(f"\n[ok] trace saved to {out_path}", flush=True)
    return trace


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="data/magnetization.csv")
    parser.add_argument("--output", type=str, default="data/trace.nc")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "pymc"])
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_full_inference(
        csv_in=args.input,
        trace_out=args.output,
        backend=args.backend,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
