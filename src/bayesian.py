"""Inferencia bayesiana de (Tc, β) a partir de la curva M(T) simulada.

Cerca de la transición de fase de segundo orden, la magnetización del modelo
de Ising 2D sigue una ley de potencias:

    M(T) ≈ (1 − T/Tc)^β   si T < Tc
    M(T) = 0             si T ≥ Tc

Planteo un modelo probabilístico con priors físicamente motivados y uso NUTS
(PyMC) para muestrear el posterior P(Tc, β | datos).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Lee la curva M(T) generada por el simulador."""
    T, M = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            T.append(float(row["T"]))
            M.append(float(row["M_mean"]))
    return np.array(T), np.array(M)


def build_model(T_obs: np.ndarray, M_obs: np.ndarray):
    """Construye el modelo PyMC con priors sobre (Tc, β, σ)."""
    import pymc as pm

    with pm.Model() as model:
        Tc = pm.Normal("Tc", mu=2.3, sigma=0.3)
        beta = pm.Normal("beta", mu=0.12, sigma=0.05)
        sigma = pm.HalfNormal("sigma", sigma=0.1)

        # M_pred = (1 - T/Tc)^beta  si T<Tc  ;  0 en otro caso
        arg = pm.math.clip(1.0 - T_obs / Tc, 1e-8, 1.0)
        M_pred = pm.math.switch(T_obs < Tc, arg**beta, 0.0)

        pm.Normal("M_obs", mu=M_pred, sigma=sigma, observed=M_obs)

    return model


def run_full_inference(
    csv_in: str = "data/magnetization.csv",
    trace_out: str = "data/trace.nc",
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 0,
):
    """Carga datos, corre NUTS, imprime summary y guarda el trace en NetCDF."""
    import arviz as az
    import pymc as pm

    T_obs, M_obs = load_data(csv_in)
    print(f"[bayesian] {len(T_obs)} puntos cargados desde {csv_in}")

    model = build_model(T_obs, M_obs)
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )

    summary = az.summary(trace, var_names=["Tc", "beta", "sigma"], hdi_prob=0.95)
    print("\n[bayesian] resumen del posterior:\n")
    print(summary)

    out_path = Path(trace_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(trace, out_path)
    print(f"\n[ok] trace guardado en {out_path}")
    return trace


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="data/magnetization.csv")
    parser.add_argument("--output", type=str, default="data/trace.nc")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_full_inference(
        csv_in=args.input,
        trace_out=args.output,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
