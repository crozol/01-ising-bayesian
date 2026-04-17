"""Simulador del modelo de Ising 2D vía Metropolis-Hastings.

La red es cuadrada N x N con spines sᵢⱼ ∈ {−1, +1} y condiciones de frontera
periódicas. La dinámica de Metropolis acepta/rechaza flips individuales según
la diferencia de energía ΔE y la temperatura T, muestreando configuraciones
del ensemble canónico.

Energía del sistema:
    E = −J · Σ<i,j> sᵢ · sⱼ

Observable medido:
    <|M|> = <|(1/N²) · Σ sᵢⱼ|>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(func):
            return func
        return wrap


@njit(cache=True)
def _sweep(lattice: np.ndarray, T: float, J: float) -> None:
    """Un barrido completo: N² intentos de flip con regla de Metropolis."""
    n = lattice.shape[0]
    for _ in range(n * n):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        s = lattice[i, j]
        neighbors = (
            lattice[(i + 1) % n, j]
            + lattice[(i - 1) % n, j]
            + lattice[i, (j + 1) % n]
            + lattice[i, (j - 1) % n]
        )
        dE = 2.0 * J * s * neighbors
        if dE <= 0.0 or np.random.random() < np.exp(-dE / T):
            lattice[i, j] = -s


def simulate_at_temperature(
    T: float,
    *,
    size: int = 32,
    n_therm: int = 2000,
    n_measure: int = 3000,
    J: float = 1.0,
    seed: int = 0,
) -> dict:
    """Mide <|M|> y su desviación estándar a temperatura T."""
    np.random.seed(seed)
    lattice = np.where(np.random.random((size, size)) < 0.5, -1, 1).astype(np.int8)

    for _ in range(n_therm):
        _sweep(lattice, T, J)

    samples = np.empty(n_measure)
    for k in range(n_measure):
        _sweep(lattice, T, J)
        samples[k] = np.abs(lattice.mean())

    return {
        "T": float(T),
        "M_mean": float(samples.mean()),
        "M_std": float(samples.std()),
    }


def run_full_simulation(
    out_csv: str = "data/magnetization.csv",
    *,
    size: int = 32,
    n_temps: int = 30,
    t_min: float = 1.5,
    t_max: float = 3.5,
    n_therm: int = 2000,
    n_measure: int = 3000,
    seed: int = 0,
) -> list[dict]:
    """Barre un rango de temperaturas y guarda la curva M(T) en CSV."""
    T_values = np.linspace(t_min, t_max, n_temps)
    results: list[dict] = []
    for i, T in enumerate(T_values):
        r = simulate_at_temperature(
            T, size=size, n_therm=n_therm, n_measure=n_measure, seed=seed + i
        )
        results.append(r)
        print(f"  T = {T:5.3f}  <|M|> = {r['M_mean']:.4f}  σ = {r['M_std']:.4f}")

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["T", "M_mean", "M_std"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[ok] guardado en {out_path}  ({len(results)} filas)")
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/magnetization.csv")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--n-temps", type=int, default=30)
    parser.add_argument("--t-min", type=float, default=1.5)
    parser.add_argument("--t-max", type=float, default=3.5)
    parser.add_argument("--n-therm", type=int, default=2000)
    parser.add_argument("--n-measure", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[metropolis] numba = {_HAS_NUMBA}")
    run_full_simulation(
        out_csv=args.out,
        size=args.size,
        n_temps=args.n_temps,
        t_min=args.t_min,
        t_max=args.t_max,
        n_therm=args.n_therm,
        n_measure=args.n_measure,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
