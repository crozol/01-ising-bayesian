"""Simulador del modelo de Ising 2D vía Metropolis-Hastings.

La red es cuadrada N x N con spines sᵢⱼ ∈ {−1, +1} y condiciones de frontera
periódicas. La dinámica de Metropolis acepta/rechaza flips individuales según
la diferencia de energía ΔE y la temperatura T, muestreando configuraciones
del ensemble canónico.

Energía del sistema:
    E = −J · Σ<i,j> sᵢ · sⱼ

Observables medidos por temperatura (sobre las configuraciones de equilibrio):
    <|M|>   = <|(1/N²) · Σ sᵢⱼ|>                    magnetización por sitio
    <ε>     = <E> / N²                              energía por sitio  (∈ [−2, 2])
    χ       = (N² / T) · (<m²> − <|m|>²)            susceptibilidad por sitio
    C       = (N² / T²) · (<ε²> − <ε>²)             calor específico por sitio
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


@njit(cache=True)
def _energy_per_site(lattice: np.ndarray, J: float) -> float:
    """Energía por sitio E/N². Cada enlace se cuenta una vez (vecinos der + abajo)."""
    n = lattice.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            s = lattice[i, j]
            total += s * (lattice[i, (j + 1) % n] + lattice[(i + 1) % n, j])
    return -J * total / (n * n)


def _block_mean_error(x: np.ndarray, n_blocks: int) -> float:
    """Error estadístico de la media vía blocking (absorbe la autocorrelación MC)."""
    n_blocks = max(2, min(n_blocks, len(x)))
    bs = len(x) // n_blocks
    block_means = x[: bs * n_blocks].reshape(n_blocks, bs).mean(axis=1)
    return float(block_means.std(ddof=1) / np.sqrt(n_blocks))


def _block_jackknife_var(x: np.ndarray, n_blocks: int) -> tuple[float, float]:
    """Varianza poblacional y su error por jackknife sobre bloques.

    El jackknife por bloques (Newman & Barkema 1999, cap. 3) da una barra de error
    fiable para un estimador no lineal como la varianza incluso cuando las muestras
    están autocorrelacionadas, siempre que el bloque sea más largo que el tiempo de
    autocorrelación. Cerca de Tc ese tiempo diverge (critical slowing down), así que
    el error crece — que es justo lo que la figura debe mostrar.
    """
    n_blocks = max(2, min(n_blocks, len(x)))
    bs = len(x) // n_blocks
    x = x[: bs * n_blocks]
    block_id = np.arange(len(x)) // bs
    var_full = float(x.var())
    leave_one = np.array([x[block_id != b].var() for b in range(n_blocks)])
    jack_mean = leave_one.mean()
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((leave_one - jack_mean) ** 2))
    return var_full, float(err)


def simulate_at_temperature(
    T: float,
    *,
    size: int = 32,
    n_therm: int = 2000,
    n_measure: int = 3000,
    J: float = 1.0,
    seed: int = 0,
    n_blocks: int = 20,
) -> dict:
    """Mide <|M|>, <ε>, susceptibilidad χ y calor específico C a temperatura T.

    χ y C llevan barra de error por jackknife sobre bloques; <ε> por blocking de
    la media. Cerca de Tc esas barras crecen por el critical slowing down.
    """
    np.random.seed(seed)
    # Cold (ordered) start: a random start can trap the low-T runs in a long-lived
    # domain-wall (stripe) state — metastable and exponentially slow to anneal out.
    # Starting from the ordered ground state samples the broken-symmetry phase
    # robustly; above Tc the burn-in disorders it just the same.
    lattice = np.ones((size, size), dtype=np.int8)

    for _ in range(n_therm):
        _sweep(lattice, T, J)

    m_samples = np.empty(n_measure)
    e_samples = np.empty(n_measure)
    for k in range(n_measure):
        _sweep(lattice, T, J)
        m_samples[k] = np.abs(lattice.mean())
        e_samples[k] = _energy_per_site(lattice, J)

    n_spins = size * size
    var_m, var_m_err = _block_jackknife_var(m_samples, n_blocks)
    var_e, var_e_err = _block_jackknife_var(e_samples, n_blocks)

    return {
        "T": float(T),
        "M_mean": float(m_samples.mean()),
        "M_std": float(m_samples.std()),
        "E_mean": float(e_samples.mean()),
        "E_err": _block_mean_error(e_samples, n_blocks),
        "chi": n_spins * var_m / T,
        "chi_err": n_spins * var_m_err / T,
        "C": n_spins * var_e / (T * T),
        "C_err": n_spins * var_e_err / (T * T),
    }


def run_full_simulation(
    out_csv: str = "data/magnetization.csv",
    *,
    size: int = 32,
    n_temps: int = 25,
    t_min: float = 1.5,
    t_max: float = 3.5,
    n_therm: int = 2000,
    n_measure: int = 3000,
    seed: int = 0,
) -> list[dict]:
    """Barre un rango de temperaturas y guarda los observables M(T), ε(T), χ(T), C(T) en CSV."""
    T_values = np.linspace(t_min, t_max, n_temps)
    results: list[dict] = []
    for i, T in enumerate(T_values):
        r = simulate_at_temperature(
            T, size=size, n_therm=n_therm, n_measure=n_measure, seed=seed + i
        )
        results.append(r)
        print(
            f"  T = {T:5.3f}  |M| = {r['M_mean']:.4f}  E = {r['E_mean']:+.4f}"
            f"  chi = {r['chi']:7.3f} +/- {r['chi_err']:6.3f}"
            f"  C = {r['C']:6.3f} +/- {r['C_err']:5.3f}"
        )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["T", "M_mean", "M_std", "E_mean", "E_err", "chi", "chi_err", "C", "C_err"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[ok] guardado en {out_path}  ({len(results)} filas)")
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="data/magnetization.csv")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--n-temps", type=int, default=25)
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
