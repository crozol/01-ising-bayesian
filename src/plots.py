"""Figuras del proyecto: curva M(T), posteriores y trace plots.

Todas las figuras se guardan en el directorio `results/` y están pensadas para
ir directo al README del repositorio.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# Solución exacta de Onsager (1944) para Ising 2D
ONSAGER_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.26918531
EXACT_BETA = 0.125  # 1/8


def _load_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T, M, S = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            T.append(float(row["T"]))
            M.append(float(row["M_mean"]))
            S.append(float(row["M_std"]))
    return np.array(T), np.array(M), np.array(S)


def plot_magnetization(csv_path: str, out_path: str) -> None:
    """Curva M(T) con barras de error y línea vertical en Tc de Onsager."""
    import matplotlib.pyplot as plt

    T, M, S = _load_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        T, M, yerr=S, fmt="o-", markersize=4,
        color="#7c5cff", ecolor="#9aa3b8", elinewidth=0.8, capsize=2,
        label=r"simulated $\langle |M| \rangle$",
    )
    ax.axvline(
        ONSAGER_TC, color="#f472b6", linestyle="--", linewidth=1.6,
        label=fr"$T_c^{{\mathrm{{Onsager}}}} \approx {ONSAGER_TC:.4f}$",
    )
    ax.set_xlabel(r"Temperature $T$ (in units of $J/k_B$)", fontsize=12)
    ax.set_ylabel(r"Magnetization $\langle |M| \rangle$", fontsize=12)
    ax.set_title("2D Ising model phase transition", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_posteriors(trace_path: str, out_path: str) -> None:
    """Distribuciones posteriores de Tc y β con los valores exactos superpuestos."""
    import arviz as az
    import matplotlib.pyplot as plt

    trace = az.from_netcdf(trace_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    az.plot_posterior(
        trace, var_names=["Tc"], ax=axes[0], hdi_prob=0.95,
        ref_val=ONSAGER_TC, color="#7c5cff",
    )
    axes[0].set_title(fr"Posterior of $T_c$ — exact: {ONSAGER_TC:.4f}", fontsize=12)

    az.plot_posterior(
        trace, var_names=["beta"], ax=axes[1], hdi_prob=0.95,
        ref_val=EXACT_BETA, color="#22d3ee",
    )
    axes[1].set_title(fr"Posterior of $\beta$ — exact: {EXACT_BETA}", fontsize=12)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_trace(trace_path: str, out_path: str) -> None:
    """Trace plots para diagnóstico visual de convergencia."""
    import arviz as az
    import matplotlib.pyplot as plt

    trace = az.from_netcdf(trace_path)
    axes = az.plot_trace(trace, var_names=["Tc", "beta", "sigma"], figsize=(12, 6))
    fig = axes.ravel()[0].figure
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_all(
    csv_path: str = "data/magnetization.csv",
    trace_path: str = "data/trace.nc",
    out_dir: str = "results",
) -> None:
    out = Path(out_dir)
    plot_magnetization(csv_path, str(out / "magnetization.png"))
    plot_posteriors(trace_path, str(out / "posteriors.png"))
    plot_trace(trace_path, str(out / "trace.png"))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="data/magnetization.csv")
    parser.add_argument("--trace", type=str, default="data/trace.nc")
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()
    plot_all(args.csv, args.trace, args.out_dir)


if __name__ == "__main__":
    _main()
