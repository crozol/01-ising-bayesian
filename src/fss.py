"""Finite-size-scaling data: magnetisation moments and the Binder cumulant.

For each lattice size L and temperature T near the transition, a long
Metropolis chain records the magnetisation per site m after every sweep, and
the script stores

    <|m|>                       order parameter
    U = 1 - <m^4> / (3 <m^2>^2) Binder cumulant (Binder 1981)

with block-jackknife errors. U is dimensionless and, to leading order, a
function of (T - Tc) L^(1/ν) only, so the curves U(T) for different L cross
at Tc; <|m|> scales as L^(-β/ν) times a function of the same variable. Both
are fitted in src/fss_fit.py.

Metropolis decorrelates slowly near Tc (autocorrelation time ~ L^2.17), so the
chains here are much longer than in the M(T) sweep: 20 000 burn-in and
200 000 measured sweeps per (L, T) by default.

Usage:
    python -m src.fss --sizes 8 16 32 64 --out data/fss.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .metropolis import _seed, _sweep, njit


@njit(cache=True)
def _chain(lattice: np.ndarray, T: float, n_therm: int, n_measure: int) -> np.ndarray:
    """Burn in, then return the signed magnetisation per site after each sweep."""
    for _ in range(n_therm):
        _sweep(lattice, T, 1.0)
    out = np.empty(n_measure)
    n2 = lattice.size
    for k in range(n_measure):
        _sweep(lattice, T, 1.0)
        s = 0
        for v in lattice.flat:
            s += v
        out[k] = s / n2
    return out


def _jackknife(m: np.ndarray, n_blocks: int, stat) -> tuple[float, float]:
    """Statistic of the full series and its block-jackknife standard error."""
    bs = len(m) // n_blocks
    m = m[: bs * n_blocks]
    full = stat(m)
    blocks = m.reshape(n_blocks, bs)
    loo = np.array([stat(np.delete(blocks, b, axis=0).ravel()) for b in range(n_blocks)])
    err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((loo - loo.mean()) ** 2))
    return float(full), float(err)


def _abs_m(m):
    return np.abs(m).mean()


def _binder(m):
    m2 = (m ** 2).mean()
    return 1.0 - (m ** 4).mean() / (3.0 * m2 ** 2)


def measure(L: int, T: float, *, n_therm: int, n_measure: int, seed: int,
            n_blocks: int = 50) -> dict:
    _seed(seed)
    lattice = np.ones((L, L), dtype=np.int8)       # cold start, as in the M(T) sweep
    m = _chain(lattice, T, n_therm, n_measure)
    M_abs, M_abs_err = _jackknife(m, n_blocks, _abs_m)
    U, U_err = _jackknife(m, n_blocks, _binder)
    return {"L": L, "T": float(T), "M_abs": M_abs, "M_abs_err": M_abs_err,
            "U": U, "U_err": U_err, "n_measure": n_measure}


def run(sizes, temps, *, n_therm: int, n_measure: int, seed: int, out: str) -> None:
    rows = []
    fields = ["L", "T", "M_abs", "M_abs_err", "U", "U_err", "n_measure"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for L in sizes:
        t0 = time.time()
        for i, T in enumerate(temps):
            r = measure(L, T, n_therm=n_therm, n_measure=n_measure, seed=seed + 1000 * L + i)
            rows.append(r)
            print(f"  L={L:4d} T={T:.4f}  <|m|>={r['M_abs']:.4f}±{r['M_abs_err']:.4f}  "
                  f"U={r['U']:.4f}±{r['U_err']:.4f}", flush=True)
            with open(out, "w", newline="") as fh:     # rewrite after every point
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        print(f"[L={L}] {time.time() - t0:.0f} s", flush=True)
    print(f"[ok] {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--t-min", type=float, default=2.20)
    p.add_argument("--t-max", type=float, default=2.34)
    p.add_argument("--n-temps", type=int, default=15)
    p.add_argument("--n-therm", type=int, default=20_000)
    p.add_argument("--n-measure", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/fss.csv")
    a = p.parse_args()
    run(a.sizes, np.linspace(a.t_min, a.t_max, a.n_temps),
        n_therm=a.n_therm, n_measure=a.n_measure, seed=a.seed, out=a.out)
