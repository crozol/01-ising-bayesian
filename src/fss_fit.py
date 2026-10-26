"""Bayesian finite-size-scaling fit of Tc, ν and β from the Binder cumulant
and the magnetisation of several lattice sizes (data from src/fss.py).

Near Tc both observables depend on T and L only through the scaling variable
x = (T − Tc) L^(1/ν), up to corrections that shrink with L:

    U(T, L)     = U* + u1 x + u2 x²
    <|m|>(T, L) = L^(−β/ν) (g0 + g1 x + g2 x²)

The two are fitted jointly, so Tc and ν are shared. Point k carries its
block-jackknife error e_k; a discrepancy term per observable (σ_U, σ_M)
absorbs what the quadratic scaling functions and the neglected corrections to
scaling do not describe, as in the M(T) fit (variance e_k² + σ²).

Priors are flat on the physical parameters — Tc ∈ (2.0, 2.6), ν ∈ (0.5, 2),
β/ν ∈ (0, 0.5) — and on the six scaling-function coefficients, with
σ_U, σ_M ~ HalfNormal(0.01); none of the exact values (2.2692, 1, 1/8)
is encoded. The likelihood is linear in the six coefficients, so they are
integrated out analytically (Gaussian integral, flat prior), and only
(Tc, ν, β/ν, σ_U, σ_M) are sampled, with adaptive random-walk Metropolis
(Haario, Saksman & Tamminen 2001). The coefficients — U* among them — are
then drawn from their exact conditional Gaussian for each posterior sample.

Default fit: L = 16, 32, 64 and T in [2.23, 2.31]. The smallest lattice is
left out because corrections to scaling are largest there, and the window
is the widest in which the fitted discrepancy terms σ_U, σ_M are consistent
with zero (the quadratic scaling functions describe the data within their
errors); neither criterion looks at the exact values. scripts/fss_systematics.py
refits with other size sets and windows to show how much the estimates move.

A window of fixed width in T reaches |x| ∝ L^(1/ν) on the largest lattices,
where the quadratic scaling functions no longer hold. --x-max instead keeps
the points with |T − T*| L^(1/ν₀) ≤ x_max, the same part of the scaling
functions for every size. T* and ν₀ are model-free: T* is the U crossing of
the two largest sizes, and ν₀ follows from dU/dT|T* ∝ L^(1/ν).

As a model-free cross-check, the crossing temperature of U(T) for each pair
of consecutive sizes is located by linear interpolation, with a bootstrap
error from the jackknife errors of U.

Usage:
    python -m src.fss_fit --csv data/fss.csv
    python -m src.fss_fit --csv data/fss_wolff.csv --sizes 16 32 64 128 --x-max 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ["Tc", "nu", "b", "sigma_U", "sigma_M"]
COEFS = ["U0", "u1", "u2", "g0", "g1", "g2"]
TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))
BOUNDS = [(2.0, 2.6), (0.5, 2.0), (0.0, 0.5), (0.0, 1.0), (0.0, 1.0)]


def load(csv: str, sizes=None, t_min=None, t_max=None):
    d = np.genfromtxt(csv, delimiter=",", names=True)
    if sizes is not None:
        d = d[np.isin(d["L"], sizes)]
    if t_min is not None:
        d = d[d["T"] >= t_min - 1e-9]
    if t_max is not None:
        d = d[d["T"] <= t_max + 1e-9]
    return {k: np.asarray(d[k], dtype=float) for k in d.dtype.names}


def preliminary(d, sizes):
    """Model-free T* and ν₀ that place the scaling window: T* is the U crossing
    of the two largest sizes; each slope dU/dT at T* is a weighted straight line
    through the three grid temperatures nearest T*, and ν₀ comes from a
    weighted log-log fit of the slopes against L."""
    t_star = binder_crossings(d, sizes[-2:], n_boot=2)[0]["T_cross"]
    slopes, errs = [], []
    for L in sizes:
        T, U, e = (d[k][d["L"] == L] for k in ("T", "U", "U_err"))
        k = np.argsort(np.abs(T - t_star))[:3]
        c, cov = np.polyfit(T[k] - t_star, U[k], 1, w=1 / e[k], cov="unscaled")
        slopes.append(-c[0])
        errs.append(np.sqrt(cov[0, 0]))
    slopes, errs = np.array(slopes), np.array(errs)
    inv_nu = np.polyfit(np.log(sizes), np.log(slopes), 1, w=slopes / errs)[0]
    return float(t_star), float(1.0 / inv_nu)


def scaling_window(d, x_max, t_star, nu0):
    """Keep the points with |T − T*| L^(1/ν₀) ≤ x_max."""
    keep = np.abs((d["T"] - t_star) * d["L"] ** (1.0 / nu0)) <= x_max + 1e-9
    return {k: v[keep] for k, v in d.items()}


def _design(th, d):
    Tc, nu, b = th[:3]
    x = (d["T"] - Tc) * d["L"] ** (1.0 / nu)
    A = np.stack([np.ones_like(x), x, x ** 2], 1)
    return A, d["L"] ** (-b)


def _gaussian_block(A, y, v):
    """Weighted least squares for y = A c + noise(var v), and the log of the
    likelihood integrated over c with a flat prior."""
    Aw = A / v[:, None]
    F = A.T @ Aw                                   # Fisher matrix A^T V^-1 A
    c = np.linalg.solve(F, Aw.T @ y)
    r = y - A @ c
    _, logdet = np.linalg.slogdet(F)
    ll = -0.5 * (np.sum(r ** 2 / v) + np.sum(np.log(v)) + logdet)
    return ll, c, F


def log_post(th, d, return_blocks=False):
    for val, (lo, hi) in zip(th, BOUNDS):
        if not lo < val < hi:
            return (-np.inf, None) if return_blocks else -np.inf
    sU, sM = th[3], th[4]
    A, scale = _design(th, d)
    llU, cU, FU = _gaussian_block(A, d["U"], d["U_err"] ** 2 + sU ** 2)
    llM, cM, FM = _gaussian_block(A * scale[:, None], d["M_abs"], d["M_abs_err"] ** 2 + sM ** 2)
    lp = llU + llM - 0.5 * ((sU / 0.01) ** 2 + (sM / 0.01) ** 2)
    if return_blocks:
        return lp, ((cU, FU), (cM, FM))
    return lp


def sample(d, *, draws=10_000, tune=10_000, chains=4, seed=0, thin=2):
    rng = np.random.default_rng(seed)
    k = len(NAMES)
    out = np.empty((chains, draws, k))
    coefs = np.empty((chains, draws, len(COEFS)))
    for c in range(chains):
        th = np.array([2.27, 1.0, 0.12, 0.003, 0.003]) + rng.normal(0, [0.02, 0.1, 0.02, 0.0, 0.0])
        lp = log_post(th, d)
        cov = np.diag([0.005, 0.05, 0.01, 0.001, 0.001]) ** 2
        hist, acc = [], 0
        for i in range(tune + draws * thin):
            if i < tune and i >= 1000 and i % 250 == 0:       # adapt during tuning only
                H = np.array(hist[len(hist) // 2:])
                cov = np.cov(H.T) * 2.38 ** 2 / k + 1e-14 * np.eye(k)
            prop = rng.multivariate_normal(th, cov)
            lp_new = log_post(prop, d)
            if np.log(rng.random() + 1e-300) < lp_new - lp:
                th, lp = prop, lp_new
                acc += i >= tune
            if i < tune:
                hist.append(th.copy())
            elif (i - tune) % thin == 0:
                j = (i - tune) // thin
                out[c, j] = th
                _, ((cU, FU), (cM, FM)) = log_post(th, d, return_blocks=True)
                coefs[c, j, :3] = rng.multivariate_normal(cU, np.linalg.inv(FU))
                coefs[c, j, 3:] = rng.multivariate_normal(cM, np.linalg.inv(FM))
        print(f"  [chain {c + 1}/{chains}] acceptance {acc / (draws * thin):.1%}", flush=True)
    return out, coefs


def binder_crossings(d, sizes, n_boot=2000, seed=0):
    """Temperature where U(T, L1) = U(T, L2) for consecutive sizes, by linear
    interpolation between grid temperatures; error by parametric bootstrap."""
    rng = np.random.default_rng(seed)
    res = []
    for L1, L2 in zip(sizes[:-1], sizes[1:]):
        a, b = d["L"] == L1, d["L"] == L2
        shared = np.isin(d["T"].round(9), np.intersect1d(d["T"][a].round(9), d["T"][b].round(9)))
        a, b = a & shared, b & shared                  # a scaling window keeps fewer T at large L
        T = d["T"][a]
        assert np.allclose(T, d["T"][b]), "sizes must share the temperature grid"

        def cross(U1, U2):
            diff = U1 - U2
            s = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
            if len(s) == 0:
                return np.nan
            j = s[0]
            return T[j] - diff[j] * (T[j + 1] - T[j]) / (diff[j + 1] - diff[j])

        t0 = cross(d["U"][a], d["U"][b])
        boots = np.array([cross(d["U"][a] + rng.normal(0, d["U_err"][a]),
                                d["U"][b] + rng.normal(0, d["U_err"][b])) for _ in range(n_boot)])
        boots = boots[np.isfinite(boots)]
        res.append({"L1": int(L1), "L2": int(L2), "T_cross": float(t0),
                    "err": float(boots.std(ddof=1)) if len(boots) > 1 else None})
    return res


def summarize(samples, coefs, d, sizes, out_json, trace_out=None):
    import arviz as az
    import xarray as xr

    chains, draws, _ = samples.shape
    post = {n: samples[:, :, i] for i, n in enumerate(NAMES)}
    post.update({n: coefs[:, :, i] for i, n in enumerate(COEFS)})
    post["beta"] = post["nu"] * post["b"]
    ds = xr.Dataset({n: (("chain", "draw"), v) for n, v in post.items()},
                    coords={"chain": np.arange(chains), "draw": np.arange(draws)})
    idata = az.InferenceData(posterior=ds)
    if trace_out:
        Path(trace_out).parent.mkdir(parents=True, exist_ok=True)
        az.to_netcdf(idata, trace_out)
    s = az.summary(idata, var_names=["Tc", "nu", "b", "beta", "U0", "sigma_U", "sigma_M"],
                   hdi_prob=0.95, round_to=6)
    print(s.to_string(), flush=True)
    result = {"sizes": [int(L) for L in sizes], "n_points": int(len(d["L"])),
              "exact": {"Tc": TC_EXACT, "nu": 1.0, "b": 0.125, "beta": 0.125, "U0": 0.6107},
              "posterior": {p: {"mean": float(s.loc[p, "mean"]), "sd": float(s.loc[p, "sd"]),
                                "hdi_95": [float(s.loc[p, "hdi_2.5%"]), float(s.loc[p, "hdi_97.5%"])],
                                "r_hat": float(s.loc[p, "r_hat"]),
                                "ess_bulk": float(s.loc[p, "ess_bulk"])} for p in s.index},
              "binder_crossings": binder_crossings(d, sizes)}
    for c in result["binder_crossings"]:
        print(f"  U crossing L={c['L1']}/{c['L2']}: T = {c['T_cross']:.4f} ± {c['err']:.4f}")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[ok] {out_json}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/fss.csv")
    p.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--t-min", type=float, default=2.23)
    p.add_argument("--t-max", type=float, default=2.31)
    p.add_argument("--x-max", type=float, default=None,
                   help="window |T - T*| L^(1/nu0) <= x_max in place of --t-min/--t-max")
    p.add_argument("--draws", type=int, default=10_000)
    p.add_argument("--tune", type=int, default=10_000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/fss_summary.json")
    p.add_argument("--trace", default="data/fss_trace.nc")
    a = p.parse_args()
    if a.x_max is None:
        d = load(a.csv, a.sizes, a.t_min, a.t_max)
        window = f"T in [{a.t_min}, {a.t_max}]"
    else:
        d = load(a.csv, a.sizes)
        t_star, nu0 = preliminary(d, sorted(a.sizes))
        d = scaling_window(d, a.x_max, t_star, nu0)
        window = f"|T - {t_star:.4f}| L^(1/{nu0:.3f}) <= {a.x_max}"
    sizes = sorted(set(int(L) for L in d["L"]))
    print(f"[fss_fit] {len(d['L'])} points, sizes {sizes}, {window}", flush=True)
    smp, cf = sample(d, draws=a.draws, tune=a.tune, chains=a.chains, seed=a.seed)
    summarize(smp, cf, d, sizes, a.out, a.trace)
