"""Sensitivity of the (Tc, β) posterior to the modelling choices.

Fits the same M(T) data under every combination of

* fit window: all temperatures, or only T < 2.25 (the ordered side, away
  from the finite-size rounding of the transition and the 1/N tail above it);
* likelihood: homoscedastic (one σ for every point) or heteroscedastic
  (point k has variance M_err[k]² + σ², with M_err its Monte Carlo
  standard error);
* β prior: Normal(0.12, 0.05), centred near the exact 1/8, or flat on
  (0.01, 0.5),

and writes posterior mean, sd and 95% HDI of Tc, β and σ, with R̂ and ESS,
to results/model_comparison.json.

Two further checks go in the same file:

* the data against Onsager's exact infinite-lattice magnetisation
  M(T) = (1 − sinh(2/T)^−4)^(1/8) on the ordered side, which tells apart a
  problem with the simulation from a problem with the fitted functional form;
* a fit with a free amplitude, M = B (1 − T/Tc)^β (flat priors on Tc, β, B),
  since the leading asymptotic form of the 2D Ising magnetisation has
  B ≈ 1.22, not 1;
* fits with the amplitude fixed at its exact value A = (8√2/Tc)^(1/8)
  = 1.2224 (expanding Onsager's M to leading order in 1 − T/Tc), so the
  known prefactor is not a free parameter, plus the largest deviation of
  A (1 − T/Tc)^(1/8) from Onsager on T ∈ [1.5, Tc) for A = 1 and A = 1.2224;
* the leading term with A fixed plus the first correction of Onsager's
  expansion, M = A t^β (1 + b t) with t = 1 − T/Tc, on the ordered side.

Usage:
    python -m scripts.model_comparison --csv data/magnetization.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from src.bayesian import _to_inferencedata, load_data, sample_mh_numpy

TC_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))
# M ≈ A (1 − T/Tc)^(1/8): sinh(2/T) − 1 ≈ (2√2/Tc)(1 − T/Tc) and 1 − sinh^−4 ≈ 4 (sinh − 1)
ONSAGER_AMPLITUDE = (8.0 * np.sqrt(2.0) / TC_EXACT) ** 0.125


def fit(T, M, M_err, *, likelihood, beta_prior, draws, tune, chains, seed,
        amplitude: float = 1.0) -> dict:
    import arviz as az

    samples = sample_mh_numpy(
        T, M, draws=draws, tune=tune, chains=chains, seed=seed,
        M_err=M_err if likelihood == "heteroscedastic" else None,
        beta_prior=beta_prior, amplitude=amplitude,
    )
    idata = _to_inferencedata(samples)
    s = az.summary(idata, var_names=["Tc", "beta", "sigma"], hdi_prob=0.95)
    out = {}
    for p in ["Tc", "beta", "sigma"]:
        out[p] = {
            "mean": float(s.loc[p, "mean"]), "sd": float(s.loc[p, "sd"]),
            "hdi_95": [float(s.loc[p, "hdi_2.5%"]), float(s.loc[p, "hdi_97.5%"])],
            "r_hat": float(s.loc[p, "r_hat"]), "ess_bulk": float(s.loc[p, "ess_bulk"]),
        }
    return out


def onsager_magnetization(T: np.ndarray) -> np.ndarray:
    """Exact spontaneous magnetisation of the infinite 2D Ising lattice (J = 1)."""
    T = np.asarray(T, dtype=float)
    out = np.zeros_like(T)
    below = T < TC_EXACT
    out[below] = (1.0 - np.sinh(2.0 / T[below]) ** -4) ** 0.125
    return out


def _log_post_amplitude(th, T, M, M_err):
    Tc, beta, B, sigma = th
    if not (1.0 < Tc < 4.0 and 0.01 < beta < 0.5 and 0.3 < B < 3.0 and sigma > 0):
        return -np.inf
    pred = np.where(T < Tc, B * np.clip(1.0 - T / Tc, 1e-12, 1.0) ** beta, 0.0)
    var = M_err ** 2 + sigma ** 2
    return -0.5 * np.sum((M - pred) ** 2 / var + np.log(var)) - 0.5 * (sigma / 0.1) ** 2


def _random_walk(log_post, init, step, names, *, draws, tune, chains, seed,
                 adapt: bool = False) -> dict:
    """Random-walk MH thinned by 8 (correlated parameters); summary per parameter.

    With `adapt`, the second half of tuning proposes from the covariance of the
    first half (scaled by 2.38²/d, Haario et al. 2001), which strongly
    correlated posteriors need; the kept draws use that fixed proposal."""
    import arviz as az

    rng = np.random.default_rng(seed)
    chains_out = np.empty((chains, draws, len(names)))
    for c in range(chains):
        th = init(rng)
        lp = log_post(th)
        chol, history = None, []
        for i in range(-tune * 8, draws * 8):        # thin by 8: four correlated parameters
            if adapt and i < 0:
                history.append(th)
                if i == -tune * 4:
                    cov = np.cov(np.array(history).T) * 2.38 ** 2 / len(th)
                    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(len(th)))
            prop = th + (chol @ rng.normal(0.0, 1.0, len(th)) if chol is not None
                         else rng.normal(0.0, step))
            lp_new = log_post(prop)
            if np.log(rng.random() + 1e-300) < lp_new - lp:
                th, lp = prop, lp_new
            if i >= 0 and i % 8 == 0:
                chains_out[c, i // 8] = th
    idata = _to_inferencedata({n: chains_out[:, :, k] for k, n in enumerate(names)})
    s = az.summary(idata, var_names=names, hdi_prob=0.95)
    return {p: {"mean": float(s.loc[p, "mean"]), "sd": float(s.loc[p, "sd"]),
                "hdi_95": [float(s.loc[p, "hdi_2.5%"]), float(s.loc[p, "hdi_97.5%"])],
                "r_hat": float(s.loc[p, "r_hat"]), "ess_bulk": float(s.loc[p, "ess_bulk"])}
            for p in names}


def fit_amplitude(T, M, M_err, *, draws, tune, chains, seed) -> dict:
    """Random-walk MH for the four-parameter model (heteroscedastic, flat priors)."""
    return _random_walk(
        lambda th: _log_post_amplitude(th, T, M, M_err),
        lambda rng: np.array([2.3 + 0.05 * rng.random(), 0.12, 1.1, 0.02]),
        np.array([0.01, 0.01, 0.02, 0.005]), ["Tc", "beta", "B", "sigma"],
        draws=draws, tune=tune, chains=chains, seed=seed)


def _correction_mean(Tc, beta, b, T):
    t = np.clip(1.0 - T / Tc, 1e-12, 1.0)
    return np.where(T < Tc, ONSAGER_AMPLITUDE * t ** beta * (1.0 + b * t), 0.0)


def _log_post_correction(th, T, M, M_err):
    # No discrepancy term σ: on T < 2.25 the O(t²) remainder of Onsager's
    # expansion is at most one Monte Carlo standard error per point, so the
    # Monte Carlo errors alone are the noise model.
    Tc, beta, b = th
    if not (1.0 < Tc < 4.0 and 0.01 < beta < 0.5 and -2.0 < b < 2.0):
        return -np.inf
    return -0.5 * np.sum((M - _correction_mean(Tc, beta, b, T)) ** 2 / M_err ** 2)


def fit_correction(T, M, M_err, *, draws, tune, chains, seed) -> dict:
    """M = A t^β (1 + b t), t = 1 − T/Tc, A = 1.2224 fixed: the leading term
    plus the first correction of Onsager's expansion (flat priors on Tc, β, b;
    Gaussian likelihood with the Monte Carlo errors)."""
    r = _random_walk(
        lambda th: _log_post_correction(th, T, M, M_err),
        lambda rng: np.array([2.3 + 0.05 * rng.random(), 0.12, -0.2]),
        np.array([0.005, 0.0007, 0.002]), ["Tc", "beta", "b"],
        draws=draws, tune=tune, chains=chains, seed=seed, adapt=True)
    resid = (M - _correction_mean(r["Tc"]["mean"], r["beta"]["mean"], r["b"]["mean"], T)) / M_err
    r["chi2"], r["dof"] = float(np.sum(resid ** 2)), int(T.size - 3)
    return r


def truncation_remainder(T: np.ndarray) -> tuple[float, np.ndarray]:
    """Best linear b for M_Onsager / (A t^(1/8)) − 1 on the window, and
    |M_Onsager − A t^(1/8) (1 + b t)| at each T: what the model leaves out."""
    t_max = 1.0 - T.min() / TC_EXACT
    tt = np.linspace(1e-4, t_max, 400)
    b = np.polyfit(tt, onsager_magnetization(TC_EXACT * (1.0 - tt)) / (ONSAGER_AMPLITUDE * tt ** 0.125) - 1.0, 1)[0]
    t = 1.0 - T / TC_EXACT
    return float(b), np.abs(onsager_magnetization(T) - ONSAGER_AMPLITUDE * t ** 0.125 * (1.0 + b * t))


def main(csv: str, out: str, draws: int, tune: int, chains: int, seed: int) -> None:
    T, M, M_err = load_data(csv)
    if M_err is None:
        raise SystemExit("the CSV has no M_err column; re-run the simulation")
    rows = []
    for window, likelihood, beta_prior in itertools.product(
            ["all", "T<2.25"], ["homoscedastic", "heteroscedastic"], ["normal", "flat"]):
        keep = T < 2.25 if window == "T<2.25" else np.ones_like(T, dtype=bool)
        print(f"\n[{window} | {likelihood} | beta {beta_prior}]  {keep.sum()} points", flush=True)
        r = fit(T[keep], M[keep], M_err[keep], likelihood=likelihood, beta_prior=beta_prior,
                draws=draws, tune=tune, chains=chains, seed=seed)
        r.update(window=window, likelihood=likelihood, beta_prior=beta_prior,
                 n_points=int(keep.sum()))
        print(f"  Tc = {r['Tc']['mean']:.3f} ± {r['Tc']['sd']:.3f}  "
              f"beta = {r['beta']['mean']:.3f} ± {r['beta']['sd']:.3f} "
              f"HDI {r['beta']['hdi_95']}  sigma = {r['sigma']['mean']:.4f}  "
              f"R̂max {max(r[p]['r_hat'] for p in ['Tc', 'beta', 'sigma']):.2f}", flush=True)
        rows.append(r)

    ordered = T < 2.25
    dev = M[ordered] - onsager_magnetization(T[ordered])
    onsager = {
        "T": T[ordered].tolist(), "M": M[ordered].tolist(), "M_err": M_err[ordered].tolist(),
        "M_onsager": onsager_magnetization(T[ordered]).tolist(),
        "max_abs_deviation": float(np.abs(dev).max()),
        "max_deviation_in_standard_errors": float(np.abs(dev / M_err[ordered]).max()),
        "power_law_exact_params_max_abs_deviation":
            float(np.abs(M[ordered] - (1.0 - T[ordered] / TC_EXACT) ** 0.125).max()),
    }
    T_dense = np.linspace(1.5, TC_EXACT, 2001)[:-1]
    for name, a in [("amplitude_1", 1.0), ("amplitude_exact", ONSAGER_AMPLITUDE)]:
        onsager[f"power_law_{name}_vs_onsager_max_abs_deviation_T_1.5_to_Tc"] = float(np.abs(
            a * (1.0 - T_dense / TC_EXACT) ** 0.125 - onsager_magnetization(T_dense)).max())
    print(f"\n[Onsager, T < 2.25] max |M - M_exact| = {onsager['max_abs_deviation']:.4f} "
          f"({onsager['max_deviation_in_standard_errors']:.1f} standard errors); "
          f"pure power law with exact Tc, beta is off by up to "
          f"{onsager['power_law_exact_params_max_abs_deviation']:.3f}", flush=True)

    amplitude = []
    for window in ["T<2.25", "all"]:
        keep = T < 2.25 if window == "T<2.25" else np.ones_like(T, dtype=bool)
        r = fit_amplitude(T[keep], M[keep], M_err[keep], draws=draws, tune=tune,
                          chains=chains, seed=seed)
        r.update(window=window, n_points=int(keep.sum()))
        print(f"[amplitude | {window}] Tc = {r['Tc']['mean']:.3f} HDI {r['Tc']['hdi_95']}  "
              f"beta = {r['beta']['mean']:.3f} HDI {r['beta']['hdi_95']}  "
              f"B = {r['B']['mean']:.3f}  R̂max {max(r[p]['r_hat'] for p in ['Tc', 'beta', 'B', 'sigma']):.2f}",
              flush=True)
        amplitude.append(r)

    fixed = []
    for window, beta_prior in itertools.product(["all", "T<2.25"], ["normal", "flat"]):
        keep = T < 2.25 if window == "T<2.25" else np.ones_like(T, dtype=bool)
        r = fit(T[keep], M[keep], M_err[keep], likelihood="heteroscedastic",
                beta_prior=beta_prior, draws=draws, tune=tune, chains=chains, seed=seed,
                amplitude=ONSAGER_AMPLITUDE)
        r.update(window=window, likelihood="heteroscedastic", beta_prior=beta_prior,
                 n_points=int(keep.sum()))
        print(f"[A = {ONSAGER_AMPLITUDE:.4f} | {window} | beta {beta_prior}] "
              f"Tc = {r['Tc']['mean']:.3f} HDI {r['Tc']['hdi_95']}  "
              f"beta = {r['beta']['mean']:.3f} HDI {r['beta']['hdi_95']}  "
              f"sigma = {r['sigma']['mean']:.4f}  "
              f"R̂max {max(r[p]['r_hat'] for p in ['Tc', 'beta', 'sigma']):.2f}", flush=True)
        fixed.append(r)

    correction = []
    for window, cut in [("T<2.25", 2.25), ("T<2.1", 2.1)]:
        keep = T < cut
        r = fit_correction(T[keep], M[keep], M_err[keep], draws=draws, tune=tune,
                           chains=chains, seed=seed)
        b_exact, rem = truncation_remainder(T[keep])
        r.update(window=window, n_points=int(keep.sum()), b_onsager=b_exact,
                 max_remainder_in_standard_errors=float((rem / M_err[keep]).max()))
        print(f"[A fixed + correction | {window}] Tc = {r['Tc']['mean']:.3f} HDI {r['Tc']['hdi_95']}  "
              f"beta = {r['beta']['mean']:.4f} HDI {r['beta']['hdi_95']}  "
              f"b = {r['b']['mean']:.3f} (Onsager {b_exact:.3f})  chi2/dof {r['chi2']:.1f}/{r['dof']}  "
              f"remainder <= {r['max_remainder_in_standard_errors']:.2f} SE  "
              f"R̂max {max(r[p]['r_hat'] for p in ['Tc', 'beta', 'b']):.2f}", flush=True)
        correction.append(r)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"csv": csv, "Tc_exact": TC_EXACT, "beta_exact": 0.125, "fits": rows,
                   "onsager_check": onsager, "free_amplitude_fits": amplitude,
                   "onsager_amplitude": ONSAGER_AMPLITUDE, "fixed_amplitude_fits": fixed,
                   "correction_to_scaling_fits": correction}, fh, indent=2)
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/magnetization.csv")
    p.add_argument("--out", default="results/model_comparison.json")
    p.add_argument("--draws", type=int, default=50_000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.csv, a.out, a.draws, a.tune, a.chains, a.seed)
