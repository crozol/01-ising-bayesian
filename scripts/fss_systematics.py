"""Systematic uncertainty of the finite-size-scaling fit (Section 12).

By default the script tabulates the classical weighted least-squares fit
(src.fss_fit.chi2_fit: minimum, errors from the curvature of χ², χ²/dof and
Q) for several lattice-size sets and windows of fixed width in the scaling
variable, |T − T*| L^(1/ν₀) ≤ x_max (src.fss_fit.scaling_window), plus cubic
and quartic scaling functions on the accepted window and the next wider one
(ORDER_CHECKS). The goodness-of-fit test picks the range; the shifts of the
variations that also pass it give the systematic error. It runs in seconds.

--mcmc instead refits every window of X_CONFIGS with the Bayesian model and
tabulates the posterior (about half an hour).

Usage:
    python -m scripts.fss_systematics
    python -m scripts.fss_systematics --mcmc --out results/fss_systematics_mcmc.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fss_fit import chi2_fit, load, preliminary, sample, scaling_window, summarize

X_CONFIGS = [
    ([16, 32, 64, 128], 1.0),
    ([16, 32, 64, 128], 1.5),
    ([32, 64, 128], 1.0),            # accepted fit (Q = 0.22)
    ([32, 64, 128], 1.5),
    ([32, 64, 128], 2.0),
    ([64, 128], 1.0),
    ([64, 128], 1.5),
]

ORDER_CHECKS = [([32, 64, 128], x_max, order) for x_max in (1.0, 1.5) for order in (3, 4)]


def _window(csv: str, sizes, x_max: float):
    full = load(csv, sizes)
    t_star, nu0 = preliminary(full, sizes)
    return scaling_window(full, x_max, t_star, nu0), {"x_max": x_max, "t_star": t_star, "nu0": nu0}


def least_squares(csv: str, out: str) -> None:
    rows = []
    for sizes, x_max, order in [(s, x, 2) for s, x in X_CONFIGS] + ORDER_CHECKS:
        d, window = _window(csv, sizes, x_max)
        rows.append({"sizes": sizes, **window, "n_points": int(len(d["L"])),
                     **chi2_fit(d, order=order)})
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(" sizes            |x|<=  order  points   chi2/dof        Q        "
          "Tc                 nu              beta             U*")
    for r in rows:
        print(f" {str(r['sizes']):16s} {r['x_max']:<5} {r['order']:^5}  {r['n_points']:>4}  "
              f"{r['chi2']:8.1f}/{r['dof']:<3d}  {r['Q']:8.2g}  "
              f"{r['Tc']:.5f}({r['Tc_err']:.5f})  {r['nu']:.4f}({r['nu_err']:.4f})  "
              f"{r['beta']:.4f}({r['beta_err']:.4f})  {r['U0']:.5f}")
    print(f"[ok] {out}")


def mcmc(csv: str, out: str, draws: int, tune: int) -> None:
    rows = []
    tmp = str(Path(out).with_suffix("")) + "_tmp.json"
    for sizes, x_max in X_CONFIGS:
        d, window = _window(csv, sizes, x_max)
        print(f"\n== sizes {sizes}, {window}, {len(d['L'])} points", flush=True)
        smp, cf = sample(d, draws=draws, tune=tune)
        r = summarize(smp, cf, d, sizes, tmp)
        rows.append({"sizes": sizes, **window, "n_points": r["n_points"],
                     "posterior": r["posterior"], "least_squares": r["least_squares"]})
    Path(tmp).unlink(missing_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n sizes            window   Tc                 nu               beta             U*")
    for r in rows:
        p = r["posterior"]
        f = lambda k: f"{p[k]['mean']:.4f} [{p[k]['hdi_95'][0]:.4f},{p[k]['hdi_95'][1]:.4f}]"  # noqa: E731
        print(f" {str(r['sizes']):16s} |x|<={r['x_max']:<4} {f('Tc')}  {f('nu')}  {f('beta')}  {f('U0')}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/fss_wolff.csv")
    p.add_argument("--out", default="results/fss_systematics.json")
    p.add_argument("--mcmc", action="store_true",
                   help="Bayesian refit of every window instead of least squares")
    p.add_argument("--draws", type=int, default=10_000)
    p.add_argument("--tune", type=int, default=10_000)
    a = p.parse_args()
    if a.mcmc:
        mcmc(a.csv, a.out, a.draws, a.tune)
    else:
        least_squares(a.csv, a.out)
