"""Corrections to scaling in the finite-size-scaling fit (Section 12).

The χ² test of Section 12 rejects every fit that includes L = 16 and every window wider than
|x| ≤ 1, and the systematic shifts of β and ν among the accepted variations point towards the
exact values as L_min grows. Both suggest corrections to scaling that the leading-order forms
leave out. This script asks whether one explicit correction term accounts for them. Each window
of scripts/fss_systematics.py is refitted by weighted least squares (src.fss_fit.chi2_fit):

* with a term c · L^(-ω) added to both scaling forms, each with its own amplitude, for
  ω = 1, the order of the corrections from the nonlinear dependence of the thermal scaling
  field on T, L^(-1/ν) (Aharony & Fisher 1983), and ω = 2, the leading correction from
  irrelevant operators in the two-dimensional Ising model (Caselle et al. 2002);
* with K = 1/T in place of T in the scaling variable, x = (K − Kc) L^(1/ν). The two choices
  differ at second order in T − Tc, so the shift this produces measures how much the result
  depends on that nonlinearity.

It runs in seconds and writes results/scaling_corrections.json.

Usage:
    python -m scripts.scaling_corrections
"""

from __future__ import annotations

import argparse
import json

from scripts.fss_systematics import _window
from src.fss_fit import chi2_fit

CONFIGS = [
    ([16, 32, 64, 128], 1.0),
    ([16, 32, 64, 128], 1.5),
    ([32, 64, 128], 1.0),            # accepted fit of Section 12 (Q = 0.22)
    ([32, 64, 128], 1.5),
    ([32, 64, 128], 2.0),
]
OMEGAS = (1.0, 2.0)


def fit_in_K(d, start=(2.27, 1.0, 0.12), order=2):
    """chi2_fit with x = (K − Kc) L^(1/ν), K = 1/T. The fit runs on T' = −1/T, which
    increases with T as T does, so that Tc' = −1/Tc; Tc and its error are mapped back
    (dTc = Tc² dTc')."""
    dk = dict(d)
    dk["T"] = -1.0 / d["T"]
    r = chi2_fit(dk, start=(-1.0 / start[0],) + tuple(start[1:]), order=order)
    tc = -1.0 / r["Tc"]
    r["Tc_err"] = float(tc ** 2 * r["Tc_err"])
    r["Tc"] = float(tc)
    r["variable"] = "K = 1/T"
    return r


def main(csv: str = "data/fss_wolff.csv", out: str = "results/scaling_corrections.json") -> list:
    rows = []
    for sizes, x_max in CONFIGS:
        d, window = _window(csv, sizes, x_max)
        base = {"sizes": sizes, **window, "n_points": int(len(d["L"]))}
        rows.append({**base, "variant": "leading order", **chi2_fit(d)})
        for om in OMEGAS:
            rows.append({**base, "variant": f"+ L^-{om:g}", **chi2_fit(d, omega=om)})
        rows.append({**base, "variant": "K = 1/T", **fit_in_K(d)})
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(" sizes            |x|<=  variant        points   chi2/dof        Q        "
          "Tc                 nu              beta")
    for r in rows:
        print(f" {str(r['sizes']):16s} {r['x_max']:<5} {r['variant']:13s}  {r['n_points']:>4}  "
              f"{r['chi2']:8.1f}/{r['dof']:<3d}  {r['Q']:8.2g}  "
              f"{r['Tc']:.5f}({r['Tc_err']:.5f})  {r['nu']:.4f}({r['nu_err']:.4f})  "
              f"{r['beta']:.4f}({r['beta_err']:.4f})")
    print(f"[ok] {out}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/fss_wolff.csv")
    p.add_argument("--out", default="results/scaling_corrections.json")
    a = p.parse_args()
    main(a.csv, a.out)
