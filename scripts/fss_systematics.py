"""Systematic uncertainty of the finite-size-scaling fit: refit the same data
with different lattice-size sets and temperature windows and tabulate the
posterior of Tc, ν, β and U*.

--x-windows refits with windows of fixed width in the scaling variable,
|T − T*| L^(1/ν₀) ≤ x_max (src.fss_fit.scaling_window), for the Wolff data.

Usage:
    python -m scripts.fss_systematics --csv data/fss.csv
    python -m scripts.fss_systematics --csv data/fss_wolff.csv --x-windows \
        --out results/fss_systematics_wolff.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fss_fit import load, preliminary, sample, scaling_window, summarize

CONFIGS = [
    ([16, 32, 64], 2.23, 2.31),      # default
    ([8, 16, 32, 64], 2.23, 2.31),
    ([16, 32, 64], 2.20, 2.34),
    ([8, 16, 32, 64], 2.20, 2.34),
    ([16, 32, 64], 2.245, 2.295),
    ([32, 64], 2.23, 2.31),
]

X_CONFIGS = [
    ([16, 32, 64, 128], 1.5),
    ([16, 32, 64, 128], 2.0),
    ([16, 32, 64, 128], 3.0),
    ([32, 64, 128], 1.5),
    ([32, 64, 128], 2.0),
    ([32, 64, 128], 3.0),
    ([64, 128], 3.0),
]


def _windows(csv: str, x_windows: bool):
    if not x_windows:
        for sizes, lo, hi in CONFIGS:
            yield sizes, load(csv, sizes, lo, hi), {"t_min": lo, "t_max": hi}
        return
    for sizes, x_max in X_CONFIGS:
        d = load(csv, sizes)
        t_star, nu0 = preliminary(d, sizes)
        yield sizes, scaling_window(d, x_max, t_star, nu0), \
            {"x_max": x_max, "t_star": t_star, "nu0": nu0}


def main(csv: str, out: str, draws: int, tune: int, x_windows: bool = False) -> None:
    rows = []
    for sizes, d, window in _windows(csv, x_windows):
        print(f"\n== sizes {sizes}, {window}, {len(d['L'])} points", flush=True)
        smp, cf = sample(d, draws=draws, tune=tune)
        r = summarize(smp, cf, d, sizes, str(Path(out).with_suffix("")) + "_tmp.json")
        rows.append({"sizes": sizes, **window, "n_points": r["n_points"],
                     "posterior": r["posterior"]})
    Path(str(Path(out).with_suffix("")) + "_tmp.json").unlink(missing_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n sizes            window          Tc                 nu               beta             U*")
    for r in rows:
        p = r["posterior"]
        f = lambda k: f"{p[k]['mean']:.3f} [{p[k]['hdi_95'][0]:.3f},{p[k]['hdi_95'][1]:.3f}]"  # noqa: E731
        w = f"|x|<={r['x_max']}" if "x_max" in r else f"[{r['t_min']},{r['t_max']}]"
        print(f" {str(r['sizes']):16s} {w:14s}  {f('Tc')}  {f('nu')}  {f('beta')}  {f('U0')}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/fss.csv")
    p.add_argument("--out", default="results/fss_systematics.json")
    p.add_argument("--draws", type=int, default=10_000)
    p.add_argument("--tune", type=int, default=10_000)
    p.add_argument("--x-windows", action="store_true",
                   help="windows of fixed width in the scaling variable (X_CONFIGS)")
    a = p.parse_args()
    main(a.csv, a.out, a.draws, a.tune, a.x_windows)
