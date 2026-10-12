"""Systematic uncertainty of the finite-size-scaling fit: refit the same data
with different lattice-size sets and temperature windows and tabulate the
posterior of Tc, ν, β and U*.

Usage:
    python -m scripts.fss_systematics --csv data/fss.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.fss_fit import load, sample, summarize

CONFIGS = [
    ([16, 32, 64], 2.23, 2.31),      # default
    ([8, 16, 32, 64], 2.23, 2.31),
    ([16, 32, 64], 2.20, 2.34),
    ([8, 16, 32, 64], 2.20, 2.34),
    ([16, 32, 64], 2.245, 2.295),
    ([32, 64], 2.23, 2.31),
]


def main(csv: str, out: str, draws: int, tune: int) -> None:
    rows = []
    for sizes, lo, hi in CONFIGS:
        d = load(csv, sizes, lo, hi)
        print(f"\n== sizes {sizes}, T in [{lo}, {hi}], {len(d['L'])} points", flush=True)
        smp, cf = sample(d, draws=draws, tune=tune)
        r = summarize(smp, cf, d, sizes, str(Path(out).with_suffix("")) + "_tmp.json")
        rows.append({"sizes": sizes, "t_min": lo, "t_max": hi, "n_points": r["n_points"],
                     "posterior": r["posterior"]})
    Path(str(Path(out).with_suffix("")) + "_tmp.json").unlink(missing_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n sizes            window          Tc                 nu               beta             U*")
    for r in rows:
        p = r["posterior"]
        f = lambda k: f"{p[k]['mean']:.3f} [{p[k]['hdi_95'][0]:.3f},{p[k]['hdi_95'][1]:.3f}]"  # noqa: E731
        print(f" {str(r['sizes']):16s} [{r['t_min']},{r['t_max']}]  {f('Tc')}  {f('nu')}  {f('beta')}  {f('U0')}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/fss.csv")
    p.add_argument("--out", default="results/fss_systematics.json")
    p.add_argument("--draws", type=int, default=10_000)
    p.add_argument("--tune", type=int, default=10_000)
    a = p.parse_args()
    main(a.csv, a.out, a.draws, a.tune)
