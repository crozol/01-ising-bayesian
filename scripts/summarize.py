"""Extract the inferred Tc, β values from the MCMC trace for README embedding."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np


ONSAGER_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))
EXACT_BETA = 0.125


def main(trace_path: str = "data/trace.nc", out_path: str = "data/summary.json") -> None:
    trace = az.from_netcdf(trace_path)
    summary = az.summary(trace, var_names=["Tc", "beta", "sigma"], hdi_prob=0.95)
    print(summary)

    tc_mean = float(summary.loc["Tc", "mean"])
    tc_sd = float(summary.loc["Tc", "sd"])
    tc_hdi_low = float(summary.loc["Tc", "hdi_2.5%"])
    tc_hdi_high = float(summary.loc["Tc", "hdi_97.5%"])
    beta_mean = float(summary.loc["beta", "mean"])
    beta_sd = float(summary.loc["beta", "sd"])
    beta_hdi_low = float(summary.loc["beta", "hdi_2.5%"])
    beta_hdi_high = float(summary.loc["beta", "hdi_97.5%"])
    r_hat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())

    out = {
        "Tc": {
            "mean": tc_mean, "sd": tc_sd,
            "hdi_95": [tc_hdi_low, tc_hdi_high],
            "exact": ONSAGER_TC,
            "diff_exact": tc_mean - ONSAGER_TC,
        },
        "beta": {
            "mean": beta_mean, "sd": beta_sd,
            "hdi_95": [beta_hdi_low, beta_hdi_high],
            "exact": EXACT_BETA,
            "diff_exact": beta_mean - EXACT_BETA,
        },
        "diagnostics": {
            "r_hat_max": r_hat_max,
            "ess_bulk_min": ess_min,
        },
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ok] summary saved to {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
