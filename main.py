"""Pipeline completo: simulación Metropolis → inferencia bayesiana → escalamiento
de tamaño finito (Wolff) → figuras.

Uso:
    python main.py                     # todo el pipeline con defaults
    python main.py --reuse-fss         # reutiliza data/fss_wolff.csv (la simulación FSS tarda ~1 h)
    python -m src.metropolis --help    # solo simulación M(T)
    python -m src.bayesian   --help    # solo inferencia con un tamaño
    python -m src.fss        --help    # solo simulación FSS
    python -m src.fss_fit    --help    # solo ajuste FSS
    python -m src.plots      --help    # solo figuras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.bayesian import run_full_inference
from src.fss import run as run_fss
from src.fss_fit import (load as load_fss, preliminary, sample as sample_fss, scaling_window,
                         summarize as summarize_fss)
from src.metropolis import run_full_simulation
from src.plots import plot_all, plot_fss


def main(reuse_fss: bool = False) -> None:
    print("=" * 60)
    print("STEP 1/5 · Simulación Metropolis del modelo de Ising 2D (L = 32)")
    print("=" * 60)
    run_full_simulation(out_csv="data/magnetization.csv")

    print("\n" + "=" * 60)
    print("STEP 2/5 · Inferencia bayesiana con un solo tamaño")
    print("=" * 60)
    run_full_inference(csv_in="data/magnetization.csv", trace_out="data/trace.nc")

    print("\n" + "=" * 60)
    print("STEP 3/5 · Simulación Wolff para escalamiento de tamaño finito (L = 16…128)")
    print("=" * 60)
    if reuse_fss and Path("data/fss_wolff.csv").exists():
        print("[skip] reutilizando data/fss_wolff.csv")
    else:
        run_fss([16, 32, 64, 128], np.linspace(2.20, 2.34, 15), n_therm=20_000,
                n_measure=200_000, seed=0, out="data/fss_wolff.csv", algorithm="wolff")

    print("\n" + "=" * 60)
    print("STEP 4/5 · Ajuste bayesiano de escalamiento de tamaño finito")
    print("=" * 60)
    sizes = [32, 64, 128]                  # ventana |x| <= 1: la más amplia que pasa la prueba de chi2
    full = load_fss("data/fss_wolff.csv", sizes)
    t_star, nu0 = preliminary(full, sizes)
    d = scaling_window(full, 1.0, t_star, nu0)
    samples, coefs = sample_fss(d, seed=0)
    summarize_fss(samples, coefs, d, sizes, "data/fss_summary.json", "data/fss_trace.nc")

    print("\n" + "=" * 60)
    print("STEP 5/5 · Generación de figuras")
    print("=" * 60)
    plot_all(csv_path="data/magnetization.csv", trace_path="data/trace.nc", out_dir="results")
    plot_fss("data/fss_wolff.csv", "data/fss_trace.nc", "results/fss.png", sizes=sizes,
             x_window=(1.0, t_star, nu0))

    print("\n[done] ver resultados en ./results/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reuse-fss", action="store_true")
    main(p.parse_args().reuse_fss)
