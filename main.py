"""Pipeline completo: simulación Metropolis → inferencia bayesiana → figuras.

Uso:
    python main.py                     # todo el pipeline con defaults
    python -m src.metropolis --help    # solo simulación
    python -m src.bayesian   --help    # solo inferencia
    python -m src.plots      --help    # solo figuras
"""

from __future__ import annotations

from src.bayesian import run_full_inference
from src.metropolis import run_full_simulation
from src.plots import plot_all


def main() -> None:
    print("=" * 60)
    print("STEP 1/3 · Simulación Metropolis del modelo de Ising 2D")
    print("=" * 60)
    run_full_simulation(out_csv="data/magnetization.csv")

    print("\n" + "=" * 60)
    print("STEP 2/3 · Inferencia bayesiana (backend NumPy MH por defecto)")
    print("=" * 60)
    run_full_inference(csv_in="data/magnetization.csv", trace_out="data/trace.nc")

    print("\n" + "=" * 60)
    print("STEP 3/3 · Generación de figuras")
    print("=" * 60)
    plot_all(
        csv_path="data/magnetization.csv",
        trace_path="data/trace.nc",
        out_dir="results",
    )

    print("\n[done] ver resultados en ./results/")


if __name__ == "__main__":
    main()
