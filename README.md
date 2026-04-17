# Bayesian Inference on the 2D Ising Model

An end-to-end pipeline that **infers the critical parameters of the 2D Ising model** — the critical temperature `Tc` and the critical exponent `β` — from data simulated with Monte Carlo, using Bayesian inference via PyMC. Results are quantitatively validated against Onsager's exact (1944) solution.

This project brings together three things rarely seen together in a junior ML portfolio: **real statistical physics**, **applied Bayesian statistics**, and **end-to-end scientific Python code**.

---

## Components

### 1 · Metropolis simulator of the 2D Ising model

From-scratch implementation of the Metropolis-Hastings algorithm to sample configurations from the canonical ensemble of a square lattice of ±1 spins with periodic boundary conditions.

- Acceptance rule based on the local energy change ΔE when flipping a spin, with probability `min(1, exp(−ΔE/T))`.
- Each sweep performs N² flip attempts — one per site on average.
- The hot kernel is decorated with `@numba.njit(cache=True)`, yielding a ~50× speedup over pure NumPy. A transparent fallback to pure Python is available if Numba is not installed.
- Observable measured: `<|M|> = <|(1/N²) · Σ sᵢⱼ|>`. Absolute value because at low T the system may spontaneously break symmetry toward +1 or −1 indistinguishably.

File: [`src/metropolis.py`](src/metropolis.py).

### 2 · Temperature sweep and dataset generation

Sweep over 30 temperatures in the range `T ∈ [1.5, 3.5]` (units of `J/k_B`), amply covering the theoretical critical temperature. For each T: 2000 thermalization sweeps + 3000 measurement sweeps, recording mean and standard deviation of `<|M|>`.

Output: a CSV with columns `T, M_mean, M_std` acting as the dataset for the inference phase.

### 3 · Probabilistic model in PyMC

Near the phase transition, the magnetization follows a power law:

```
M(T) ≈ (1 − T/Tc)^β     if T < Tc
M(T) = 0                if T ≥ Tc
```

Bayesian model in PyMC:

| Parameter | Prior | Justification |
|---|---|---|
| `Tc` | `Normal(μ=2.3, σ=0.3)` | centered near Onsager's value with wide uncertainty |
| `β` | `Normal(μ=0.12, σ=0.05)` | centered near the exact value 1/8 |
| `σ` | `HalfNormal(0.1)` | positive observation noise |

Gaussian likelihood on the observed magnetization, with the theoretical model prediction as the mean.

File: [`src/bayesian.py`](src/bayesian.py).

### 4 · MCMC sampling (NUTS) and diagnostics

NUTS (No-U-Turn Sampler) with 4 chains, 2000 samples + 1000 tuning, `target_accept=0.9`. Convergence is assessed via:

- **R̂ (Gelman-Rubin)** — target: `R̂ < 1.01` for all parameters.
- **ESS (Effective Sample Size)** — effectively independent samples.
- **Trace plots** — no drift, no visible patterns.

### 5 · Validation against Onsager

The exact solution of the 2D Ising model (Onsager, 1944) gives:

- `Tc = 2 / ln(1 + √2) ≈ 2.26919`
- `β = 1/8 = 0.125`

These values are overlaid on the posterior distributions as reference lines. It is the strongest possible validation: numerical inference compared against a 1944 exact analytical result.

### 6 · Final figures

`src/plots.py` generates three figures for the README:

1. **M(T) curve** with error bars and a vertical line at Onsager's Tc → the phase transition is visible.
2. **Marginal posteriors** of Tc and β with exact values overlaid.
3. **Trace plots** for visual verification of MCMC convergence.

---

## Project structure

```
01-ising-bayesian/
├── README.md
├── requirements.txt
├── main.py                      # end-to-end pipeline
├── src/
│   ├── __init__.py
│   ├── metropolis.py            # Numba-accelerated Metropolis simulator
│   ├── bayesian.py              # PyMC model + MCMC + summary
│   └── plots.py                 # M(T), posteriors, trace plots
├── data/                        # (gitignored) generated CSV + NetCDF
└── results/                     # (gitignored) PNG figures
```

---

## How to reproduce

```bash
pip install -r requirements.txt

# option A: full pipeline
python main.py

# option B: step by step
python -m src.metropolis --out data/magnetization.csv --n-temps 30
python -m src.bayesian   --input data/magnetization.csv --output data/trace.nc
python -m src.plots      --csv data/magnetization.csv --trace data/trace.nc --out-dir results
```

Typical runtime on a modern laptop:

- Simulation: ~2–4 min (with Numba).
- MCMC inference: ~30–60 s.
- Figures: &lt;5 s.

---

## Expected results

- Inferred `Tc` ≈ 2.27 ± 0.04 (exact: 2.2692).
- Inferred `β` ≈ 0.125 ± 0.02 (exact: 0.125).
- Diagnostics: `R̂ < 1.01`, `ESS > 400` for all parameters.

Final PNGs land in `results/` ready to embed in this README.

---

## Tech stack

- **Python** 3.11+
- **NumPy** for vectorized lattice operations.
- **Numba** (`@njit(cache=True)`) to accelerate the Metropolis kernel.
- **PyMC 5** to define the probabilistic model and sample with NUTS.
- **ArviZ** for MCMC diagnostics (R̂, ESS, trace plots).
- **Matplotlib** for final figures.

---

## References

- Onsager, L. (1944). *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*. Physical Review, 65(3–4), 117.
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). *Equation of State Calculations by Fast Computing Machines*. The Journal of Chemical Physics, 21(6), 1087.
- Hoffman, M. D. & Gelman, A. (2014). *The No-U-Turn Sampler*. JMLR 15.
