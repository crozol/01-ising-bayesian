# Bayesian inference on the 2D Ising model

[![CI](https://img.shields.io/github/actions/workflow/status/crozol/01-ising-bayesian/tests.yml?label=CI&style=flat-square)](https://github.com/crozol/01-ising-bayesian/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)
[![Portfolio page](https://img.shields.io/badge/Portfolio-page-7C3AED?style=flat-square)](https://crozol.github.io/projects/01-ising-bayesian.html)
[![Live demo on Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face%20Space-FFD21E?style=flat-square)](https://huggingface.co/spaces/crozol/ising-bayesian-demo)

**TL;DR** — Bayesian inference of the 2D Ising critical point from Monte Carlo data. A finite-size-scaling fit of the Binder cumulant and the magnetisation, on Wolff cluster data for L = 32, 64, 128, gives `Tc = 2.2691(1)(1)` (Onsager `2.26919`), `ν = 0.981(10)(20)` (exact `1`) and `β = 0.1220(15)(24)` (exact `1/8`), with statistical and systematic errors and flat priors on all three. The fit range is the widest that passes a χ² goodness-of-fit test (`Q = 0.22`), the systematic error is the spread of the variations that also pass it, and the statistical interval alone would exclude `β = 1/8` (Section 12). A first fit of a pure power law to a single lattice (L = 32) gives `Tc = 2.37` and a `β` near `1/8` only through two compensating errors: on the ordered side the simulation matches Onsager's exact magnetisation, but the power law is only its leading term; adding the first correction of Onsager's expansion, the same data give `β = 0.124 [0.122, 0.127]` (Section 11). Stack: NumPy · Numba JIT · ArviZ (PyMC 5 optional).

A numerical experiment: simulate a grid of interacting magnetic spins, observe
how its macroscopic state changes with temperature, and then try to recover the
two numbers that govern that transition — the critical temperature `Tc` and the
critical exponent `β` — from the noisy simulated data using Bayesian inference.
Both numbers are known exactly for this system (Onsager, 1944), so the
experiment has an unambiguous reference to validate against.

The document walks through the measurement and the analysis in the order you
would actually perform them, without spoiling the outcome up front. Following
along produces a physical intuition for the phase transition and a clean
probabilistic estimate of its parameters.

---

## Contents

1. [Objective](#1--objective)
2. [Equipment (software stack)](#2--equipment-software-stack)
3. [Background: what is the 2D Ising model?](#3--background-what-is-the-2d-ising-model)
4. [The observable: magnetization per site](#4--the-observable-magnetization-per-site)
5. [Procedure](#5--procedure)
   - [5.1 · The Monte Carlo "thermometer"](#51--the-monte-carlo-thermometer)
   - [5.2 · Running the system at a fixed temperature](#52--running-the-system-at-a-fixed-temperature)
   - [5.3 · Sweeping across temperatures](#53--sweeping-across-temperatures)
6. [First observation: what does the M(T) curve look like?](#6--first-observation-what-does-the-mt-curve-look-like)
7. [Energy, susceptibility and specific heat](#7--energy-susceptibility-and-specific-heat)
8. [Quantitative analysis: Bayesian inference of Tc and β](#8--quantitative-analysis-bayesian-inference-of-tc-and-β)
9. [Reading the posteriors](#9--reading-the-posteriors)
10. [MCMC diagnostics — did the sampler actually work?](#10--mcmc-diagnostics--did-the-sampler-actually-work)
11. [Is the single-size fit right?](#11--is-the-single-size-fit-right)
12. [Finite-size scaling: several lattice sizes](#12--finite-size-scaling-several-lattice-sizes)
13. [Comparison with Onsager's exact solution](#13--comparison-with-onsagers-exact-solution)
14. [Reproducing the experiment](#14--reproducing-the-experiment)
15. [Possible extensions](#15--possible-extensions)
16. [File map](#16--file-map)
17. [References](#17--references)

---

## 1 · Objective

Measure the magnetization curve `⟨|M|⟩(T)` of a 2D Ising system on a finite
lattice, and from that curve alone — treated as *experimental* data with
statistical noise — infer:

- the temperature `Tc` at which the system changes phase, and
- the exponent `β` that governs how the magnetization vanishes near `Tc`.

Report each quantity as a full **posterior distribution**, not a single point
estimate. Then compare to the exact values derived analytically for an
infinite lattice — and, when a single lattice turns out not to be enough,
repeat the inference with several lattice sizes (finite-size scaling).

## 2 · Equipment (software stack)

| Component | What it does in this experiment |
|---|---|
| Python 3.11+, NumPy | Array arithmetic and base numerics |
| Numba (`@njit`) | JIT-compiles the Metropolis and Wolff inner loops (~50× speed-up for Metropolis) |
| SciPy (`curve_fit`) | Smooth phenomenological curve drawn through the `M(T)` data in the figure |
| NumPy MCMC samplers | Random-walk Metropolis-Hastings (single size) and adaptive Metropolis (finite-size scaling) |
| ArviZ | Posterior summaries, HDIs, `R̂`, ESS |
| PyMC 5 (optional) | Alternative NUTS backend for the single-size fit |
| Matplotlib | Static figures |

No GPU is needed. The single-size pipeline runs in under a minute on a laptop;
the finite-size-scaling simulation with the Wolff algorithm takes about an hour,
dominated by the 128×128 lattice.

## 3 · Background: what is the 2D Ising model?

Picture a flat square grid, `N` sites on a side, with periodic boundaries so
that the left edge connects to the right edge and the top connects to the
bottom (a torus). At each site there is a small magnet — a **spin** —
constrained to point either up (`+1`) or down (`−1`). Neighboring spins prefer
to align: the energy cost of any configuration `s` is

```
E(s) = −J · Σ⟨i,j⟩ sᵢ sⱼ
```

where the sum runs over nearest-neighbor pairs and `J > 0` is the interaction
strength. In this project we set `J = 1` and measure temperature in the same
units (so `T` is dimensionless, in units of `J/k_B`).

The system is in contact with a thermal bath at temperature `T`. At thermal
equilibrium the probability of observing a configuration `s` follows the
Boltzmann distribution:

```
P(s) ∝ exp(−E(s) / T)
```

Two extremes are easy to reason about:

- **Low `T`**: the Boltzmann factor punishes high-energy configurations
  strongly, so the system lives near the energy minimum. The minimum has all
  spins pointing the same way — there are two such ground states, all-up and
  all-down, and the system picks one. The lattice is **magnetized**.
- **High `T`**: the Boltzmann factor is nearly flat, and thermal noise wins
  over the interaction. Spins are essentially independent coin flips. The net
  magnetization averages to zero.

The question this experiment asks is: *what happens in between?* Does the
magnetization decrease smoothly with `T`, or does it collapse suddenly at some
special temperature? That question is exactly what we are going to measure.

Before any curve, a picture of what the two regimes look like on the lattice.
One equilibrium configuration below and one above what will turn out to be
the critical temperature:

![Lattice snapshots](figures/snapshots.png)

Both panels are `32×32` lattices sampled after `2500` Metropolis sweeps from
a random initial condition. Left: almost all spins point the same way; the
mean magnetization is close to `1`. Right: both orientations are present in
roughly equal areas; the mean magnetization fluctuates around `0`. The
transition between these two regimes is what the rest of this experiment
characterizes quantitatively.

## 4 · The observable: magnetization per site

The natural macroscopic quantity is the **average spin**, normalized by the
number of sites:

```
M(s) = (1 / N²) · Σᵢⱼ sᵢⱼ
```

`M` ranges from `−1` (all down) to `+1` (all up) and is `0` for a perfectly
random configuration.

Because the Hamiltonian is symmetric under flipping every spin, a finite
system at low temperature spends roughly half its time in the all-up-ish
region and half in the all-down-ish region. If we averaged `M` directly we
would get something close to `0` even at very low `T`. The physically
meaningful quantity is therefore the **absolute value**:

```
⟨|M|⟩   =   average of |M(s)| over many equilibrium configurations
```

This is the single number we will record per temperature.

## 5 · Procedure

### 5.1 · The Monte Carlo "thermometer"

We do not have an actual thermal bath. Instead we simulate one with the
**Metropolis algorithm** — a simple Markov chain that samples configurations
from the Boltzmann distribution.

One iteration (`src/metropolis.py:_sweep`):

1. Pick a random site `(i, j)`.
2. Compute the energy change `ΔE = 2·J·sᵢⱼ · (sum of the 4 neighbors)`
   that would result from flipping `sᵢⱼ`.
3. Accept the flip with probability `min(1, exp(−ΔE / T))`.

A **sweep** is `N²` single-site proposals, so on average every site is
touched once. Each run starts from a cold (fully ordered) configuration: a
random start can trap the low-temperature runs in a long-lived domain-wall
state, whereas the ordered start samples the broken-symmetry phase cleanly and
disorders just as readily above `Tc`. We do `n_therm = 2000` sweeps of burn-in
to reach equilibrium, followed by `n_measure = 3000` sweeps during which we
record the magnetization and energy after each sweep.

The hot inner loop is JIT-compiled with `@numba.njit(cache=True)`. A pure-Python
fallback activates automatically if Numba is not installed.

### 5.2 · Running the system at a fixed temperature

Before sweeping across temperatures it helps to think about what we
*expect* to see at a single `T`:

- At `T = 1.5` (very cold) the chain should quickly fall into an all-aligned
  state and stay there. The recorded `|M|` values should cluster near `1`
  with very small variance.
- At `T = 3.5` (very hot) the chain should wander across roughly equally
  probable configurations with no preferred direction. Individual `|M|`
  samples should still be positive (we took the absolute value), but small
  and noisy — on the order of `1/N` because it is the typical size of a
  finite-sum fluctuation.
- Somewhere in the middle, something interesting must happen to connect
  these two regimes. We do not assume *where* — we measure it.

### 5.3 · Sweeping across temperatures

`src/metropolis.py:run_full_simulation` runs the procedure above at `25`
temperatures spaced evenly in `T ∈ [1.5, 3.5]`, which brackets both
extremes, and writes the result to `data/magnetization.csv`:

| column | meaning |
|---|---|
| `T` | temperature (units of `J/k_B`) |
| `M_mean` | sample mean of `|M|` over the `n_measure` equilibrium sweeps |
| `M_std` | sample standard deviation of `|M|` over the same sweeps |
| `M_err` | standard error of `⟨|M|⟩`, by blocking |
| `E_mean`, `E_err` | mean energy per site and its blocking error |
| `chi`, `chi_err` | magnetic susceptibility and its block-jackknife error |
| `C`, `C_err` | specific heat and its block-jackknife error |

`M_std` and `M_err` answer different questions. `M_std` is the width of the
distribution of `|M|` at that temperature — how much the magnetisation
fluctuates from sweep to sweep, a physical quantity that peaks at the
transition. `M_err` is the uncertainty of the *average* `⟨|M|⟩`, estimated by
blocking so that the autocorrelation of consecutive sweeps is accounted for;
it is what the likelihood in Section 8 uses. The remaining columns are the
thermodynamic observables analysed in Section 7.

Nothing about Onsager's solution is used at any point here. The simulator
is blind to the theoretical answer.

## 6 · First observation: what does the M(T) curve look like?

Before looking at the figure, try to predict it. A smooth interpolation
between `⟨|M|⟩ ≈ 1` at cold and `⟨|M|⟩ ≈ 0` at hot is the minimum that
physics demands. Does the decline happen linearly? Gently? Abruptly? At
what `T`?

Now the data:

![M(T)](figures/magnetization.png)

Three features are worth naming out loud before moving on:

1. **Two plateaus.** `⟨|M|⟩` saturates near `1` for `T ≲ 2` and near `0` for
   `T ≳ 2.7`. The behavior in those regions is dull and not very
   informative — the magnetization is not changing.
2. **A sharp drop between ~2.2 and ~2.5.** Something dramatic is happening
   in a narrow window. The curve is not a slow roll; the system changes
   character over a range of roughly `0.2` units of `T`. This is the
   signature of a **phase transition**.
3. **Fluctuations grow dramatically around the drop.** At `T ≈ 2.33` the
   standard deviation across sweeps is `0.22` — the equilibrium
   configuration is no longer a single well-defined state but a fluctuating
   mixture. These are the critical fluctuations that give phase transitions
   their reputation for being numerically delicate.

At this point we can *guess* that the critical temperature sits somewhere
between `2.25` and `2.45`. But "somewhere" is not an estimate; we want a
number with a credible interval. That is what the Bayesian analysis below
delivers. First, though, three more observables give an independent look at the
same transition.

## 7 · Energy, susceptibility and specific heat

The magnetization is the order parameter, but it is not the only quantity that
feels the transition. Three companion observables, all measured from the *same*
equilibrium configurations, give independent views of the same critical point:

- the **energy per site** `⟨ε⟩ = ⟨E⟩/N²`, counting how many neighbour bonds are
  satisfied;
- the **magnetic susceptibility** `χ = (N²/T)·(⟨m²⟩ − ⟨|m|⟩²)`, the variance of
  the magnetization;
- the **specific heat** `C = (N²/T²)·(⟨ε²⟩ − ⟨ε⟩²)`, the variance of the energy.

The last two are fluctuation observables: by the fluctuation–dissipation
relation they measure the system's response — to a magnetic field and to
heating — and both are expected to peak at the transition.

![Observables](figures/observables.png)

**Energy.** `⟨ε⟩` climbs smoothly from the ground-state value `−2J` (every bond
satisfied) toward `0`, with its steepest slope in the critical region. Its error
bars — the statistical error of the mean, estimated by blocking — are tiny: a
mean is an easy thing to measure well.

**Susceptibility and specific heat.** Both peak in the critical region: `χ` at
`T = 2.33` and `C` at `T = 2.25`, the grid points nearest the transition on
either side of Onsager's infinite-lattice `Tc = 2.2692`. On a finite lattice
these peaks are *pseudo-critical* temperatures, displaced from `Tc` by an
amount that shrinks as `L^(−1/ν)`; with a grid spacing of `0.083` this sweep
can only place them to within a grid step. Section 12 locates `Tc` precisely
with several lattice sizes.

**Why the peaks are ragged.** Unlike the energy, `χ` and `C` carry large, unequal
error bars near the transition, and `χ` even shows a spurious secondary bump
around `T ≈ 2.17`. This is not a second transition — it is *critical slowing
down*. Near `Tc` the correlation length grows to the size of the lattice and the
autocorrelation time of single-spin-flip Metropolis dynamics diverges as
`τ ∝ ξ^z`, with a dynamic exponent `z ≈ 2.17` for the 2D Ising model
(Hohenberg & Halperin 1977; Nightingale & Blöte 1996). The chain decorrelates so
slowly there that a fixed-length run holds only a handful of independent samples,
so variance-based estimators like `χ` and `C` turn noisy — which is exactly why
their error bars balloon in the critical region while `⟨ε⟩` stays clean. The
block-jackknife bars (Newman & Barkema 1999) make this honest: the secondary bump
sits well within its own uncertainty. The standard cure is a cluster algorithm
(Swendsen–Wang 1987; Wolff 1989), which updates whole correlated domains at
once; Section 12 uses the Wolff algorithm for the finite-size-scaling data.

## 8 · Quantitative analysis: Bayesian inference of Tc and β

Close to a second-order phase transition, statistical mechanics predicts that
the order parameter vanishes as a power law of the reduced temperature:

```
⟨|M|⟩(T)   ≈   (1 − T/Tc)^β        if T < Tc
⟨|M|⟩(T)   =   0                     if T > Tc
```

`β` is called a **critical exponent**, and for the 2D Ising universality class
theory predicts `β = 1/8` exactly. `Tc` is the critical temperature itself.

### Setting up the probabilistic model

The simulated data `{T_k, M_k, e_k}` (with `e_k` the standard error `M_err`)
are treated as noisy observations of that piecewise power law. The
probabilistic model (`src/bayesian.py`) is:

**Priors** — what we believe about the parameters *before* seeing the data:

| Parameter | Prior | Why |
|---|---|---|
| `Tc` | `Normal(2.3, 0.3)` | broad envelope over the whole transition region |
| `β`  | `Uniform(0.01, 0.5)` | flat, so that no knowledge of the exact `1/8` enters through the prior |
| `σ`  | `HalfNormal(0.1)` | scale of what the power law does not describe |

An earlier version used `β ~ Normal(0.12, 0.05)`, centred almost on the exact
value; a posterior that "recovers" `1/8` under such a prior is partly the
prior speaking. Section 11 compares both.

**Likelihood** — Gaussian, with a variance per point:

```
M_k  ~  Normal( (1 − T_k/Tc)^β ,  √(e_k² + σ²) )      if T_k < Tc
M_k  ~  Normal( 0              ,  √(e_k² + σ²) )      otherwise
```

`e_k` is the Monte Carlo error of each point, known from the simulation; `σ`
is a free *discrepancy* term that absorbs whatever the power law does not
describe. If the model were exact, `σ` would come out close to zero.

**Posterior** — by Bayes' theorem:

```
P(Tc, β, σ | data)   ∝   P(data | Tc, β, σ) · P(Tc, β, σ)
```

There is one more modelling choice here, and it matters: the mean is a pure
power law with amplitude fixed to `1`, and it is zero above `Tc`. On a finite
lattice `⟨|M|⟩` is not zero above `Tc` (it decays like `1/L`), and below `Tc`
the power law is only the leading term of the exact magnetisation. Section 11
tests what these simplifications do to the result.

The left-hand side has no closed form; we sample from it with MCMC.

### Sampling the posterior

Two backends are shipped, both producing an `arviz.InferenceData` so the
downstream figures are backend-agnostic:

- **`numpy` (default)** — a random-walk Metropolis-Hastings sampler
  implemented directly on the log-posterior. No C compiler required; runs in
  about 15 seconds. 4 chains × (1000 tune + 50 000 draws).
- **`pymc`** — reference implementation with PyMC + NUTS (gradient-based).
  More efficient but relies on PyTensor's C backend.

Select with `--backend numpy|pymc`.

## 9 · Reading the posteriors

![Posteriors](figures/posteriors.png)

**Left panel · posterior of `Tc`.** Most of the probability mass lies between
`2.33` and `2.43`, with a sharp peak near `2.34` and a long tail to higher
temperatures. The pink vertical line is the posterior mean (`2.368`); the
orange dashed line is Onsager's exact value `Tc = 2.2692`; the bar is the 95%
highest-density interval (HDI). The exact value falls **outside** the HDI, by
about `0.1`.

The asymmetric shape is a feature of the likelihood: the piecewise mean has
a hard threshold at `T = Tc`, so as `Tc` slides across a data temperature the
number of points in the power-law arm changes by one, producing kinks in the
log-posterior.

**Right panel · posterior of `β`.** The posterior is wide — the 95% HDI
`[0.049, 0.151]` spans a factor of three — and the exact `1/8` sits inside it,
toward the upper end. A credible interval this wide is not much of a
measurement, and Section 11 shows that its location depends on modelling
choices more than on the data.

The posterior of the discrepancy term, `σ = 0.115 [0.083, 0.151]`, is the
first warning: it is about a hundred times larger than the Monte Carlo
errors `e_k` (`10⁻⁴` to `4 × 10⁻²`). The power law misses the data by far more
than the data's own uncertainty.

## 10 · MCMC diagnostics — did the sampler actually work?

A sampler that has not converged produces numbers that look like samples
but are not. These three checks are non-negotiable:

![Trace plots](figures/trace.png)

- **`R̂` (Gelman-Rubin).** Compares variance between chains to variance within
  chains. `R̂ ≤ 1.01` is the gold standard, `≤ 1.05` is tolerable. Here
  `R̂ = 1.00` for all three parameters.
- **`ESS_bulk` (effective sample size).** Accounts for autocorrelation —
  one independent sample is worth many correlated ones. `≥ 200` is a
  working threshold. All three clear it: `Tc` 3 983, `σ` 5 628 and `β` 2 074.
  The hard threshold at `T = Tc` makes `β` the slowest-mixing parameter of
  this random-walk sampler, which is why the chains are 50 000 draws long:
  with 5 000, the ESS of `β` was 187.
- **Visual inspection.** Healthy chains overlap, do not drift with
  iteration, do not stick in any one region. Each subplot above shows all
  four chains on top of each other.

The acceptance rate is 29–31% per chain. The sampler explores this posterior
adequately, with `β` the parameter to watch. Convergence, though, only says
the posterior was sampled correctly — not that the model behind it is right.

## 11 · Is the single-size fit right?

The single-size fit leaves `Tc` about `0.1` above the exact value and `β`
poorly constrained, with a discrepancy term far larger than the data's own
errors. `scripts/model_comparison.py` refits the same data under each
combination of three choices — fitting all temperatures or only the ordered
side `T < 2.25` (away from the rounded transition and the `1/L` tail above
it), a homoscedastic or heteroscedastic likelihood, and the normal or the flat
prior on `β`:

| Fit window | Likelihood | `β` prior | `Tc` | `β` (95% HDI) | `σ` |
|---|---|---|---|---|---|
| all `T` | homoscedastic | normal | 2.370 ± 0.036 | 0.102 [0.057, 0.150] | 0.119 |
| all `T` | homoscedastic | flat | 2.368 ± 0.037 | 0.099 [0.052, 0.154] | 0.119 |
| all `T` | heteroscedastic | normal | 2.369 ± 0.035 | 0.098 [0.056, 0.145] | 0.115 |
| all `T` | heteroscedastic | flat (**used**) | 2.368 ± 0.036 | 0.095 [0.049, 0.151] | 0.115 |
| `T < 2.25` | homoscedastic | normal | 2.31 ± 0.19 | 0.046 [0.025, 0.080] | 0.035 |
| `T < 2.25` | homoscedastic | flat | 2.26 ± 0.14 | 0.041 [0.024, 0.067] | 0.032 |
| `T < 2.25` | heteroscedastic | normal | 2.31 ± 0.18 | 0.046 [0.025, 0.079] | 0.034 |
| `T < 2.25` | heteroscedastic | flat | 2.25 ± 0.13 | 0.040 [0.024, 0.065] | 0.031 |

Three things follow.

1. **The likelihood and the prior barely matter.** The heteroscedastic
   likelihood cannot help when `σ` dominates every `e_k`, and the flat prior
   moves `β` by less than its own uncertainty.
2. **The fit window matters a great deal.** On the ordered side alone, `β`
   drops to `0.04` and its HDI excludes `1/8`, while `Tc` is barely
   constrained (`± 0.13` to `± 0.19`).
3. **The simulation is not the problem.** For `T < 2.25` the measured
   `⟨|M|⟩` agrees with Onsager's exact infinite-lattice magnetisation
   `M(T) = (1 − sinh(2/T)⁻⁴)^(1/8)` (Yang 1952) to within `0.012`, or 1.8
   standard errors. The pure power law with the *exact* `Tc` and `β` misses the
   same points by up to `0.14`: it is only the leading term of `M(T)` as
   `T → Tc`, with an amplitude of about `1.22` rather than `1`.

Freeing the amplitude, `M = B (1 − T/Tc)^β`, does not rescue the fit: on the
ordered side it gives `β = 0.074 [0.055, 0.103]`, which excludes `1/8`, and over
the full range `β = 0.38`, far from it.

Fixing the amplitude at its exact value does not rescue it either, and the
way it fails shows what is missing. To leading order in `t = 1 − T/Tc`,
Onsager's magnetisation is `M ≈ A t^(1/8)` with `A = (8√2/Tc)^(1/8) = 1.2224`.
With `A`, `Tc` and `β` all exact, that leading term still misses the exact
curve by up to `0.081` on `T ∈ [1.5, Tc)` (`0.146` with `A = 1`); at `t = 0.3`
the exact `M` lies 7% below it. The ordered-side data sit at `t` between
`0.01` and `0.34`, where the next term of the expansion is not negligible, and
a fit without that term compensates with the only exponent it has: `β` rises
to `0.19–0.23` across the four fixed-amplitude fits (both windows, both
priors), and every HDI excludes `1/8`.

The fix is to include that term. To first order,
`M_Onsager / (A t^(1/8)) = 1 + b t + O(t²)` with `b = −0.225`, and on `T < 2.25`
the `O(t²)` remainder is at most `0.97` Monte Carlo standard errors at any data
point. The model `M = A t^β (1 + b t)` therefore needs no discrepancy term `σ`:
the likelihood uses the Monte Carlo errors alone, with flat priors on `Tc`,
`β` and `b`. On the same nine points as above it gives `β = 0.124 [0.122, 0.127]`,
`Tc = 2.262 [2.244, 2.281]` and `b = −0.227 [−0.233, −0.221]`, with
`χ² = 8.1` for 6 degrees of freedom: all three compatible with the exact
`1/8`, `2.269` and `−0.225`. On `T < 2.1` (eight points) the result is the
same (`β = 0.124`, `Tc = 2.260`).

This is a consistency check, not an independent measurement: `A` and the form
of the correction come from the exact solution, which a measurement on an
unsolved model would not have. What it establishes is that the single-size
data are right and that the failures above come from the functional form.
The estimate that uses no exact input is the finite-size scaling of
Section 12.

The conclusion is that a single lattice cannot determine `β` from the leading
power law alone. Close to `Tc`, where that power law holds, the finite lattice
rounds the transition and leaves a `1/L` tail; far from `Tc`, where the
lattice behaves like an infinite one, the leading term is not enough and the
correction above is needed. The `β ≈ 0.09–0.10` of the
full-range fit is a compromise between those two errors, and its proximity to
`1/8` is partly luck. The `+0.1` offset in `Tc` is a finite-size effect only
in that the `1/L` tail above `Tc` exists because the lattice is finite. It is
not a shift of the transition: finite-size scaling of several lattices gives
`Tc = 2.2691(1)(1)`, compatible with the exact value (Section 12). The
offset comes from the model, which sets `M = 0` above `Tc` and cannot describe
that tail; the fitted threshold moves up to absorb it.

## 12 · Finite-size scaling: several lattice sizes

The finite lattice is not just a nuisance; how the observables change with
`L` carries the critical behaviour (Fisher & Barber 1972). Near `Tc` they
depend on `T` and `L` only through the scaling variable
`x = (T − Tc) · L^(1/ν)`, up to corrections that shrink with `L`. Two
observables are used:

- the **Binder cumulant** `U = 1 − ⟨m⁴⟩ / (3⟨m²⟩²)` (Binder 1981), which is
  dimensionless, so `U(T, L) = f(x)`: the curves for different `L` cross at
  `Tc`, at a universal value `U* ≈ 0.6107` for the periodic square lattice
  (Salas & Sokal 2000);
- the **magnetisation**, `⟨|m|⟩(T, L) = L^(−β/ν) g(x)`, whose size dependence
  at `Tc` gives the ratio `β/ν`.

**Data.** `src/fss.py --algorithm wolff` runs `L = 16, 32, 64, 128` at 15
temperatures in `[2.20, 2.34]`, with 20 000 burn-in and 200 000 measurements
per point, each a fixed number of cluster updates that flips about `L²` spins
(one sweep's worth; `src/wolff.py`). Errors come from a block jackknife whose
blocks are at least 20 integrated autocorrelation times `τ_int` long
(`src/autocorr.py`, with Sokal's automatic window); `τ_int` is measured at
every point.

**Why the Wolff algorithm.** Local Metropolis updates decorrelate slowly near
`Tc` (critical slowing down). Measured `τ_int` of `|m|` at `T = 2.27`, in
sweeps:

| `L` | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Metropolis | 9.9 | 42 | 159 | 1 700 ± 800 | — |
| Wolff (single cluster) | — | 1.28 | 1.43 | 1.64 | 1.77 |

For Metropolis `τ_int` grows roughly as `L²` (from `L = 8` to `32`), as expected
for local dynamics (`z ≈ 2.17`, Nightingale & Blöte 1996); for Wolff it grows
by a factor `1.4` from `L = 16` to `128` (Wolff 1989). Metropolis chains of
200 000 sweeps for `L = 8–64` (`data/fss_metropolis_tau.csv`) split into 50
jackknife blocks give blocks of 4 000 sweeps, about `2.3 τ_int` at `L = 64` near
`Tc`, too short: with blocks of at least `20 τ_int` the error bars on `⟨|m|⟩` and
`U` grow by up to `1.6×`. The Metropolis data are not used below; the Wolff
means agree with them within the corrected errors (`χ²/n = 0.7–1.4` per size),
with errors 4 to 50 times smaller.

**Model.** `src/fss_fit.py` fits both observables jointly, with `f` and `g`
expanded to second order in `x`:

```
U(T, L)      =  U* + u₁ x + u₂ x²
⟨|m|⟩(T, L)  =  L^(−β/ν) (g₀ + g₁ x + g₂ x²),       x = (T − Tc) L^(1/ν)
```

The priors are flat on `Tc ∈ (2.0, 2.6)`, `ν ∈ (0.5, 2)`, `β/ν ∈ (0, 0.5)` and
on the six expansion coefficients, so none of the exact values is encoded.
Each observable has a discrepancy term, as in Section 8. The likelihood is
linear in the six coefficients, which are integrated out analytically; the
remaining five parameters are sampled with adaptive Metropolis (Haario,
Saksman & Tamminen 2001), 4 chains × 10 000 draws, `R̂ ≤ 1.002`,
`ESS_bulk > 2 600` for every parameter.

**Fit window.** The expansion holds for small `|x|`, so the window is fixed in
`x`, not in `T`: a fixed `T` window reaches `|x| ∝ L^(1/ν)`, twice as far on each
doubling of `L` (`[2.23, 2.31]` reaches `|x| = 2.6` at `L = 64` and `5.2` at
`L = 128`). Points are kept when `|T − T*| L^(1/ν₀) ≤ x_max`, with `T* = 2.2684`,
the Binder crossing of `L = 64` and `128`, and `ν₀ = 1.04`, from the growth of
the slope `dU/dT` at `T*` as `L^(1/ν)`; neither uses the exact values. The range
of the fit (`x_max` and the smallest lattice `L_min`) is chosen by the classical
goodness-of-fit test: the same model fitted by weighted least squares on the
Monte Carlo errors alone, without discrepancy terms, gives `χ²_min` with
`2N − 9` degrees of freedom for quadratic `f` and `g`, and the probability `Q`
of a larger `χ²`; a fit passes when `Q > 0.05`. The errors in the table come
from the curvature of `χ²` at its minimum (`Δχ² = 1`):

| Sizes | `\|x\| ≤` | Order of `f`, `g` | Points | `χ²/dof` | `Q` | `Tc` | `ν` | `β` |
|---|---|---|---|---|---|---|---|---|
| 16, 32, 64, 128 | 1 | 2 | 26 | 78.4/43 | 8 × 10⁻⁴ | 2.26887(7) | 0.977(3) | 0.1198(5) |
| 16, 32, 64, 128 | 1.5 | 2 | 34 | 235.1/59 | < 10⁻²⁰ | 2.26890(6) | 0.980(2) | 0.1202(4) |
| **32, 64, 128** | **1** | **2** | **13** | **21.1/17** | **0.22** | **2.26908(10)** | **0.980(7)** | **0.1217(10)** |
| 32, 64, 128 | 1.5 | 2 | 19 | 165.9/29 | < 10⁻²⁰ | 2.26898(8) | 0.989(3) | 0.1214(6) |
| 32, 64, 128 | 2 | 2 | 26 | 1250.8/43 | < 10⁻²⁰ | 2.26903(7) | 0.978(2) | 0.1214(4) |
| 64, 128 | 1 | 2 | 6 | 5.5/3 | 0.14 | 2.26904(17) | 1.000(18) | 0.1241(22) |
| 64, 128 | 1.5 | 2 | 10 | 231.4/11 | < 10⁻²⁰ | 2.26879(14) | 1.010(7) | 0.1215(15) |
| 32, 64, 128 | 1 | 3 | 13 | 17.7/15 | 0.28 | 2.26900(11) | 0.982(7) | 0.1209(11) |
| 32, 64, 128 | 1 | 4 | 13 | 15.8/13 | 0.26 | 2.26901(11) | 0.984(8) | 0.1215(12) |
| 32, 64, 128 | 1.5 | 3 | 19 | 46.5/27 | 0.011 | 2.26902(8) | 0.989(3) | 0.1220(6) |
| 32, 64, 128 | 1.5 | 4 | 19 | 45.4/25 | 8 × 10⁻³ | 2.26902(8) | 0.988(3) | 0.1219(6) |

The test rejects every window wider than `|x| ≤ 1` and every fit that includes
`L = 16`. Cubic and quartic `f` and `g` do not rescue the next wider window
(`L ≥ 32`, `|x| ≤ 1.5`: `Q = 0.011` and `0.008`), so the misfit is not only the
truncated expansion: with errors of `10⁻⁵` to `10⁻³`, the corrections to scaling
that the leading-order form leaves out are resolved. The accepted fit is
`L = 32, 64, 128` with `|x| ≤ 1` (7, 4 and 2 temperatures; `χ²/dof = 21.1/17`,
`Q = 0.22`), and in the Bayesian fit of that window both discrepancy terms are
consistent with zero.

![Finite-size scaling](figures/fss.png)

Panel (a) shows the Binder curves crossing; panels (b) and (c) show that,
with the fitted `Tc`, `ν` and `β/ν`, the points from all sizes fall on a single
curve. Hollow points lie outside the fitted sizes or window.

| Parameter | Estimate (stat)(syst) | 95% HDI | Exact |
|---|---|---|---|
| `Tc` | **2.2691(1)(1)** | [2.2689, 2.2693] | 2.26919 |
| `ν` | **0.981(10)(20)** | [0.960, 1.001] | 1 |
| `β = ν · (β/ν)` | **0.1220(15)(24)** | [0.1190, 0.1248] | 0.125 |
| `U*` | **0.6112(3)(3)** | [0.6106, 0.6117] | 0.6107 |

The statistical error is the posterior standard deviation. The systematic
error is the largest shift of the least-squares minimum, relative to the
accepted fit, among the variations that also pass the `χ²` test: `L_min = 64`
(`Q = 0.14`) and cubic or quartic `f`, `g` (`Q = 0.28` and `0.26`). The exact
values lie within the combined uncertainty (statistical and systematic in
quadrature): `Tc` at 0.5, `ν` at 0.8, `β` at 1.1 and `U*` at 1.2 standard
deviations. The statistical HDI alone is too narrow for `β`: it excludes `1/8`.
As a model-free check, the Binder curves of consecutive sizes cross at
`2.2666(7)`, `2.2699(5)` and `2.2684(3)`; these straight-line interpolations
between grid temperatures `0.01` apart scatter by more than their errors.

**Limitation.** With a grid of `0.01` in `T`, only 4 temperatures of `L = 64`
and 2 of `L = 128` fall inside `|x| ≤ 1`. A finer grid near `Tc` for the
largest lattices would tighten `ν` and `β` and allow corrections to scaling to
be fitted explicitly instead of being excluded through `L_min`.

## 13 · Comparison with Onsager's exact solution

For the 2D Ising model on an *infinite* lattice, the critical parameters were
derived analytically by Onsager (1944) and Yang (1952):

```
Tc   =   2 / ln(1 + √2)   ≈   2.2692
β    =   1/8              =   0.125
ν    =   1
```

| Method | `Tc` (95% HDI) | `β` (95% HDI) | Exact in HDI? |
|---|---|---|---|
| Single lattice, `L = 32`, power law (Sections 8–11) | 2.368 [2.333, 2.434] | 0.095 [0.049, 0.151] | `Tc` no · `β` yes, but not robust |
| Finite-size scaling, `L = 32–128` (Section 12) | 2.2691 [2.2689, 2.2693] | 0.1220 [0.1190, 0.1248] | `Tc` yes · `β` no, but within 1.1 σ with the systematic error |

The single-size fit is not a usable estimator of either quantity: its `Tc` is
off by `+0.1`, and its `β` interval is wide and moves with the fit window.
Finite-size scaling recovers `Tc` to within `0.0001` and `β` to within its
statistical and systematic uncertainty, with the same kind of Bayesian
model — the difference is that it models how the finite lattice distorts the
transition instead of ignoring it.

## 14 · Reproducing the experiment

```bash
pip install -r requirements.txt

# Option A — full pipeline end-to-end (about 1 hour, almost all of it the Wolff simulation)
python main.py
python main.py --reuse-fss      # skip the finite-size-scaling simulation if data/fss_wolff.csv exists

# Option B — step by step
python -m src.metropolis --out data/magnetization.csv --size 32 --n-temps 25
python -m src.bayesian   --input data/magnetization.csv --output data/trace.nc
python -m src.fss        # Wolff, L = 16–128 -> data/fss_wolff.csv
python -m src.fss_fit    # L = 32–128, |x| <= 1 -> data/fss_summary.json
python -m src.plots      --csv data/magnetization.csv --trace data/trace.nc \
                         --out-dir figures --lattice-size 32

# Section 11 and Section 12 tables
python -m scripts.model_comparison --csv data/magnetization.csv
python -m scripts.fss_systematics   # chi2 test and systematics, a few seconds

# Section 12 autocorrelation times with Metropolis
python -m src.fss --algorithm metropolis --sizes 8 16 32 64 --out data/fss_metropolis_tau.csv
```

Every simulation is seeded (`src/metropolis.py::_seed` seeds the RNG inside
Numba's compiled code), so these commands reproduce the numbers above exactly.

Typical runtimes on a laptop CPU:

| Stage | Time |
|---|---|
| Single-size simulation (32×32, 25 temperatures) | ~10 s |
| Single-size MCMC (4 chains × 50 000 draws) | about 15 s |
| Finite-size-scaling simulation, Wolff (L = 16–128, 15 temperatures) | ~1 h (`L = 128`: ~50 min) |
| Finite-size-scaling fit | ~30 s |
| Figures | < 5 s |

The Wolff time is extrapolated from chains a tenth as long.

A companion notebook `notebooks/explore.ipynb` steps through the single-size
pipeline interactively with intermediate prints.

## 15 · Possible extensions

**A finer grid and larger lattices.** Steps of `0.0025` in `T` around `T*` for
`L = 64` and `128`, and `L = 256`, which the Wolff algorithm makes affordable,
would put more points inside the accepted window of Section 12 and tighten
`ν` and `β`.

**Corrections to scaling.** Adding the leading correction term `L^(−ω)` to
the scaling functions would let the fit use `L = 16` and wider windows,
which the χ² test of Section 12 now rejects.

## 16 · File map

```
01-ising-bayesian/
├── README.md
├── requirements.txt
├── main.py                      # end-to-end pipeline
├── src/
│   ├── __init__.py
│   ├── metropolis.py            # Numba-accelerated Metropolis simulator
│   ├── bayesian.py              # single-size model: NumPy MH + PyMC backends
│   ├── wolff.py                 # Wolff single-cluster updates
│   ├── autocorr.py              # integrated autocorrelation time, jackknife block length
│   ├── fss.py                   # finite-size-scaling data: <|m|>, Binder cumulant, τ_int
│   ├── fss_fit.py               # Bayesian finite-size-scaling fit and least-squares χ² test
│   └── plots.py                 # static figures
├── scripts/
│   ├── model_comparison.py      # Section 11: table, Onsager check, free/fixed amplitude, correction to scaling
│   ├── fss_systematics.py       # Section 12 goodness-of-fit and systematics table
│   ├── summarize.py             # posterior summary → JSON
│   └── export_json.py           # arrays → JSON for the portfolio page
├── notebooks/
│   └── explore.ipynb            # step-by-step walkthrough
├── figures/                     # tracked PNGs embedded in this README
│   ├── snapshots.png            # two equilibrium lattices, one per phase
│   ├── magnetization.png        # M(T) curve + fit + Onsager line
│   ├── observables.png          # energy, susceptibility, specific heat vs. T
│   ├── posteriors.png           # marginal posteriors of Tc and β (single size)
│   ├── trace.png                # per-chain trace plots + diagnostics
│   └── fss.png                  # Binder crossing and data collapse
├── data/                        # (gitignored) generated CSV + NetCDF
└── results/                     # (gitignored) staging for fresh PNGs
```

## 17 · References

- Onsager, L. (1944). *Crystal Statistics. I. A Two-Dimensional Model with
  an Order-Disorder Transition.* Physical Review **65**, 117.
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &
  Teller, E. (1953). *Equation of State Calculations by Fast Computing
  Machines.* Journal of Chemical Physics **21**, 1087.
- Hoffman, M. D. & Gelman, A. (2014). *The No-U-Turn Sampler.*
  Journal of Machine Learning Research **15**.
- Ferrenberg, A. M. & Landau, D. P. (1991). *Critical behavior of the
  three-dimensional Ising model: A high-resolution Monte Carlo study.*
  Physical Review B **44**, 5081.  (finite-size scaling reference)
- Hohenberg, P. C. & Halperin, B. I. (1977). *Theory of Dynamic Critical
  Phenomena.* Reviews of Modern Physics **49**, 435.  (dynamic critical
  exponents and critical slowing down)
- Nightingale, M. P. & Blöte, H. W. J. (1996). *Dynamic Exponent of the
  Two-Dimensional Ising Model and Monte Carlo Computation of the Subdominant
  Eigenvalue of the Stochastic Matrix.* Physical Review Letters **76**, 4548.
  (dynamic exponent `z ≈ 2.17`)
- Swendsen, R. H. & Wang, J.-S. (1987). *Nonuniversal Critical Dynamics in
  Monte Carlo Simulations.* Physical Review Letters **58**, 86.  (cluster
  algorithm)
- Wolff, U. (1989). *Collective Monte Carlo Updating for Spin Systems.*
  Physical Review Letters **62**, 361.  (single-cluster algorithm)
- Newman, M. E. J. & Barkema, G. T. (1999). *Monte Carlo Methods in
  Statistical Physics.* Oxford University Press.  (critical slowing down;
  blocking and jackknife error analysis)
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed., Chapman &
  Hall.  (general reference for priors, HDIs, and diagnostics)
- Yang, C. N. (1952). *The Spontaneous Magnetization of a Two-Dimensional
  Ising Model.* Physical Review **85**, 808.  (exact magnetisation `M(T)`)
- Fisher, M. E. & Barber, M. N. (1972). *Scaling Theory for Finite-Size
  Effects in the Critical Region.* Physical Review Letters **28**, 1516.
- Binder, K. (1981). *Finite size scaling analysis of Ising model block
  distribution functions.* Zeitschrift für Physik B **43**, 119.  (Binder
  cumulant)
- Salas, J. & Sokal, A. D. (2000). *Universal Amplitude Ratios in the
  Critical Two-Dimensional Ising Model on a Torus.* Journal of Statistical
  Physics **98**, 551.  (`U* = 0.6107`)
- Sokal, A. D. (1997). *Monte Carlo Methods in Statistical Mechanics:
  Foundations and New Algorithms.* In *Functional Integration: Basics and
  Applications*, NATO ASI Series B **361**, 131. Springer.  (automatic window
  for `τ_int`)
- Haario, H., Saksman, E. & Tamminen, J. (2001). *An adaptive Metropolis
  algorithm.* Bernoulli **7**, 223.
