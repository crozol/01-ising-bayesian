"""Integrated autocorrelation time of a Monte Carlo series.

For a stationary series x_t with normalised autocorrelation ρ(t),

    τ_int = 1/2 + Σ_{t≥1} ρ(t),

so that the variance of the sample mean of n points is 2 τ_int σ²/n: the
chain carries n / (2 τ_int) independent samples. The sum is truncated with
Sokal's automatic window (Sokal 1997; Madras & Sokal 1988): the smallest M
with M ≥ c τ_int(M), c = 6, which keeps the bias below e^(-c) while the noise
of the tail stays bounded. The statistical error of the estimate is
τ_int · sqrt(2 (2M + 1) / n).

Block-jackknife error bars are reliable only when each block spans many
autocorrelation times; `n_blocks_for` picks the number of blocks so that
every block is at least 20 τ_int long, as asked for near Tc where τ_int
grows as L^z (z ≈ 2.17 for Metropolis, ≈ 0.25 for Wolff).
"""

from __future__ import annotations

import math

import numpy as np

MIN_BLOCK_TAUS = 20


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation ρ(t), t = 0 … n−1, via FFT."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    f = np.fft.rfft(d, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0.0:                       # constant series: no fluctuations at all
        out = np.zeros(n)
        out[0] = 1.0
        return out
    return acf / acf[0]


def tau_int(x: np.ndarray, c: float = 6.0) -> dict:
    """τ_int with Sokal's window; returns the estimate, its error and the window M."""
    rho = autocorrelation(x)
    n = len(rho)
    tau = 0.5
    for m in range(1, n):
        tau += rho[m]
        if m >= c * tau:
            break
    tau = max(tau, 0.5)
    return {"tau": float(tau), "err": float(tau * math.sqrt(2.0 * (2 * m + 1) / n)),
            "window": int(m)}


def n_blocks_for(n: int, tau: float, max_blocks: int = 50) -> int:
    """Number of jackknife blocks such that each block spans ≥ 20 τ_int,
    capped at `max_blocks` and never below 2."""
    by_tau = n // max(1, math.ceil(MIN_BLOCK_TAUS * tau))
    return int(max(2, min(max_blocks, by_tau)))
