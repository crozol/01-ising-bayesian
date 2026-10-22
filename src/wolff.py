"""Wolff single-cluster algorithm for the 2D Ising model (Wolff 1989).

Each step picks a random seed spin, grows a cluster by adding aligned
neighbours with probability p_add = 1 − exp(−2J/T) and flips the whole
cluster; the move satisfies detailed balance with acceptance 1. Near Tc the
clusters are of the size of the correlated regions, so the dynamic exponent
drops from z ≈ 2.17 (Metropolis) to z ≈ 0.25, which is what makes long
chains on L = 64 and L = 128 affordable.

To keep the units comparable with Metropolis, a measurement is taken every
k Wolff steps, with k = round(L² / <cluster size>) fixed during burn-in, so
that one measurement costs about one sweep. k must not depend on the current
state: stopping "once L² spins have been flipped" picks measurement times
that correlate with the configuration and biases the averages (on the 4 × 4
lattice at T = 2.5 it shifts <e> by 0.2). The returned sweeps per
measurement, k <cluster size> / L², converts autocorrelation times to sweeps.
"""

from __future__ import annotations

import numpy as np

from .metropolis import _energy_per_site, njit


@njit(cache=True)
def _wolff_step(lattice: np.ndarray, p_add: float, stack: np.ndarray) -> int:
    """Grow and flip one cluster; returns its size. Spins are flipped as they
    join, so an already-added site is never added twice."""
    n = lattice.shape[0]
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    s = lattice[i, j]
    lattice[i, j] = -s
    stack[0] = i * n + j
    top = 1
    size = 1
    while top > 0:
        top -= 1
        k = stack[top]
        a = k // n
        b = k % n
        for x, y in ((a + 1) % n, b), ((a - 1) % n, b), (a, (b + 1) % n), (a, (b - 1) % n):
            if lattice[x, y] == s and np.random.random() < p_add:
                lattice[x, y] = -s
                stack[top] = x * n + y
                top += 1
                size += 1
    return size


@njit(cache=True)
def wolff_chain(lattice: np.ndarray, T: float, n_therm: int, n_measure: int):
    """Burn in for about n_therm sweeps, then return the signed magnetisation
    and the energy per site every k steps, and the sweeps per measurement."""
    p_add = 1.0 - np.exp(-2.0 / T)
    stack = np.empty(lattice.size, dtype=np.int64)
    n2 = lattice.size
    flipped = 0
    steps = 0
    while flipped < n_therm * n2:
        flipped += _wolff_step(lattice, p_add, stack)
        steps += 1
    k = max(1, int(round(n2 * steps / flipped)))
    m = np.empty(n_measure)
    e = np.empty(n_measure)
    flipped = 0
    for i in range(n_measure):
        for _ in range(k):
            flipped += _wolff_step(lattice, p_add, stack)
        s = 0
        for v in lattice.flat:
            s += v
        m[i] = s / n2
        e[i] = _energy_per_site(lattice, 1.0)
    return m, e, flipped / (n_measure * n2)
