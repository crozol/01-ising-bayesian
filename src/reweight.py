"""Multiple-histogram reweighting (Ferrenberg & Swendsen 1989) of the Wolff runs of one lattice size.

Each run j, at inverse temperature b_j = 1/T_j, gives n_j measurements of the total energy E and
magnetisation M (src/fss.py --series-dir). The Boltzmann weight depends on a configuration only
through E, so all the runs of a size estimate one density of states Omega(E). With g_j = 1 + 2 tau_j
the statistical inefficiency of run j and H_j(E) its histogram,

    Omega(E) = sum_j H_j(E) / g_j  /  sum_j (n_j / g_j) exp(f_j - b_j E),
    exp(-f_j) = sum_E Omega(E) exp(-b_j E),

solved by iteration in logarithms. Given E, the distribution of M does not depend on T, so an
observable O(M) at any T is

    <O>(T) = sum_E Omega(E) exp(-E/T) O(E) / sum_E Omega(E) exp(-E/T),

with O(E) the mean of O over every measurement with energy E. The runs are used only where they
overlap: the curves are evaluated inside the simulated range of T.

Errors: the series of every run is cut into the same number of consecutive blocks; deleting block b
of every run and redoing the whole reweighting gives n_b jackknife curves. Their covariance over a
grid of temperatures, between <|m|> and U too, is what src/fss_fit.py uses as a full covariance
matrix (points reweighted from the same runs are strongly correlated).

Everything works on sufficient statistics per run and block (energy histograms and the sums of
|m|, m^2 and m^4 per energy), so a jackknife curve costs a few milliseconds.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .autocorr import n_blocks_for


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


class Runs:
    """The runs of one lattice size as sufficient statistics on a common energy grid.

    ``counts``, ``s1``, ``s2``, ``s4`` have shape (n_runs, n_blocks, n_E): per run and block, the
    number of measurements at each energy and the sums of |m|, m^2 and m^4 there (m = M / L^2)."""

    def __init__(self, files: list[str], n_blocks: int | None = None):
        z = [np.load(f) for f in files]
        self.L = int(z[0]["L"])
        if any(int(r["L"]) != self.L for r in z):
            raise ValueError("all runs must have the same lattice size")
        self.T = np.array([float(r["T"]) for r in z])
        order = np.argsort(self.T)
        z, self.T = [z[i] for i in order], self.T[order]
        self.tau = np.array([float(r["tau"]) for r in z])
        n = np.array([len(r["E"]) for r in z])
        if n_blocks is None:            # the same blocks for every run, each >= 20 tau
            n_blocks = min(n_blocks_for(int(k), float(t)) for k, t in zip(n, self.tau))
        self.n_blocks = n_blocks
        energies = np.unique(np.concatenate([r["E"] for r in z]))
        self.E = energies.astype(float)
        shape = (len(z), n_blocks, len(energies))
        self.counts, self.s1, self.s2, self.s4 = (np.zeros(shape) for _ in range(4))
        n2 = float(self.L * self.L)
        for j, r in enumerate(z):
            bs = len(r["E"]) // n_blocks
            e_idx = np.searchsorted(energies, r["E"][: bs * n_blocks])
            m = r["M"][: bs * n_blocks] / n2
            blk = np.repeat(np.arange(n_blocks), bs)
            for arr, w in ((self.counts, np.ones_like(m)), (self.s1, np.abs(m)), (self.s2, m ** 2),
                           (self.s4, m ** 4)):
                np.add.at(arr[j], (blk, e_idx), w)

    @staticmethod
    def from_dir(directory: str, L: int, n_blocks: int | None = None) -> "Runs":
        files = sorted(glob.glob(str(Path(directory) / f"L{L}_T*.npz")))
        if not files:
            raise FileNotFoundError(f"no series for L = {L} in {directory}")
        return Runs(files, n_blocks)

    def _totals(self, drop: int | None):
        tot = [a.sum(axis=1) for a in (self.counts, self.s1, self.s2, self.s4)]
        if drop is not None:
            tot = [t - a[:, drop] for t, a in zip(tot, (self.counts, self.s1, self.s2, self.s4))]
        return tot

    def solve(self, drop: int | None = None, f0: np.ndarray | None = None, tol: float = 1e-11,
              max_iter: int = 100_000) -> dict:
        """Density of states from all runs (minus block ``drop`` of each), and the per-energy means."""
        H, S1, S2, S4 = self._totals(drop)
        g = 1.0 + 2.0 * self.tau
        n = H.sum(axis=1)
        b = 1.0 / self.T
        num = np.log(np.maximum((H / g[:, None]).sum(axis=0), 1e-300))
        keep = (H.sum(axis=0) > 0)
        E, num = self.E[keep], num[keep]
        f = np.zeros(len(self.T)) if f0 is None else f0.copy()
        for _ in range(max_iter):
            log_den = _logsumexp(np.log(n / g)[:, None] + f[:, None] - b[:, None] * E[None, :], axis=0)
            log_omega = num - log_den
            f_new = -_logsumexp(log_omega[None, :] - b[:, None] * E[None, :], axis=1)
            f_new -= f_new[0]
            if np.max(np.abs(f_new - f)) < tol:
                f = f_new
                break
            f = f_new
        else:
            raise RuntimeError("multiple-histogram iteration did not converge")
        hk = H.sum(axis=0)[keep]
        return {"E": E, "log_omega": log_omega, "f": f,
                "o1": S1.sum(axis=0)[keep] / hk, "o2": S2.sum(axis=0)[keep] / hk,
                "o4": S4.sum(axis=0)[keep] / hk}

    @staticmethod
    def observables(sol: dict, temps) -> dict:
        """<|m|> and the Binder cumulant U = 1 - <m^4> / (3 <m^2>^2) at each temperature."""
        temps = np.atleast_1d(np.asarray(temps, dtype=float))
        lw = sol["log_omega"][None, :] - sol["E"][None, :] / temps[:, None]
        w = np.exp(lw - lw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        m1, m2, m4 = (w @ sol[k] for k in ("o1", "o2", "o4"))
        return {"M_abs": m1, "U": 1.0 - m4 / (3.0 * m2 ** 2)}

    def curves(self, temps) -> dict:
        """The reweighted <|m|> and U at ``temps`` and their block-jackknife samples."""
        temps = np.asarray(temps, dtype=float)
        if temps.min() < self.T.min() - 1e-12 or temps.max() > self.T.max() + 1e-12:
            raise ValueError("temperatures outside the simulated range")
        full = self.solve()
        out = self.observables(full, temps)
        jk = [self.observables(self.solve(drop=b, f0=full["f"]), temps) for b in range(self.n_blocks)]
        out["jackknife"] = {k: np.array([s[k] for s in jk]) for k in ("M_abs", "U")}
        return out


def jackknife_cov(samples: np.ndarray) -> np.ndarray:
    """Covariance of the full estimate from leave-one-block-out samples (rows)."""
    n = samples.shape[0]
    d = samples - samples.mean(axis=0)
    return (n - 1) / n * d.T @ d
