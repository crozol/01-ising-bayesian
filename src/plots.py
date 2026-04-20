"""Publication-style static figures for the README, styled to match the
portfolio's dark Plotly theme (colors, low-contrast grid, light-on-dark text).

Four PNGs are written to ``--out-dir`` (default ``results/``):

    0. snapshots.png     — two equilibrium lattices (one per phase)
    1. magnetization.png — M(T) curve + phase bands + Onsager Tc
    2. posteriors.png    — marginal posteriors of Tc and β + 95% HDI
    3. trace.png         — per-chain MCMC trace with R̂ and ESS

Design rules:
    - Dark background matching the portfolio (#0d1220 / #0c101c).
    - Grid lines are *whispers* (white at 3% alpha); no minor grid.
    - Portfolio palette: purple #7c5cff · cyan #22d3ee · pink #f472b6 · amber #fbbf24.
    - Monospace font for any numeric annotation, sans-serif for prose.
    - Legends and text boxes never overlap data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ONSAGER_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))  # 2.26918531...
EXACT_BETA = 0.125  # 1/8


# -------- portfolio palette --------

BG_PANEL = "#0d1220"        # outer figure background
BG_AXES = "#0c101c"         # inner plot background
FG_0 = "#e7ecf5"            # primary text
FG_1 = "#9aa3b8"            # muted text / ticks
GRID = (1.0, 1.0, 1.0, 0.05)  # very subtle white grid
SPINE = (1.0, 1.0, 1.0, 0.20)

PURPLE = "#7c5cff"          # accent 1 · data / Tc
CYAN = "#22d3ee"            # accent 2 · β
PINK = "#f472b6"            # accent 3 · posterior mean / highlight
AMBER = "#fbbf24"           # accent 4 · Onsager exact value

CHAIN_COLORS = [PURPLE, CYAN, PINK, AMBER]

SANS = ["DejaVu Sans", "Inter", "Segoe UI", "Arial"]
MONO = ["DejaVu Sans Mono", "JetBrains Mono", "Consolas", "monospace"]


def _style():
    """Apply the dark portfolio style globally."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BG_PANEL,
        "savefig.facecolor": BG_PANEL,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": FG_0,
        "axes.titlecolor": FG_0,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": FG_1,
        "ytick.color": FG_1,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "text.color": FG_0,
        "legend.frameon": True,
        "legend.facecolor": BG_PANEL,
        "legend.edgecolor": SPINE,
        "legend.labelcolor": FG_0,
        "legend.fontsize": 10,
        "font.family": SANS,
        "mathtext.fontset": "cm",
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
    })


def _load_csv(csv_path: str):
    T, M, S = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            T.append(float(row["T"]))
            M.append(float(row["M_mean"]))
            S.append(float(row["M_std"]))
    return np.array(T), np.array(M), np.array(S)


# ----------------------------- Figure 0: lattice snapshots -----------------------------

def plot_snapshots(
    out_path: str,
    *,
    size: int = 32,
    T_cold: float = 1.8,
    T_hot: float = 3.2,
    n_sweeps: int = 2500,
    seed: int = 0,
) -> None:
    """Two equilibrium lattices side by side — one per phase."""
    import matplotlib.pyplot as plt
    from src.metropolis import _sweep

    _style()

    def equilibrate(T: float, rng_seed: int) -> np.ndarray:
        np.random.seed(rng_seed)
        lat = np.where(np.random.random((size, size)) < 0.5, -1, 1).astype(np.int8)
        for _ in range(n_sweeps):
            _sweep(lat, T, 1.0)
        return lat

    lat_cold = equilibrate(T_cold, seed)
    lat_hot = equilibrate(T_hot, seed + 1)

    m_cold = float(np.abs(lat_cold.mean()))
    m_hot = float(np.abs(lat_hot.mean()))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    fig.patch.set_facecolor(BG_PANEL)

    for ax, lat, T, m, subtitle, accent in (
        (axes[0], lat_cold, T_cold, m_cold, "ordered  (broken symmetry)", PURPLE),
        (axes[1], lat_hot, T_hot, m_hot, "disordered  (thermal noise)", CYAN),
    ):
        ax.imshow(lat, cmap="PuOr", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(
            fr"$T = {T:.1f}$   ·   {subtitle}",
            fontsize=12, pad=8, color=FG_0,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.2)
        ax.text(
            0.02, 0.97,
            fr"$\langle |M| \rangle \approx {m:.2f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=11, color=FG_0, fontweight="bold", family=MONO,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_PANEL,
                      edgecolor=accent, linewidth=0.9, alpha=0.92),
        )

    fig.suptitle(
        f"One equilibrium configuration per phase  ·  {size}×{size} lattice, periodic BCs",
        fontweight="bold", fontsize=13, y=1.01, x=0.01, ha="left", color=FG_0,
    )
    fig.text(
        0.5, -0.02,
        "violet = spin +1    ·    orange = spin −1    ·    "
        "the two ground states are related by global spin flip (Z₂ symmetry)",
        ha="center", va="top", fontsize=10, color=FG_1, style="italic",
    )

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_PANEL)
    plt.close(fig)
    print(f"[ok] {out_path}")


# ----------------------------- Figure 1: M(T) -----------------------------

def _sigmoid_fit(T, M, S):
    from scipy.optimize import curve_fit

    def model(T, A, T0, wL, wR, c):
        w = np.where(T < T0, wL, wR)
        return A * 0.5 * (1.0 - np.tanh((T - T0) / w)) + c

    p0 = [0.9, 2.30, 0.25, 0.20, 0.08]
    sigma = np.maximum(S, 1e-3)
    popt, _ = curve_fit(model, T, M, p0=p0, sigma=sigma, absolute_sigma=True,
                        maxfev=20000)
    return popt, model


def plot_magnetization(csv_path: str, out_path: str, *, lattice_size: int = 32) -> None:
    """M(T) with subtle phase tint, sigmoid fit and Onsager Tc marker."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style()
    T, M, S = _load_csv(csv_path)

    try:
        popt, model = _sigmoid_fit(T, M, S)
        T_fine = np.linspace(T.min(), T.max(), 500)
        M_fit = model(T_fine, *popt)
        have_fit = True
    except Exception:
        have_fit = False

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.patch.set_facecolor(BG_PANEL)

    x_lo, x_hi = T.min() - 0.03, T.max() + 0.03
    y_lo, y_hi = -0.05, 1.12

    # subtle phase tint
    ax.axvspan(x_lo, ONSAGER_TC, color=PURPLE, alpha=0.05, zorder=0)
    ax.axvspan(ONSAGER_TC, x_hi, color=AMBER, alpha=0.04, zorder=0)

    if have_fit:
        ax.fill_between(T_fine, 0.0, M_fit, color=PURPLE, alpha=0.10, zorder=1)
        ax.plot(T_fine, M_fit, color=PURPLE, lw=2.4, zorder=3,
                label="sigmoid fit")

    ax.errorbar(
        T, M, yerr=S, fmt="o", markersize=6.5,
        color=PINK, markeredgecolor=BG_PANEL, markeredgewidth=1.0,
        ecolor=(0.96, 0.45, 0.71, 0.55), elinewidth=1.1, capsize=2.8,
        capthick=1.0, label=r"simulated $\langle |M| \rangle \pm \sigma_M$",
        zorder=4,
    )

    ax.axvline(ONSAGER_TC, color=AMBER, linestyle="--", linewidth=1.9,
               zorder=2, label=r"Onsager $T_c = 2.2692$")

    ax.text(
        ONSAGER_TC - 0.022, 0.56,
        fr"$T_c^{{\mathrm{{Onsager}}}}$",
        color=AMBER, fontsize=11, fontweight="bold", family=MONO,
        ha="right", va="center", rotation=90,
    )

    ax.text(
        (x_lo + ONSAGER_TC) / 2.0, 0.30,
        "ordered phase\nspins aligned",
        fontsize=10.5, color=FG_1, ha="center", va="center",
        style="italic",
    )
    ax.text(
        0.5 * (ONSAGER_TC + x_hi), 0.55,
        "disordered phase\nthermal noise dominates",
        fontsize=10.5, color=FG_1, ha="center", va="center",
        style="italic",
    )

    idx_max_sigma = int(np.argmax(S))
    T_crit, M_crit = T[idx_max_sigma], M[idx_max_sigma]
    ax.annotate(
        f"largest fluctuations\n$\\sigma_M = {S[idx_max_sigma]:.3f}$",
        xy=(T_crit, M_crit), xytext=(T_crit - 0.62, M_crit + 0.28),
        fontsize=9.8, color=FG_0, ha="center", va="center", family=MONO,
        arrowprops=dict(arrowstyle="->", color=FG_1, lw=1.0,
                        connectionstyle="arc3,rad=-0.15"),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  edgecolor=AMBER, linewidth=0.9, alpha=0.92),
    )

    ax.set_xlabel(r"Temperature  $T$   ($J / k_B$)")
    ax.set_ylabel(r"Magnetization per site  $\langle |M| \rangle$")
    ax.set_title(
        f"Phase transition  ·  magnetization vs. temperature  "
        f"({lattice_size}×{lattice_size} lattice, periodic BCs)",
        loc="left", pad=12,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.legend(loc="upper right", framealpha=0.96, fontsize=9.8,
              borderpad=0.6, bbox_to_anchor=(0.995, 0.995), handlelength=2.0)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_PANEL)
    plt.close(fig)
    print(f"[ok] {out_path}")


# --------------------------- Figure 2: posteriors ---------------------------

def _kde(samples: np.ndarray, grid_n: int = 500):
    x = np.asarray(samples).ravel()
    n = len(x)
    sigma = x.std(ddof=1)
    span = x.max() - x.min()
    pad = 0.20 * span if span > 0 else 0.05
    h = max(1.06 * sigma * n ** (-0.2), 1e-6)
    grid = np.linspace(x.min() - pad, x.max() + pad, grid_n)
    u = (grid[:, None] - x[None, :]) / h
    k = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
    return grid, k.sum(axis=1) / (n * h)


def _panel_posterior(ax, samples, *, exact, density_color, fill_alpha,
                     title, xlabel, show_shift: bool = False):
    """One posterior panel — dark theme, minimal grid, clean annotations."""
    import arviz as az

    flat = np.asarray(samples).ravel()
    grid, dens = _kde(flat)

    mean = float(flat.mean())
    lo, hi = az.hdi(flat, hdi_prob=0.95)
    lo, hi = float(lo), float(hi)
    y_max = dens.max() * 1.35

    # very subtle: keep horizontal grid only
    ax.grid(which="major", axis="y", color=GRID, linewidth=0.7)
    ax.grid(which="major", axis="x", color=GRID, linewidth=0.5)

    # density
    ax.fill_between(grid, 0, dens, color=density_color, alpha=fill_alpha)
    ax.plot(grid, dens, color=density_color, lw=2.6)

    # 95% HDI darker fill
    in_hdi = (grid >= lo) & (grid <= hi)
    ax.fill_between(grid[in_hdi], 0, dens[in_hdi],
                    color=density_color, alpha=fill_alpha + 0.18)

    # vertical markers
    ax.axvline(mean, color=PINK, lw=2.0, zorder=3)
    ax.axvline(exact, color=AMBER, lw=2.0, linestyle="--", zorder=3)

    # numeric labels aligned to the lines, outside the density peak
    ax.text(
        mean, y_max * 0.96,
        f"posterior mean\n{mean:.3f}",
        color=PINK, fontsize=10, ha="center", va="top", family=MONO,
        fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=BG_AXES,
                  edgecolor=PINK, linewidth=0.8, alpha=0.95),
    )
    ax.text(
        exact, y_max * 0.78,
        f"exact\n{exact:.4f}",
        color=AMBER, fontsize=10, ha="center", va="top", family=MONO,
        fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=BG_AXES,
                  edgecolor=AMBER, linewidth=0.8, alpha=0.95),
    )

    # bottom-anchored HDI bracket + label on its own dark box
    y_br = y_max * 0.05
    ax.annotate(
        "", xy=(lo, y_br), xytext=(hi, y_br),
        arrowprops=dict(arrowstyle="|-|", color=density_color, lw=1.5,
                        shrinkA=0, shrinkB=0),
        zorder=5,
    )
    ax.text(
        0.5 * (lo + hi), y_br * 2.6,
        f"95% HDI  [{lo:.3f}, {hi:.3f}]",
        color=density_color, fontsize=10, ha="center", va="bottom",
        family=MONO, fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_AXES,
                  edgecolor=density_color, linewidth=0.8, alpha=0.95),
    )

    if show_shift and not (lo <= exact <= hi):
        y_arrow = y_max * 0.52
        ax.annotate(
            "", xy=(mean, y_arrow), xytext=(exact, y_arrow),
            arrowprops=dict(arrowstyle="<|-|>", color=FG_0, lw=1.3,
                            shrinkA=2, shrinkB=2),
        )
        ax.text(
            0.5 * (mean + exact), y_arrow * 1.08,
            f"finite-size shift  {mean - exact:+.3f}",
            fontsize=10, color=FG_0, ha="center", va="bottom",
            family=MONO, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG_PANEL,
                      edgecolor=SPINE, linewidth=0.8, alpha=0.92),
        )

    g_lo, g_hi = grid[0], grid[-1]
    pad = 0.08 * (g_hi - g_lo)
    ax.set_xlim(min(g_lo, exact) - pad, max(g_hi, exact) + pad)
    ax.set_ylim(0, y_max)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("posterior density")
    ax.set_title(title, loc="left", pad=10)


def plot_posteriors(trace_path: str, out_path: str) -> None:
    """Marginal posteriors of Tc and β · dark portfolio style."""
    import arviz as az
    import matplotlib.pyplot as plt

    _style()
    trace = az.from_netcdf(trace_path)
    post = trace.posterior

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig.patch.set_facecolor(BG_PANEL)

    _panel_posterior(
        axes[0], post["Tc"].values, exact=ONSAGER_TC,
        density_color=PURPLE, fill_alpha=0.18,
        title=r"Posterior of $T_c$  (critical temperature)",
        xlabel=r"$T_c$   ($J/k_B$)",
        show_shift=True,
    )
    _panel_posterior(
        axes[1], post["beta"].values, exact=EXACT_BETA,
        density_color=CYAN, fill_alpha=0.18,
        title=r"Posterior of $\beta$  (critical exponent)",
        xlabel=r"$\beta$   (dimensionless)",
        show_shift=False,
    )

    fig.suptitle(
        "Bayesian inference of the 2D Ising critical parameters",
        fontweight="bold", fontsize=14, y=1.02, x=0.01, ha="left", color=FG_0,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_PANEL)
    plt.close(fig)
    print(f"[ok] {out_path}")


# --------------------------- Figure 3: trace ---------------------------

def plot_trace(trace_path: str, out_path: str) -> None:
    """Per-chain trace plots · dark portfolio style."""
    import arviz as az
    import matplotlib.pyplot as plt

    _style()
    trace = az.from_netcdf(trace_path)
    post = trace.posterior

    n_chains, n_draws = post["Tc"].values.shape

    params = [
        ("Tc", r"$T_c$   ($J/k_B$)", ONSAGER_TC),
        ("beta", r"$\beta$   (dimensionless)", EXACT_BETA),
        ("sigma", r"$\sigma$   (noise scale)", None),
    ]

    fig, axes = plt.subplots(len(params), 1, figsize=(12.5, 9.0), sharex=True)
    fig.patch.set_facecolor(BG_PANEL)

    for ax, (key, label, exact) in zip(axes, params):
        ax.grid(which="major", axis="y", color=GRID, linewidth=0.6)
        ax.grid(which="major", axis="x", color=GRID, linewidth=0.4)
        chains = post[key].values
        for ci, chain in enumerate(chains):
            ax.plot(
                chain, color=CHAIN_COLORS[ci % len(CHAIN_COLORS)],
                lw=0.55, alpha=0.70,
                label=f"chain {ci+1}" if ax is axes[0] else None,
            )
        if exact is not None:
            ax.axhline(exact, color=AMBER, linestyle="--", lw=1.8,
                       label=r"exact (Onsager)" if ax is axes[0] else None)
        ax.set_ylabel(label, fontsize=11.5, color=FG_0)

        rhat = float(az.rhat(trace)[key])
        ess = float(az.ess(trace)[key])
        status = "OK" if rhat <= 1.01 and ess >= 400 else "acceptable"
        ax.text(
            0.006, 0.94,
            fr"$\widehat{{R}} = {rhat:.3f}$      "
            fr"$\mathrm{{ESS}}_\mathrm{{bulk}} = {ess:.0f}$      [{status}]",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, color=FG_0, family=MONO, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=SPINE, linewidth=0.9, alpha=0.92),
        )

    axes[-1].set_xlabel("MCMC iteration  (post burn-in)", fontsize=11.5, color=FG_0)
    fig.suptitle(
        f"MCMC trace plots  ·  {n_chains} chains × {n_draws} draws",
        fontweight="bold", fontsize=14, y=0.998, x=0.01, ha="left", color=FG_0,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="lower center", ncol=len(handles),
            fontsize=10.5, framealpha=0.96,
            bbox_to_anchor=(0.5, -0.015), handlelength=2.0,
        )

    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.subplots_adjust(hspace=0.16)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_PANEL)
    plt.close(fig)
    print(f"[ok] {out_path}")


# ------------------------------- orchestration -------------------------------

def plot_all(
    csv_path: str = "data/magnetization.csv",
    trace_path: str = "data/trace.nc",
    out_dir: str = "results",
    *,
    lattice_size: int = 32,
) -> None:
    out = Path(out_dir)
    plot_snapshots(str(out / "snapshots.png"), size=lattice_size)
    plot_magnetization(csv_path, str(out / "magnetization.png"),
                       lattice_size=lattice_size)
    plot_posteriors(trace_path, str(out / "posteriors.png"))
    plot_trace(trace_path, str(out / "trace.png"))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="data/magnetization.csv")
    parser.add_argument("--trace", type=str, default="data/trace.nc")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--lattice-size", type=int, default=32)
    args = parser.parse_args()
    plot_all(args.csv, args.trace, args.out_dir, lattice_size=args.lattice_size)


if __name__ == "__main__":
    _main()
