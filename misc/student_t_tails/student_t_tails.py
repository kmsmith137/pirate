#!/usr/bin/env python3
"""Upper-tail probabilities of the studentized statistic y = x/sigma_hat.

Generates plans/figs/student_t_tails.{png,pdf}, the figure embedded in
plans/student_t_tails.md.  Run from the pirate/ directory (or anywhere -- output
paths are relative to this file).
"""

import pathlib

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
from scipy.stats import norm, t


# ---- palette ------------------------------------------------------------
# Single-hue blue ramp, light->dark with increasing departure from Gaussian
# (i.e. with decreasing nu).  Validated as an ordinal ramp on the light
# surface: monotone L, adjacent dL >= 0.06, light end 2.06:1 vs surface.

SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_2     = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
AXIS      = '#c3c2b7'

NUS    = [250, 500, 1000, 2000, 4000]
COLORS = {250: '#0d366b', 500: '#1c5cab', 1000: '#2a78d6', 2000: '#5598e7',
          4000: '#86b6ef'}

U_MAX = 12.0
Y_MIN = 1.0e-34


def make_figure(path_stem):
    u = np.linspace(0.0, U_MAX, 2401)

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # t curves first (largest tail on top of the legend), Gaussian last so the
    # dashed reference line draws over the bundle it is the limit of.
    for nu in NUS:
        ax.semilogy(u, t.sf(u, df=nu), lw=2.0, color=COLORS[nu],
                    solid_capstyle='round', label=rf'$\nu = {nu}$')

    ax.semilogy(u, norm.sf(u), lw=2.0, color=INK, ls=(0, (5, 3)),
                dash_capstyle='round', label=r'Gaussian ($\nu = \infty$)')

    # ---- scales & chrome ------------------------------------------------
    ax.set_xlim(0.0, U_MAX)
    ax.set_ylim(Y_MIN, 1.5)

    # Decade gridlines, but a tick label only every 4th decade -- 34 labels
    # would be unreadable.
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=40))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([10.0**-k for k in range(0, 34, 4)], minor=False)
    ax.set_yticks([10.0**-k for k in range(0, 34) if k % 4], minor=True)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(True, which='major', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(which='both', colors=MUTED, labelcolor=INK_2, length=4,
                   width=0.8)
    ax.tick_params(which='minor', length=2.5)

    ax.set_xlabel(r'threshold $u$', color=INK_2, fontsize=12, labelpad=8)
    ax.set_ylabel(r'$P(y > u)$   (one-sided)', color=INK_2, fontsize=12,
                  labelpad=8)
    ax.set_title('Tail of $y = x/\\hat\\sigma$ for a $\\nu$-dof variance estimate',
                 color=INK, fontsize=14, pad=14, loc='left')

    # ---- legend ---------------------------------------------------------
    # The curves are a tight bundle (< 1 decade apart at fixed u), so direct
    # labels cannot be placed unambiguously.  Legend order matches the
    # top-to-bottom order of the curves at the right edge of the plot.
    leg = ax.legend(loc='upper right', frameon=False, fontsize=11.5,
                    labelcolor=INK_2, handlelength=2.6, borderaxespad=1.2,
                    labelspacing=0.7,
                    title='top $\\to$ bottom at right edge')
    leg.get_title().set_color(MUTED)
    leg.get_title().set_fontsize(10.5)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path_stem}.{ext}', dpi=200, facecolor=SURFACE)
    plt.close(fig)


def print_table():
    """Numbers quoted in the accompanying markdown."""
    print(f'{"u":>4}  {"Gaussian":>10}  '
          + '  '.join(f'nu={nu:<9d}' for nu in NUS)
          + f'   ratio({NUS[0]})')
    for u in (4, 5, 6, 7, 8, 9, 10, 11, 12):
        g = norm.sf(u)
        row = [t.sf(u, df=nu) for nu in NUS]
        print(f'{u:4d}  {g:10.3e}  '
              + '  '.join(f'{r:11.3e}' for r in row)
              + f'   {row[0] / g:10.3g}')

    print()
    print('threshold u_nu giving the same tail probability as a Gaussian u,')
    print('exact vs the two-term Cornish-Fisher expansion:')
    print(f'{"u":>4}  ' + '  '.join(f'nu={nu:<15d}' for nu in NUS))
    for u in (5, 6, 8, 10, 12):
        p = norm.sf(u)
        cells = []
        for nu in NUS:
            cf = (u + (u**3 + u) / (4 * nu)
                  + (5 * u**5 + 16 * u**3 + 3 * u) / (96 * nu**2))
            cells.append(f'{t.isf(p, df=nu):7.3f} /{cf:7.3f}')
        print(f'{u:4d}  ' + '  '.join(cells))

    print()
    print('ln[S_nu(u)/Sgauss(u)] exact vs the u^4/(4 nu) approximation:')
    print(f'{"u":>4}  ' + '  '.join(f'nu={nu:<17d}' for nu in NUS))
    for u in (6, 7, 8, 9, 10, 11, 12):
        cells = []
        for nu in NUS:
            exact = np.log(t.sf(u, df=nu) / norm.sf(u))
            approx = u**4 / (4.0 * nu)
            cells.append(f'{exact:6.3f} /{approx:6.3f} ({approx / exact - 1:+5.1%})')
        print(f'{u:4d}  ' + '  '.join(cells))

    print()
    print('nu needed to hold S_nu(u) within a factor (1+eps) of Gaussian,')
    print('from nu >= u^4 / (4 ln(1+eps)):')
    print(f'{"u":>4}  {"eps=10%":>10}  {"factor 2":>10}')
    for u in (5, 6, 8, 10, 12):
        print(f'{u:4d}  ' + '  '.join(
            f'{u**4 / (4 * np.log(1 + eps)):10.1e}' for eps in (0.1, 1.0)))


def check_studentized_residual(n=50, ntrial=2_000_000, seed=137):
    """Monte Carlo check of the "x is one of the samples" claim in the markdown.

    Claim: with sigma_hat^2 the unbiased sample variance of the same n samples
    that contain x_i, the quantity n/(n-1)^2 * y^2 is Beta(1/2, (n-2)/2).  Small
    n is used deliberately -- the deviation from t_{n-1} is what is being
    checked, and it is invisible at n = 1000.
    """
    from scipy.stats import beta, kstest

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((ntrial, n))
    y = (z[:, 0] - z.mean(axis=1)) / z.std(axis=1, ddof=1)
    w = n * y**2 / (n - 1) ** 2

    ks = kstest(w, beta(0.5, (n - 2) / 2).cdf)
    print()
    print(f'studentized-residual check (n={n}, {ntrial} trials):')
    print(f'  KS vs Beta(1/2,(n-2)/2): D={ks.statistic:.5f}  p={ks.pvalue:.3f}')
    print(f'  max |y| observed {np.abs(y).max():.4f}, bound (n-1)/sqrt(n) '
          f'= {(n - 1) / np.sqrt(n):.4f}')


if __name__ == '__main__':
    here = pathlib.Path(__file__).resolve().parent
    figs = here / 'figs'
    figs.mkdir(exist_ok=True)
    make_figure(str(figs / 'student_t_tails'))
    print_table()
    check_studentized_residual()
