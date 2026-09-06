#!/usr/bin/env python3
"""Generate notes/figs/detrender_response.pdf.

The Fourier-space response F(omega) of five detrender configurations, as defined
in the appendix "Fourier-space response of the detrenders" of notes/detrending.tex.
F is the response of the DETRENDER, i.e. of the residual r = d - fhat: F = 1
means a sinusoid at that frequency survives untouched, F = 0 means it is entirely
absorbed into the baseline.

Three of the curves are stationary detrenders, for which F is a single function
given in closed form by the appendix:

    local polynomial subtraction, n=1, W=128
    local polynomial subtraction, n=2, W=256      (the shipped configuration)
    Kalman filter, k=2, tau=64                    (the L -> infinity limit)

The other two are the CHIME time detrender (rf_pipelines polynomial_detrender,
polydeg=4, nt_chunk=1024), which fits each disjoint 1024-sample block separately
and so is NOT time-invariant.  A single F does not exist for it; the appendix
defines two, and both are plotted.

Nothing here imports pirate_frb: every curve is a closed form, so the figure can
be regenerated with numpy alone.

Run:  python3 notes/figs/plot_detrender_response.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(_HERE, "detrender_response.pdf")

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"


# ----------------------------------------------------------- stationary cases

def dirichlet(W, w):
    """D_W(omega) = sum_{s=-W}^{W} exp(-i omega s), real and even."""
    return np.sin((2*W + 1)*w/2) / np.sin(w/2)


def F_lps(W, n, w):
    """
    Local polynomial subtraction, degree n, half-width W, on a fully valid mask.

    The equivalent kernel is the boxcar at n=1 and a + b s^2 at n=2; see the
    appendix for a, b.  Only n = 1, 2 are implemented, which is all the figure
    needs.
    """
    N = 2*W + 1.0
    D = dirichlet(W, w)
    if n == 1:
        return 1.0 - D/N
    if n != 2:
        raise ValueError(f"F_lps: n={n} not implemented (only n=1,2)")
    P = W*(W + 1.0)
    a = 3*(3*P - 1) / (N*(4*P - 3))
    b = -15.0 / (N*(4*P - 3))
    # S_W(omega) = sum_s s^2 exp(-i omega s) = -D_W''(omega)
    S = ((N**2 + 1)/4.0)*D + (N*np.cos(N*w/2)*np.cos(w/2) - D)/(2*np.sin(w/2)**2)
    return 1.0 - (a*D + b*S)


def F_kf(tau, w, k=2):
    """Fixed-lag Kalman filter in the two-sided (L -> infinity) limit."""
    z = (2*tau*np.sin(w/2))**(2*k)
    return z/(1.0 + z)


# ------------------------------------------------------------- the CHIME case

class BlockPolyDetrender:
    """
    CHIME's time detrender: subtract a degree-'polydeg' fit from each disjoint
    block of 'nt' samples, on a fixed lattice.  Uniform weights, so the operator
    R = I - P is an exact orthogonal projection.

    R is periodically time-varying, so a sinusoid comes back as a comb and there
    is no single response.  F_coh is the symbol of the phase-averaged operator
    and F_pow the symbol of the phase-averaged R^T R; both reduce to one block.
    """

    def __init__(self, polydeg=4, nt=1024):
        self.polydeg, self.nt = polydeg, nt
        # The kernel's block coordinate: z runs over (-1, 1) across the block.
        z = (np.arange(nt) - 0.5*(nt - 1)) * (2.0/nt)
        # QR of the Legendre design matrix gives an orthonormal basis for
        # degree <= polydeg, so the projection is Q Q^T with no solve.
        self.Q, _ = np.linalg.qr(np.polynomial.legendre.legvander(z, polydeg))

    def F_coh(self, w):
        """|A_0|: the amplitude that survives AT the input frequency."""
        w = np.atleast_1d(w)
        u = np.exp(1j*np.outer(w, np.arange(self.nt)))
        return np.maximum(1.0 - (np.abs(u @ self.Q)**2).sum(axis=1)/self.nt, 0.0)

    def F_pow(self, w):
        """Total surviving power, at any frequency.  Equals sqrt(F_coh)."""
        return np.sqrt(self.F_coh(w))


# -------------------------------------------------------------------- figure

def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": INK3, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "pdf.fonttype": 42,
    })

    w = np.logspace(-3, np.log10(np.pi), 4000)
    chime = BlockPolyDetrender(polydeg=4, nt=1024)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(w, np.abs(F_lps(128, 1, w)), color=BLUE, lw=1.6,
            label=r"local poly  $n=1$,  $W=128$")
    ax.plot(w, np.abs(F_lps(256, 2, w)), color=ORANGE, lw=1.6,
            label=r"local poly  $n=2$,  $W=256$")
    ax.plot(w, F_kf(64.0, w), color=AQUA, lw=1.6,
            label=r"Kalman  $k=2$,  $\tau=64$,  $L\to\infty$")
    ax.plot(w, chime.F_coh(w), color=VIOLET, lw=1.7,
            label=r"CHIME  polydeg 4, $n_t=1024$:  $F_{\rm coh}$")
    ax.plot(w, chime.F_pow(w), color=VIOLET, lw=1.4, ls=(0, (3, 2)),
            label=r"CHIME  polydeg 4, $n_t=1024$:  $F_{\rm pow}$")

    ax.axhline(0.5, color=INK3, lw=0.7, ls=(0, (1, 2.5)))
    ax.axhline(1.0, color=INK3, lw=0.7, ls=(0, (1, 2.5)))
    ax.text(2.9, 0.53, "half power", color=INK3, fontsize=8, ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, np.pi)
    ax.set_ylim(1e-6, 2.0)
    ax.grid(True, which="major", color="#e8e7e3", lw=0.6)
    ax.grid(True, which="minor", color="#f3f2ef", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(r"angular frequency  $\omega$  [rad/sample]")
    ax.set_ylabel(r"detrender response  $F(\omega)$")
    ax.legend(loc="lower right", fontsize=8.2)

    fig.tight_layout()
    fig.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
