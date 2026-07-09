#!/usr/bin/env python3
"""Generate notes/figs/early_triggers.pdf.

Illustrates the early-trigger geometry for configs/dedispersion/chord_sb2_et.yml,
showing a single dispersed burst in two coordinate systems:

  - left panel:  the "tree-freq" plane (tree-freq index vs time), where the
                 burst is a straight, positive-slope line;
  - right panel: the radio-frequency plane (MHz, LINEAR scale, vs time), where
                 the same burst is the familiar nu^-2 dispersion sweep.

In both panels the early triggers e=0,1,2,3 are dashed horizontal lines: an
early trigger of level e dedisperses only the top 2^(r-e) tree-freq
channels, so its dedispersed sum is complete when the burst sweeps down to that
line. Time is in units of the full-band dispersion delay D, so the lines are
crossed at t/D = 2^-e (the trigger's latency fraction).

The trigger tree-freq indices (2^(r-e)) and frequencies
(delay_to_frequency(2^(r-e))) are read from the config; nothing is hardcoded.

Run:  python3 notes/figs/plot_early_triggers.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pirate_frb import DedispersionConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(_HERE, "..", "..", "configs", "dedispersion", "chord_sb2_et.yml")
OUTPUT = os.path.join(_HERE, "early_triggers.pdf")

# One color per early-trigger level e (shared with plot_tree_segments.py).
ET_COLORS = ["#1f4e8c", "#2e8b3d", "#c44e1f", "#8a4fbf"]


def main(output=OUTPUT):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
    })

    config = DedispersionConfig.from_yaml(CONFIG)
    r = config.toplevel_tree_rank
    N = 1 << r                       # 2^r tree-freq channels
    et_levels = [0, 1, 2, 3]         # early-trigger levels illustrated

    # Trigger lines: et_level -> (tree-freq index, radio frequency).
    trig_idx = {e: 1 << (r - e) for e in et_levels}          # 2^(r-e)
    trig_mhz = {e: config.delay_to_frequency(trig_idx[e]) for e in et_levels}
    fhi = config.zone_freq_edges[-1]
    flo = config.zone_freq_edges[0]

    # Burst track, parameterized by tau = t/D in [0,1]: at tau the burst has swept
    # down to tree-freq index tau*N, i.e. radio frequency delay_to_frequency(tau*N).
    tau = np.linspace(1e-4, 1.0, 400)
    track_idx = tau * N
    track_mhz = np.array([config.delay_to_frequency(min(t, N)) for t in track_idx])

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    fig.subplots_adjust(wspace=0.36, bottom=0.13, top=0.88)

    def decorate_x(ax):
        ax.set_xlim(0, 1.06)
        ax.set_xticks([])
        ax.set_xlabel("Time")

    # --- Left panel: tree-freq index (linear), burst is a straight line ---
    ax_t.plot([0, 1], [0, N], color="0.15", lw=2.0, zorder=4)
    ymax_t = 1.08 * N
    for e in et_levels:
        y = trig_idx[e]
        c = ET_COLORS[e]
        ax_t.axhline(y, ls="--", lw=1.1, color=c, zorder=2,
                     label=rf"$e={e}$  ({2 ** (6 - e)}K tree-freqs)")
        ax_t.plot(2.0 ** (-e), y, "o", ms=4.5, color=c, zorder=5)  # trigger fires here
    decorate_x(ax_t)
    ax_t.set_ylim(0, ymax_t)
    ax_t.set_yticks([0, N // 8, N // 4, N // 2, N])
    ax_t.set_yticklabels(["0", "8K", "16K", "32K", "64K"])
    ax_t.set_ylabel(r"tree-freq index  ($0\leq f<2^{%d}$)" % r)
    ax_t.set_title("(tree-freq, time) plane", fontsize=10)
    # Legend in the upper-left, its top just below the e=0 line (order 0 -> 3).
    ax_t.legend(loc="upper left", bbox_to_anchor=(0.03, (N / ymax_t) - 0.03),
                fontsize=8, frameon=True, framealpha=0.9, edgecolor="0.8",
                handlelength=1.7, borderpad=0.5, labelspacing=0.4)

    # --- Right panel: radio frequency (MHz, linear), burst is a nu^-2 sweep ---
    ax_f.plot(tau, track_mhz, color="0.15", lw=2.0, zorder=4)
    for e in et_levels:
        y = trig_mhz[e]
        c = ET_COLORS[e]
        ax_f.axhline(y, ls="--", lw=1.1, color=c, zorder=2,
                     label=rf"$e={e}$  ({trig_mhz[e]:.0f} MHz trigger)")
        ax_f.plot(2.0 ** (-e), y, "o", ms=4.5, color=c, zorder=5)
    decorate_x(ax_f)
    ax_f.set_ylim(flo - 12, fhi)   # dip below the band bottom so the e=0 line is visible
    ax_f.set_yticks([300, 600, 900, 1200, 1500])
    ax_f.set_ylabel("radio frequency (MHz)")
    ax_f.set_title("(radio-frequency, time) plane", fontsize=10)
    # Legend in the upper-right (order 3 -> 0).
    h, l = ax_f.get_legend_handles_labels()
    ax_f.legend(h[::-1], l[::-1], loc="upper right", fontsize=8,
                frameon=True, framealpha=0.9, edgecolor="0.8",
                handlelength=1.7, borderpad=0.5, labelspacing=0.4)

    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
