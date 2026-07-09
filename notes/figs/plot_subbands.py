#!/usr/bin/env python3
"""Generate notes/figs/chord_subbands.pdf.

Shows the CHORD frequency subbands (configs/dedispersion/chord_sb2.yml) in two
coordinate systems:

  - top panel:  "tree-freq" ranges, on a LINEAR scale;
  - bottom panel: the original frequency coordinate (MHz), on a LOG scale.

Subbands are grouped and colored by peak-finding level l (the notes' notation).
The frequency ranges are generated procedurally from the config (no hardcoded
numbers): subband index-ranges come from pirate's FrequencySubbands, and the
tree-freq -> MHz map comes from the config's channel_map + index_to_frequency.

Run:  python3 notes/figs/plot_subbands.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from pirate_frb import DedispersionConfig
from pirate_frb.cuda_generator.FrequencySubbands import FrequencySubbands

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(_HERE, "..", "..", "configs", "dedispersion", "chord_sb2.yml")
OUTPUT = os.path.join(_HERE, "chord_subbands.pdf")

# One color per peak-finding level l (palette carried over from the talk slide).
LEVEL_COLORS = ["#1f4e8c", "#c44e1f", "#2e8b3d", "#8a4fbf", "#d4a017"]


def collect_subbands(config):
    """Procedurally build the subband list from the config.

    Returns (r, R, coarse, subbands), where subbands is a list of dicts with
    keys: level, tlo/thi (tree-freq index range, 0..2^r) and freq_lo/freq_hi (frequency
    range in MHz).
    """
    r = config.toplevel_tree_rank
    counts = list(config.frequency_subband_counts)
    R = len(counts) - 1
    fs = FrequencySubbands(counts)
    coarse = 1 << (r - R)                                   # tree-freq channels per coarse channel
    cm = np.asarray(config.make_channel_map(), dtype=float) # tree-freq index -> freq-channel index
    nfreq = float(config.get_total_nfreq())

    def tree_to_mhz(t):
        fch = min(max(float(cm[t]), 0.0), nfreq)
        return config.index_to_frequency(fch)

    subbands = []
    for level in range(R + 1):
        for b in range(counts[level]):
            flo, fhi = fs.get_band_index_range(level, b)   # coarse (f) units, 0..2^R
            tlo, thi = flo * coarse, fhi * coarse          # tree-freq index, 0..2^r
            freq_a, freq_b = tree_to_mhz(tlo), tree_to_mhz(thi)  # MHz (freq decreases with tree-freq)
            subbands.append(dict(level=level, tlo=tlo, thi=thi,
                                 freq_lo=min(freq_a, freq_b), freq_hi=max(freq_a, freq_b)))

    # Coarse-channel (i-grid) boundaries, in tree-freq index and in MHz.
    bnd_t = [i * coarse for i in range(2 ** R + 1)]
    bnd_mhz = [tree_to_mhz(t) for t in bnd_t]
    return r, R, coarse, subbands, bnd_t, bnd_mhz


def main(output=OUTPUT):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.7,
    })

    config = DedispersionConfig.from_yaml(CONFIG)
    r, R, coarse, subbands, bnd_t, bnd_mhz = collect_subbands(config)
    N = 1 << r

    # Lay out one row per subband, grouped by level (l=0 at top), gap between levels.
    row_hh = 0.34       # half-height of each subband's shaded rectangle
    gap = 1.0
    yrow = {}
    yticks, ylevels = [], []
    cursor = 0.0
    for level in range(R + 1):
        members = [i for i, sb in enumerate(subbands) if sb["level"] == level]
        if not members:
            continue
        y0 = cursor
        for i in members:
            yrow[i] = cursor
            cursor += 1.0
        yticks.append(y0 + 0.5 * (len(members) - 1))
        ylevels.append(level)
        cursor += gap
    ytop, ybot = -0.7, max(yrow.values()) + 0.7

    fig, (ax_t, ax_f) = plt.subplots(
        2, 1, figsize=(6.5, 7.2),
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.26))

    def decorate_y(ax):
        ax.set_ylim(ybot, ytop)                  # inverted -> l=0 on top
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$l={k}$" for k in ylevels])
        for tick, k in zip(ax.get_yticklabels(), ylevels):
            tick.set_color(LEVEL_COLORS[k])
        ax.tick_params(axis="y", length=0)

    def draw_bars(ax, lo_key, hi_key):
        for i, sb in enumerate(subbands):
            c = LEVEL_COLORS[sb["level"]]
            yy, lo, hi = yrow[i], sb[lo_key], sb[hi_key]
            ax.fill_between([lo, hi], yy - row_hh, yy + row_hh,
                            facecolor=c, alpha=0.14, edgecolor="none", zorder=2)
            ax.plot([lo, hi], [yy, yy], color=c, lw=2.2,
                    solid_capstyle="round", zorder=3)

    # --- Top panel: tree-freq, linear ---
    for t in bnd_t:
        ax_t.axvline(t, color="0.90", lw=0.5, zorder=0)
    draw_bars(ax_t, "tlo", "thi")
    decorate_y(ax_t)
    ax_t.set_xlim(0, N)
    ax_t.set_xticks(np.linspace(0, N, 5))
    ax_t.set_xticklabels([f"{int(v)}" for v in np.linspace(0, N, 5)])
    ax_t.set_xlabel(r"tree-freq index  (linear,  $0 \leq f < 2^{%d}$)" % r)

    # --- Bottom panel: frequency, log (increasing left-to-right) ---
    for fmhz in bnd_mhz:
        ax_f.axvline(fmhz, color="0.90", lw=0.5, zorder=0)
    draw_bars(ax_f, "freq_lo", "freq_hi")
    decorate_y(ax_f)
    f_hi = max(sb["freq_hi"] for sb in subbands)
    f_lo = min(sb["freq_lo"] for sb in subbands)
    ax_f.set_xscale("log")
    ax_f.set_xlim(f_lo, f_hi)
    fticks = [t for t in (300, 400, 500, 600, 800, 1000, 1500) if f_lo - 1 <= t <= f_hi + 1]
    ax_f.set_xticks(fticks)
    ax_f.xaxis.set_major_formatter(ScalarFormatter())
    ax_f.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_f.set_xlabel("frequency (MHz, log scale)")

    fig.savefig(output, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
