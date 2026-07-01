"""
plot_pulses: a throwaway script from visually debugging the pulse simulation. Ported from
simpulse/visual_check/plot-pulses.py. make_plots() writes plot1.png, plot2.png, plot3.png into
the current directory; the reference_plot*.png files in the original repo
(../extern/simpulse/visual_check/) show what the pulse shapes should look like.

Since the time sampling is now a construction parameter, each pulse is built at TWO fixed
time_sample_ms values (framed identically near t=0): a coarse one (~100 samples, drawn solid so the
individual time-sample rectangles are visible) and a fine one (~1000 samples, drawn dotted as a
smooth reference). This mirrors the original plots.

Run it via 'python -m pirate_frb test_simpulse'. matplotlib is imported lazily inside make_plots(),
so that 'import pirate_frb.simpulse' does not require matplotlib.
"""

import numpy as np

from .. import simpulse


def _make_pulse(freq_lo_MHz, freq_hi_MHz, nfreq, dm, sm, intrinsic_width, fluence, spectral_index,
                nsamp):
    """Build a SinglePulse with equal-width channels, framed near t=0 and sampled at ~nsamp samples
    across the pulse. The framing (extent/undispersed_arrival_time_sec) is independent of nsamp, so
    coarse- and fine-nsamp pulses overlay on the time axis."""
    d_hi = simpulse.dispersion_delay(dm, freq_hi_MHz)
    d_lo = simpulse.dispersion_delay(dm, freq_lo_MHz)

    # Generous leading/trailing margins (cover intrinsic width, scattering, and intra-channel
    # dispersion smearing) so the pulse sits fully at t > 0 on the zero-based grid.
    lead = 0.1*(d_lo - d_hi) + 4.0*intrinsic_width + simpulse.scattering_time(sm, freq_hi_MHz) + 1e-6
    trail = 0.1*(d_lo - d_hi) + 4.0*intrinsic_width + 10.0*simpulse.scattering_time(sm, freq_lo_MHz) + 1e-6
    extent = (d_lo - d_hi) + lead + trail

    edges = np.linspace(freq_lo_MHz, freq_hi_MHz, nfreq + 1)
    return simpulse.SinglePulse(
        internal_nt = 1024,
        time_sample_ms = 1.0e3 * extent / nsamp,
        freq_edges_MHz = edges,
        dm = dm, sm = sm, intrinsic_width = intrinsic_width,
        fluence = fluence, spectral_index = spectral_index,
        undispersed_arrival_time_sec = -d_hi + lead)


def make_plot(plt, matplotlib, pulse_args, ifreq_list, color_list, label_list, filename):
    assert len(ifreq_list) == len(color_list) == len(label_list)

    # Coarse (solid; shows the time-sample rectangles) + fine (dotted; smooth reference).
    for (nsamp, ls, labelled) in [(100, '-', True), (1000, ':', False)]:
        sp = _make_pulse(nsamp=nsamp, **pulse_args)
        out_nt = sp.nt_min
        ts = np.zeros((sp.nfreq, out_nt), dtype=np.float32)
        sp.add_to_timestream(ts)

        # Sample 'it' spans [it*dt, (it+1)*dt]; draw each channel as a histogram (step) vs time in ms.
        edges_ms = np.arange(out_nt + 1) * sp.time_sample_ms
        x = np.zeros(2*out_nt)
        x[0::2] = edges_ms[:-1]
        x[1::2] = edges_ms[1:]
        y = np.zeros(2*out_nt)

        for (ifreq, color, label) in zip(ifreq_list, color_list, label_list):
            y[0::2] = ts[ifreq, :]
            y[1::2] = ts[ifreq, :]

            font = {'family': 'serif', 'size': 14}
            matplotlib.rc('font', **font)
            if labelled:
                plt.plot(x, y, color=color, ls=ls, label=label)
            else:
                plt.plot(x, y, color=color, ls=ls)

    plt.xlabel('time (ms)')
    plt.legend(loc='upper right').draw_frame(False)
    plt.savefig(filename)
    plt.clf()
    print('wrote', filename)


def plot1(plt, matplotlib):
    """Some Gaussians with a little bit of dispersion and scattering (low-freq channel scatters)."""
    make_plot(plt, matplotlib,
              dict(freq_lo_MHz=1000.0, freq_hi_MHz=2000.0, nfreq=512,
                   dm=10.0, sm=4.0, intrinsic_width=0.005, fluence=30.0, spectral_index=2.0),
              ifreq_list=[0, 256, 511],
              color_list=['r', 'b', 'm'],
              label_list=['1 GHz', '1.5 GHz', '2.0 GHz'],
              filename='plot1.png')


def plot2(plt, matplotlib):
    """Boxcars (pure dispersion; no scattering or intrinsic width)."""
    make_plot(plt, matplotlib,
              dict(freq_lo_MHz=1000.0, freq_hi_MHz=2000.0, nfreq=7,
                   dm=10.0, sm=0.0, intrinsic_width=0.0, fluence=1.0, spectral_index=0.0),
              ifreq_list=[0, 2, 4, 6],
              color_list=['r', 'b', 'm', 'k'],
              label_list=['ch 0', 'ch 2', 'ch 4', 'ch 6'],
              filename='plot2.png')


def plot3(plt, matplotlib):
    """Like plot1, but intrinsic_width=0 (scattering-dominated) and spectral_index=0."""
    make_plot(plt, matplotlib,
              dict(freq_lo_MHz=1000.0, freq_hi_MHz=2000.0, nfreq=512,
                   dm=10.0, sm=4.0, intrinsic_width=0.0, fluence=30.0, spectral_index=0.0),
              ifreq_list=[0, 256, 511],
              color_list=['r', 'b', 'm'],
              label_list=['1 GHz', '1.5 GHz', '2.0 GHz'],
              filename='plot3.png')


def plot4(plt, matplotlib):
    """Two channelizations of the same dispersed + scattered pulse, as waterfalls (32 channels x
    time): channels even in freq (the freq^-2 dispersion sweep -> a CURVED track) vs even in
    freq^-2 (== even in dispersion delay -> a STRAIGHT track). Low-freq channels show the longer
    scattering tails (sm = 0.1 ms at 1 GHz)."""
    freq_lo, freq_hi = 400.0, 800.0
    time_sample_ms = 1.0
    dm, sm, width = 2.0, 0.1, 0.5e-3    # DM=2, SM=0.1 ms at 1 GHz, pulse width 0.5 ms
    nchan = 32

    # Frame the pulse near t=0 on the zero-based grid (shared by both panels).
    d_hi = simpulse.dispersion_delay(dm, freq_hi)
    d_lo = simpulse.dispersion_delay(dm, freq_lo)
    lead = 0.1*(d_lo - d_hi) + 4.0*width + simpulse.scattering_time(sm, freq_hi) + 2.0*time_sample_ms*1e-3
    uat = -d_hi + lead

    edges_top = np.linspace(freq_lo, freq_hi, nchan + 1)                    # even in freq
    edges_bot = np.linspace(freq_lo**-2, freq_hi**-2, nchan + 1) ** -0.5    # even in freq^-2 (ordered low->high freq)

    def make(edges):
        return simpulse.SinglePulse(internal_nt=1024, time_sample_ms=time_sample_ms, freq_edges_MHz=edges,
                                    dm=dm, sm=sm, intrinsic_width=width, fluence=1.0,
                                    spectral_index=0.0, undispersed_arrival_time_sec=uat)

    sp_top, sp_bot = make(edges_top), make(edges_bot)
    out_nt = max(sp_top.nt_min, sp_bot.nt_min)

    def waterfall(sp):
        a = np.zeros((sp.nfreq, out_nt), dtype=np.float32)
        sp.add_to_timestream(a)
        return a

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for (ax, sp, title) in [(axes[0], sp_top, 'top: 32 channels even in freq'),
                            (axes[1], sp_bot, r'bottom: 32 channels even in freq$^{-2}$')]:
        im = ax.imshow(waterfall(sp), origin='lower', aspect='auto', interpolation='nearest',
                       extent=[0, out_nt*time_sample_ms, 0, nchan], cmap='inferno')
        ax.set_ylabel('channel (low -> high freq)')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='intensity')
    axes[1].set_xlabel('time (ms)')
    fig.suptitle('DM=2, SM=0.1 ms@1GHz, width=0.5 ms, 400-800 MHz, dt=1 ms')
    fig.tight_layout()
    fig.savefig('plot4.png')
    plt.close(fig)
    print('wrote plot4.png')


def make_plots():
    """Generate plot1.png, plot2.png, plot3.png, plot4.png in the current directory."""
    import matplotlib
    matplotlib.use('Agg')   # 'Agg' enables silent output to file
    import matplotlib.pyplot as plt

    plot1(plt, matplotlib)
    plot2(plt, matplotlib)
    plot3(plt, matplotlib)
    plot4(plt, matplotlib)
