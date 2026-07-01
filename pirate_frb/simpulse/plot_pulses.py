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
    across the pulse. The framing (extent/undispersed_arrival_time) is independent of nsamp, so
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
        pulse_nt = 1024,
        time_sample_ms = 1.0e3 * extent / nsamp,
        freq_edges_MHz = edges,
        dm = dm, sm = sm, intrinsic_width = intrinsic_width,
        fluence = fluence, spectral_index = spectral_index,
        undispersed_arrival_time = -d_hi + lead)


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


def make_plots():
    """Generate plot1.png, plot2.png, plot3.png in the current directory."""
    import matplotlib
    matplotlib.use('Agg')   # 'Agg' enables silent output to file
    import matplotlib.pyplot as plt

    plot1(plt, matplotlib)
    plot2(plt, matplotlib)
    plot3(plt, matplotlib)
