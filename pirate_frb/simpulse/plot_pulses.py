"""
plot_pulses: a throwaway script from visually debugging the pulse simulation. Ported from
simpulse/visual_check/plot-pulses.py. make_plots() writes plot1.png, plot2.png, plot3.png into
the current directory; the reference_plot*.png files in the original repo
(../extern/simpulse/visual_check/) show what the results should look like.

Run it via 'python -m pirate_frb test_simpulse'. matplotlib is imported lazily inside make_plots(),
so that 'import pirate_frb.simpulse' does not require matplotlib.
"""

import numpy as np

from .. import simpulse


# 'plt' and 'matplotlib' are passed in from make_plots() (which imports them lazily and selects the
# Agg backend), so that importing this module does not require matplotlib.

def make_plot(plt, matplotlib, sp, ifreq_list, color_list, label_list, filename):
    # check whether inputs are valid objects
    assert isinstance(sp, simpulse.SinglePulse)
    assert len(ifreq_list) == len(color_list) == len(label_list)

    # given sp, get the end points of the time axis
    (plot_t0, plot_t1) = sp.get_endpoints()

    # loop over various plot types, e.g., with(out) labels
    for (plot_nt, ls, lflag) in [(100, '-', True), (1000, ':', False)]:

        # make an empty array of (freq, time). add_to_timestream() requires float32.
        ts = np.zeros((sp.nfreq, plot_nt), dtype=np.float32)

        # add the pulse to the array, constrained by the end points
        sp.add_to_timestream(ts, plot_t0, plot_t1)

        # upsample by 2 for a histogram look
        x = np.zeros(2*plot_nt)
        y = np.zeros(2*plot_nt)

        # assign time values
        t = np.linspace(plot_t0, plot_t1, plot_nt+1)
        x[0::2] = t[:-1]
        x[1::2] = t[1:]

        # looping over a set of freq channels, assign freq values to the y-axis vector
        for (ifreq, color, label) in zip(ifreq_list, color_list, label_list):
            y[0::2] = ts[ifreq, :]
            y[1::2] = ts[ifreq, :]

            # choose a good font
            font = {'family': 'serif', 'size': 14}
            matplotlib.rc('font', **font)

            # plot with(out) label
            if lflag:
                plt.plot(x, y, color=color, ls=ls, label=label)
            else:
                plt.plot(x, y, color=color, ls=ls)

    plt.legend(loc='upper right').draw_frame(False)
    plt.savefig(filename)
    plt.clf()
    print('wrote', filename)


def plot1(plt, matplotlib):
    """
    Some Gaussians with a little bit of dispersion and scattering.

    Visual checks:
       - arrival times of pulses should be 141.5, 118.4, 110.4 ms
       - fluences should be { 13.3, 30, 53.3 }
       - low frequency pulse should show scattering
       - no dispersion broadening visible
    """
    sp = simpulse.SinglePulse(pulse_nt=1024, freq_edges_MHz=np.linspace(1000.0, 2000.0, 513),
                              dm=10.0, sm=4.0, intrinsic_width=0.005, fluence=30.0,
                              spectral_index=2.0, undispersed_arrival_time=0.100)

    make_plot(plt, matplotlib, sp,
              ifreq_list=[0, 256, 511],
              color_list=['r', 'b', 'm'],
              label_list=['1 GHz', '1.5 GHz', '2.0 GHz'],
              filename='plot1.png')


def plot2(plt, matplotlib):
    """Boxcars labeled by time intervals."""
    sp = simpulse.SinglePulse(pulse_nt=1024, freq_edges_MHz=np.linspace(1000.0, 2000.0, 8),
                              dm=10.0, sm=0.0, intrinsic_width=0.0, fluence=1.0,
                              spectral_index=0.0, undispersed_arrival_time=0.100)

    make_plot(plt, matplotlib, sp,
              ifreq_list=[0, 2, 4, 6],
              color_list=['r', 'b', 'm', 'k'],
              label_list=['131.8-141.5', '116.8-120.3', '114.1-116.8', '110.4-112.0'],
              filename='plot2.png')


def plot3(plt, matplotlib):
    """
    Same as plot1, but with intrinsic_width set to zero (so pulse width is dominated by scattering)
    and spectral_index set to zero.
    """
    sp = simpulse.SinglePulse(pulse_nt=1024, freq_edges_MHz=np.linspace(1000.0, 2000.0, 513),
                              dm=10.0, sm=4.0, intrinsic_width=0.0, fluence=30.0,
                              spectral_index=0.0, undispersed_arrival_time=0.100)

    make_plot(plt, matplotlib, sp,
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
