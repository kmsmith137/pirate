"""
test_pulse_upsampling: checks that the pulse simulation is (approximately) invariant under
upsampling in frequency and/or time. Ported from simpulse/test_pulse_upsampling.py.

Time sampling is now a construction parameter (time_sample_ms), and the grid is zero-based, so a
time-upsampled pulse is one built with a smaller time_sample_ms (= coarse / nupsample). The pulse
is framed near t=0 (via undispersed_arrival_time) so the zero-based dense arrays stay compact.

The test is APPROXIMATE and is not expected to pass to machine precision. It PRINTS the correlation
coefficient and residual for the operator to interpret; it does not raise. Run it via
'python -m pirate_frb test_simpulse'.
"""

import numpy as np

from .. import simpulse


def log_uniform(xmin, xmax, size=None):
    assert 0 < xmin < xmax
    return np.exp(np.random.uniform(np.log(xmin), np.log(xmax), size=size))


def downsample_2d(arr, new_nfreq, new_ntime):
    assert arr.ndim == 2
    assert new_nfreq > 0
    assert new_ntime > 0

    (nfreq, ntime) = arr.shape
    assert nfreq % new_nfreq == 0
    assert ntime % new_ntime == 0
    arr = np.reshape(arr, (new_nfreq, nfreq//new_nfreq, new_ntime, ntime//new_ntime))
    arr = np.sum(arr, axis=3)
    arr = np.sum(arr, axis=1)
    return arr


####################################################################################################


class upsampling_test_instance:
    def __init__(self):
        self.nfreq = np.random.randint(1000, 2000)
        self.freq_lo_MHz = np.random.uniform(200.0, 1000.0)
        self.freq_hi_MHz = self.freq_lo_MHz * np.random.uniform(1.1, 3.0)
        self.tsamp = np.random.uniform(0.001, 0.01)    # sample length in seconds (not milliseconds)

        # The "diagonal DM" is defined so that the total time delay is (nfreq * tsamp).
        t0 = simpulse.dispersion_delay(1.0, self.freq_hi_MHz)
        t1 = simpulse.dispersion_delay(1.0, self.freq_lo_MHz)
        self.diagonal_dm = self.nfreq * self.tsamp / (t1 - t0)

        # We define "sm0" to be the scattering measure such that the scattering time at the central
        # frequency is one time sample.
        fmid = (self.freq_lo_MHz + self.freq_hi_MHz) / 2.0
        self.sm0 = self.tsamp / simpulse.scattering_time(1.0, fmid)

        self.dm = np.random.uniform(0.5, 2.0) * self.diagonal_dm
        self.sm = log_uniform(1.0e-2, 3.0) * self.sm0
        self.intrinsic_width = log_uniform(0.01, 10.0) * self.tsamp
        self.fluence = np.random.uniform(1.0, 2.0)
        self.spectral_index = np.random.uniform(-1.0, 1.0)

        # Frame the pulse near t=0 on the zero-based grid: choose undispersed_arrival_time so the
        # earliest (highest-frequency) channel's pulse starts just after t=0. (Uses the same
        # conservative leading-margin estimate as the original code's window placement.)
        d0 = simpulse.dispersion_delay(self.dm, self.freq_hi_MHz)
        d1 = simpulse.dispersion_delay(self.dm, self.freq_lo_MHz + (self.freq_hi_MHz - self.freq_lo_MHz) / self.nfreq)
        d2 = simpulse.dispersion_delay(self.dm, self.freq_lo_MHz)
        lead = 2.0*self.tsamp + 2.0*(d2 - d1) + 5.0*self.intrinsic_width   # dt_s + dt_d + dt_i
        self.undispersed_arrival_time = -d0 + lead

        # Non-uniform (but not-too-irregular) coarse channel edges spanning [freq_lo, freq_hi], to
        # exercise the unequal-width code path. Channel widths stay within [0.2, 1.8] x the uniform
        # width -- deliberately not very irregular (very narrow channels tend to expose corner cases).
        N = self.nfreq
        df = np.random.uniform(0.0, 0.8 * (self.freq_hi_MHz - self.freq_lo_MHz) / N, size=N)
        edges = np.concatenate([[0.0], np.cumsum(df)])   # length N+1, edges[0]=0, edges[-1]=sum(df)
        self.coarse_edges_MHz = edges + np.linspace(self.freq_lo_MHz, self.freq_hi_MHz - edges[-1], N + 1)

    def __repr__(self):
        keys = ['nfreq', 'freq_lo_MHz', 'freq_hi_MHz', 'tsamp', 'diagonal_dm',
                'sm0', 'dm', 'sm', 'intrinsic_width', 'fluence', 'spectral_index',
                'undispersed_arrival_time']

        ret = 'upsampling_test_instance('
        first = True

        for k in keys:
            if not first:
                ret += ','
            ret += '\n    %s = %s' % (k, getattr(self, k))
            first = False

        ret += '\n)'
        return ret

    @staticmethod
    def _upsample_edges(edges, nupfreq):
        """Subdivide each coarse channel [edges[i], edges[i+1]] into 'nupfreq' EQUAL sub-channels.
        Returns the fine edge array of length nupfreq*(len(edges)-1) + 1."""
        if nupfreq == 1:
            return edges
        n = len(edges) - 1
        parts = [np.linspace(edges[i], edges[i+1], nupfreq + 1)[:-1] for i in range(n)]
        parts.append(edges[-1:])
        return np.concatenate(parts)

    def _make_single_pulse_object(self, freq_edges_MHz, tsamp_sec):
        return simpulse.SinglePulse(
            pulse_nt = 1024,
            time_sample_ms = 1.0e3 * tsamp_sec,
            freq_edges_MHz = freq_edges_MHz,
            dm = self.dm,
            sm = self.sm,
            intrinsic_width = self.intrinsic_width,
            fluence = self.fluence,
            spectral_index = self.spectral_index,
            undispersed_arrival_time = self.undispersed_arrival_time
        )

    def run_test(self, nupfreq, nupsample):
        # Coarse (non-uniform) channelization at time_sample = tsamp; the fine pulse subdivides each
        # coarse channel into nupfreq equal sub-channels AND samples nupsample times finer in time.
        coarse_edges = self.coarse_edges_MHz
        fine_edges = self._upsample_edges(coarse_edges, nupfreq)

        s0 = self._make_single_pulse_object(coarse_edges, self.tsamp)
        s1 = self._make_single_pulse_object(fine_edges, self.tsamp / nupsample)

        # Size the coarse grid to hold both pulses (they share undispersed_arrival_time, so
        # s1.nt_min ~ nupsample * s0.nt_min up to rounding).
        out_nt = max(int(s0.nt_min), -(-int(s1.nt_min) // nupsample))

        # add_to_timestream() requires float32 output arrays.
        a0 = np.zeros((self.nfreq, out_nt), dtype=np.float32)
        a1 = np.zeros((self.nfreq * nupfreq, out_nt * nupsample), dtype=np.float32)

        s0.add_to_timestream(a0)
        s1.add_to_timestream(a1)

        # Upcast to float64 for the (approximate) correlation diagnostics below. The output arrays
        # are float32; that quantization is the thing being tested, so we keep it in a0/a1.
        a0 = a0.astype(np.float64)
        a1 = downsample_2d(a1.astype(np.float64), self.nfreq, out_nt) / (nupfreq * nupsample)

        t = np.sum(a0*a0) * np.sum(a1*a1)
        d = np.sum((a0-a1)**2)**0.5 / t**0.25
        r = np.sum(a0*a1) / t**0.5

        print(self)
        print()
        print('(nupfreq, nupsample) = (%d, %d)' % (nupfreq, nupsample))
        print('Correlation coefficient:', r, '  (expect ~1; original float64 threshold was |r-1| < 1e-5)')
        print('Residual difference:', d, '  (expect ~0; original float64 threshold was |d| < 1e-3)')
        print()


def run_tests(niter=100):
    combos = [(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

    for it in range(niter):
        print('Iteration %d/%d' % (it, niter))
        print()

        nupfreq, nupsample = combos[np.random.randint(0, len(combos))]
        t = upsampling_test_instance()
        t.run_test(nupfreq, nupsample)
