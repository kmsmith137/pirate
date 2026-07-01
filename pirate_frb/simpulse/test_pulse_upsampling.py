"""
test_pulse_upsampling: checks that the pulse simulation is (approximately) invariant under
upsampling in frequency and/or time. Ported from simpulse/test_pulse_upsampling.py.

The test is APPROXIMATE and is not expected to pass to machine precision. It PRINTS the
correlation coefficient and residual for the operator to interpret; it does not raise.
Run it via 'python -m pirate_frb test_simpulse'.
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
        self.undispersed_arrival_time = np.random.uniform(0.0, 1000.0 * self.tsamp)

        t0 = simpulse.dispersion_delay(self.dm, self.freq_hi_MHz)
        t1 = simpulse.dispersion_delay(self.dm, self.freq_lo_MHz + (self.freq_hi_MHz - self.freq_lo_MHz) / self.nfreq)
        t2 = simpulse.dispersion_delay(self.dm, self.freq_lo_MHz)

        dt_s = 2.0 * self.tsamp
        dt_d = 2.0 * (t2 - t1)
        dt_i = 5.0 * self.intrinsic_width
        dt_sc = 10.0 * simpulse.scattering_time(self.sm, self.freq_lo_MHz)

        self.pulse_t0 = self.undispersed_arrival_time + t0 - dt_s - dt_d - dt_i
        self.pulse_t1 = self.undispersed_arrival_time + t2 + dt_s + dt_d + dt_i + dt_sc
        self.conservative_pulse_width = dt_s + dt_d + dt_i + dt_sc

        self.out_t0 = self.pulse_t0 + np.random.uniform(0.0, 100.0 * self.tsamp)
        self.out_nt = int((self.pulse_t1 - self.out_t0) / self.tsamp) + np.random.randint(1, 100)
        self.out_t1 = self.out_t0 + self.out_nt * self.tsamp

    def __repr__(self):
        keys = ['nfreq', 'freq_lo_MHz', 'freq_hi_MHz', 'tsamp', 'diagonal_dm',
                'sm0', 'dm', 'sm', 'intrinsic_width', 'fluence', 'spectral_index',
                'undispersed_arrival_time', 'pulse_t0', 'pulse_t1',
                'conservative_pulse_width', 'out_t0', 'out_t1', 'out_nt']

        ret = 'upsampling_test_instance('
        first = True

        for k in keys:
            if not first:
                ret += ','
            ret += '\n    %s = %s' % (k, getattr(self, k))
            first = False

        ret += '\n)'
        return ret

    def _make_single_pulse_object(self, nupfreq=1):
        return simpulse.SinglePulse(
            pulse_nt = 1024,
            nfreq = self.nfreq * nupfreq,
            freq_lo_MHz = self.freq_lo_MHz,
            freq_hi_MHz = self.freq_hi_MHz,
            dm = self.dm,
            sm = self.sm,
            intrinsic_width = self.intrinsic_width,
            fluence = self.fluence,
            spectral_index = self.spectral_index,
            undispersed_arrival_time = self.undispersed_arrival_time
        )

    def run_test(self, nupfreq, nupsample):
        s0 = self._make_single_pulse_object()
        s1 = self._make_single_pulse_object(nupfreq) if (nupfreq != 1) else s0

        # add_to_timestream() requires float32 output arrays.
        a0 = np.zeros((self.nfreq, self.out_nt), dtype=np.float32)
        a1 = np.zeros((self.nfreq * nupfreq, self.out_nt * nupsample), dtype=np.float32)

        s0.add_to_timestream(a0, self.out_t0, self.out_t1)
        s1.add_to_timestream(a1, self.out_t0, self.out_t1)

        # Upcast to float64 for the (approximate) correlation diagnostics below. The output arrays
        # are float32; that quantization is the thing being tested, so we keep it in a0/a1.
        a0 = a0.astype(np.float64)
        a1 = downsample_2d(a1.astype(np.float64), self.nfreq, self.out_nt) / (nupfreq * nupsample)

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
