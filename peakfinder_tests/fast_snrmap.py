"""Fast analytic PIRATE-like S/N maps for peak-finder experiments.

This module deliberately starts *after* channelized-voltage simulation.  It builds
the deterministic DM--time response of one or more dispersed Gaussian pulses and
adds a small correlated-background model.  The intended use is a fast, controlled
separability experiment; it is not a replacement for end-to-end PIRATE validation.

The important non-circularity property is that no peak-finder footprint is used to
make the signal.  The butterfly follows only from the cold-plasma delay law, the
injected sub-band, the pulse width, and coordinates decoded from the real PIRATE
plan.  The production full-band bowtie therefore sees an independently generated
map rather than a response built from its own peak-search footprint.
"""

from dataclasses import asdict, dataclass
import math
import time

import numpy as np


# With frequencies in MHz, K_DM has units of seconds MHz^2 / (pc cm^-3).
K_DM = 4148.808
GAUSSIAN_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

FAST_MODEL_NOTE = (
    "analytic-dm-time-v1: Gaussian bursts are summed before a fine-time "
    "maximum; residual-DM broadening is the variance of a uniform delay interval; "
    "the profile bank is represented by the decoded profile whose nominal width is "
    "closest to the injected Gaussian FWHM; background is a reproducible correlated "
    "Gaussian field calibrated to PIRATE out_max order statistics. All argmax tokens "
    "belong to the injected plan sub-band. Validate against end-to-end PIRATE before "
    "using the model for scientific conclusions."
)
FAST_RNG_NOTE = (
    "Analytic backgrounds are exactly reproducible from --analytic-seed. Within "
    "each (sub-band, trial), every separation uses the same background realization "
    "to make separability comparisons paired rather than adding avoidable Monte Carlo "
    "scatter. Different sub-bands and trials use deterministic independent seeds."
)


@dataclass(frozen=True)
class FastBackgroundParameters:
    """Small stationary approximation to the background of ``out_max``.

    The location and scale are intentionally parameters rather than hidden magic
    constants.  Their defaults reproduce the rough median (~2.7), scatter (~0.7),
    and nearest-neighbour texture seen in the supplied PIRATE S/N-map example.
    They are not suitable for a false-alarm or near-threshold study without a
    dedicated calibration set.
    """

    location: float = 2.7
    scale: float = 0.7
    dm_neighbour_weight: float = 0.55
    time_neighbour_weight: float = 0.35

    def validate(self):
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("background parameters must be finite")
        if self.scale < 0:
            raise ValueError("background scale must be non-negative")
        if self.dm_neighbour_weight < 0 or self.time_neighbour_weight < 0:
            raise ValueError("background neighbour weights must be non-negative")


def _require_shape(name, array, shape):
    if tuple(array.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")


def make_correlated_background(xp, shape, *, seed, parameters=None):
    """Return one reproducible float32 background map using ``numpy`` or ``cupy``.

    A five-tap stencil gives correlations in both map directions without importing
    ``cupyx.scipy`` or launching an FFT.  The analytic normalization preserves the
    requested marginal scale away from the periodic boundaries introduced by
    ``roll``.  Periodicity is harmless here because peak finders exclude map edges.
    """

    parameters = parameters or FastBackgroundParameters()
    parameters.validate()
    if len(shape) != 2 or any(int(n) < 2 for n in shape):
        raise ValueError(f"background shape must be two-dimensional, got {shape}")

    rng = xp.random.default_rng(int(seed))
    white = rng.standard_normal(tuple(map(int, shape))).astype(xp.float32)
    a = float(parameters.dm_neighbour_weight)
    b = float(parameters.time_neighbour_weight)
    correlated = (
        white
        + a * (xp.roll(white, 1, axis=0) + xp.roll(white, -1, axis=0))
        + b * (xp.roll(white, 1, axis=1) + xp.roll(white, -1, axis=1))
    )
    normalization = math.sqrt(1.0 + 2.0 * a * a + 2.0 * b * b)
    correlated *= np.float32(parameters.scale / normalization)
    correlated += np.float32(parameters.location)
    return correlated.astype(xp.float32, copy=False)


def make_analytic_signal_template(
    xp,
    *,
    dm_axis,
    timestamp_by_fine_s,
    fine_tokens,
    nt,
    time_step_s,
    time_chunk_start_s,
    injected_dm,
    injected_toas_s,
    injected_snrs,
    injected_width_s,
    freq_lo_mhz,
    freq_hi_mhz,
    reference_freq_mhz,
    profile_width_s,
):
    """Construct a deterministic signal map and its valid argmax-token map.

    ``timestamp_by_fine_s[k, idm]`` is the chunk-relative physical timestamp
    decoded from ``fine_tokens[k]`` at coarse time index zero.  Adding
    ``itime*time_step_s`` therefore gives the physical coordinate represented by
    each fine-time state of a coarse output pixel.

    For a trial DM ``D`` and injected DM ``D0``, the residual delay distribution
    across the injected band has midpoint

        K_DM (D0-D) (u_mid-u_ref)

    and variance ``span**2/12``, where ``u = frequency**-2``.  Convolving that
    distribution with the injected Gaussian gives ``sigma_eff``.  Its peak S/N is
    attenuated by ``sqrt(sigma_injected/sigma_eff)``, as expected for a
    width-matched search at fixed fluence.

    Crucially, all pulses are summed at every fine-time state *before* the maximum
    over that state.  Two already-maximized S/N maps are never added together.
    """

    dm_axis = xp.asarray(dm_axis, dtype=xp.float32)
    timestamp_by_fine_s = xp.asarray(timestamp_by_fine_s, dtype=xp.float32)
    fine_tokens = xp.asarray(fine_tokens, dtype=xp.uint32)
    ndm = int(dm_axis.size)
    nt = int(nt)
    _require_shape("timestamp_by_fine_s", timestamp_by_fine_s,
                   (int(fine_tokens.size), ndm))
    if nt < 1 or fine_tokens.size < 1:
        raise ValueError("nt and the number of fine-time states must be positive")
    if not np.isfinite(time_step_s) or time_step_s <= 0:
        raise ValueError("time_step_s must be finite and positive")
    if not np.isfinite(injected_width_s) or injected_width_s <= 0:
        raise ValueError("injected_width_s must be finite and positive")
    if not np.isfinite(profile_width_s) or profile_width_s <= 0:
        raise ValueError("profile_width_s must be finite and positive")
    if not reference_freq_mhz <= freq_lo_mhz < freq_hi_mhz:
        raise ValueError("expected reference <= sub-band low edge < high edge")

    toas = tuple(float(value) for value in injected_toas_s)
    snrs = tuple(float(value) for value in injected_snrs)
    if not toas or len(toas) != len(snrs):
        raise ValueError("injected_toas_s and injected_snrs must have equal nonzero length")
    if not all(np.isfinite(value) for value in toas + snrs):
        raise ValueError("injected TOAs and S/N values must be finite")
    if any(value <= 0 for value in snrs):
        raise ValueError("injected S/N values must be positive")

    delta_dm = np.float32(float(injected_dm)) - dm_axis
    u_lo = float(freq_lo_mhz) ** -2
    u_hi = float(freq_hi_mhz) ** -2
    u_ref = float(reference_freq_mhz) ** -2
    u_mid = 0.5 * (u_lo + u_hi)

    centre_shift_s = delta_dm * np.float32(K_DM * (u_mid - u_ref))
    delay_span_s = xp.abs(delta_dm) * np.float32(K_DM * (u_lo - u_hi))
    sigma_eff_s = xp.sqrt(
        np.float32(injected_width_s * injected_width_s)
        + delay_span_s * delay_span_s / np.float32(12.0)
    )

    # The selected PIRATE profile broadens the time response but is normalized not
    # to change the requested S/N at the correct DM.  This is an envelope model of
    # the profile maximum, not a reimplementation of PeakFindingKernel.
    profile_sigma_s = np.float32(profile_width_s / math.sqrt(12.0))
    response_sigma_s = xp.sqrt(sigma_eff_s * sigma_eff_s
                               + profile_sigma_s * profile_sigma_s)
    attenuation = xp.sqrt(np.float32(injected_width_s) / sigma_eff_s)

    itime_offset_s = xp.arange(nt, dtype=xp.float32) * np.float32(time_step_s)
    best = xp.full((ndm, nt), -xp.inf, dtype=xp.float32)
    best_token = xp.full((ndm, nt), fine_tokens[0], dtype=xp.uint32)

    for ifine in range(int(fine_tokens.size)):
        coordinate_s = (
            timestamp_by_fine_s[ifine, :, None]
            + itime_offset_s[None, :]
            + np.float32(time_chunk_start_s)
        )
        combined = xp.zeros((ndm, nt), dtype=xp.float32)
        for toa_s, snr in zip(toas, snrs):
            centre_s = np.float32(toa_s) + centre_shift_s
            residual = (coordinate_s - centre_s[:, None]) / response_sigma_s[:, None]
            combined += (
                np.float32(snr) * attenuation[:, None]
                * xp.exp(np.float32(-0.5) * residual * residual)
            )
        better = combined > best
        best = xp.where(better, combined, best)
        best_token = xp.where(better, fine_tokens[ifine], best_token)

    return (best.astype(xp.float32, copy=False),
            best_token.astype(xp.uint32, copy=False))


from .producer_metadata import ARGMAX_ENCODING


class FastSnrMapSimulator:
    """Cache analytic signal templates and correlated backgrounds on one backend.

    ``xp`` must be ``cupy`` in the real benchmark.  Keeping it explicit also makes
    the mathematical core testable with NumPy on machines without CUDA.
    """

    def __init__(
        self,
        plan,
        itree,
        subband,
        *,
        xp,
        dcores,
        time_sample_s,
        nt_in,
        reference_freq_mhz,
        injected_dm,
        injected_snr,
        injected_width_s,
        base_seed=12345,
        background_parameters=None,
    ):
        from .peakfinders import producer_dcore_array

        self.plan = plan
        self.dcores = producer_dcore_array(plan, dcores)
        self.itree = int(itree)
        self.subband = subband
        self.xp = xp
        self.time_sample_s = float(time_sample_s)
        self.nt_in = int(nt_in)
        self.reference_freq_mhz = float(reference_freq_mhz)
        self.injected_dm = float(injected_dm)
        self.injected_snr = float(injected_snr)
        self.injected_width_s = float(injected_width_s)
        self.base_seed = int(base_seed)
        self.background_parameters = background_parameters or FastBackgroundParameters()
        self.background_parameters.validate()

        if not (0 <= self.itree < int(plan.ntrees)):
            raise ValueError(f"tree {self.itree} is outside plan with {plan.ntrees} trees")
        if self.time_sample_s <= 0 or self.nt_in <= 0:
            raise ValueError("time_sample_s and nt_in must be positive")
        if self.injected_snr <= 0 or self.injected_width_s <= 0:
            raise ValueError("injected S/N and width must be positive")
        if self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")

        self.tree = plan.trees[self.itree]
        self.ndm = int(self.tree.ndm_out)
        self.nt = int(self.tree.nt_out)
        self.time_step_s = self.time_sample_s * self.nt_in / self.nt
        self.chunk_duration_s = self.time_sample_s * self.nt_in
        self._signal_cache = {}
        self._argmax_cache = {}
        self._background_cache = {}
        self._precompute_wall_time_s = 0.0

        coordinate_model = self._make_coordinate_model()
        self.representative_multiplet = coordinate_model["multiplet"]
        self.extra_dm = coordinate_model["extra_dm"]
        self.profile = coordinate_model["profile"]
        self.profile_width_s = coordinate_model["profile_width_s"]
        self.fine_tokens_cpu = coordinate_model["fine_tokens"]
        self.dm_axis_cpu = coordinate_model["dm_axis"]
        self.timestamp_by_fine_cpu = coordinate_model["timestamp_by_fine_s"]
        self.dm_step = float(np.median(np.diff(self.dm_axis_cpu)))

        self.dm_axis = xp.asarray(self.dm_axis_cpu, dtype=xp.float32)
        self.timestamp_by_fine_s = xp.asarray(
            self.timestamp_by_fine_cpu, dtype=xp.float32)
        self.fine_tokens = xp.asarray(self.fine_tokens_cpu, dtype=xp.uint32)

    def _decode_physical(self, tokens, idms, itimes):
        tokens = np.ascontiguousarray(tokens, dtype=np.uint32)
        idms = np.ascontiguousarray(idms, dtype=np.int64)
        itimes = np.ascontiguousarray(itimes, dtype=np.int64)
        itrees = np.full(tokens.size, self.itree, dtype=np.int64)
        decoded = self.plan.decode_argmax_batch(
            tokens, itrees, idms, itimes, dcores=self.dcores)
        return decoded, self.plan.decode_argmax2_batch(itrees, *decoded)

    def _choose_multiplet(self):
        fs = self.tree.frequency_subbands
        members = np.flatnonzero(
            np.asarray(fs.m_to_n, dtype=np.int64) == int(self.subband.band_index)
        )
        if not members.size:
            raise RuntimeError(f"sub-band {self.subband.subband_id} has no multiplets")

        fraction = ((self.injected_dm - float(self.tree.dm_min))
                    / (float(self.tree.dm_max) - float(self.tree.dm_min)))
        nominal_idm = int(np.clip(math.floor(fraction * self.ndm), 0, self.ndm - 1))
        nearby = np.arange(max(0, nominal_idm - 2),
                           min(self.ndm, nominal_idm + 3), dtype=np.int64)
        extra_count, remainder = divmod(int(self.tree.dm_downsampling), 1 << int(fs.pf_rank))
        if remainder or not 1 <= extra_count <= 256:
            raise ValueError("tree has invalid PIRATE 1.5 extra-DM token geometry")
        trial_m, trial_mu, trial_idm = (
            values.ravel() for values in np.meshgrid(
                members, np.arange(extra_count, dtype=np.int64), nearby, indexing="ij")
        )
        tokens = ((trial_m.astype(np.uint32) << np.uint32(16))
                  | (trial_mu.astype(np.uint32) << np.uint32(24)))
        _, physical = self._decode_physical(
            tokens, trial_idm, np.zeros(tokens.size, dtype=np.int64))
        dms = np.asarray(physical[2], dtype=np.float64)
        best = int(np.argmin(np.abs(dms - self.injected_dm)))
        return int(trial_m[best]), int(trial_mu[best])

    def _choose_profile(self, multiplet, extra_dm):
        profiles = np.arange(int(self.tree.nprofiles), dtype=np.int64)
        tokens = ((np.uint32(extra_dm) << np.uint32(24))
                  | (np.uint32(multiplet) << np.uint32(16))
                  | (profiles.astype(np.uint32) << np.uint32(8)))
        idm = np.full(profiles.size, self.ndm // 2, dtype=np.int64)
        _, physical = self._decode_physical(
            tokens, idm, np.zeros(profiles.size, dtype=np.int64))
        widths_s = np.asarray(physical[4], dtype=np.float64) * self.time_sample_s
        target = GAUSSIAN_FWHM_PER_SIGMA * self.injected_width_s
        profile = int(np.argmin(np.abs(widths_s - target)))
        return profile, float(widths_s[profile])

    def _make_coordinate_model(self):
        multiplet, extra_dm = self._choose_multiplet()
        profile, profile_width_s = self._choose_profile(multiplet, extra_dm)

        dout = int(self.tree.nt_ds) // self.nt
        if dout < 1 or int(self.tree.nt_ds) % self.nt:
            raise RuntimeError("tree nt_ds/nt_out is not a positive integer")
        lpf = ((profile - 1) // 3) if profile else 0
        token_quantization = min(int(self.dcores[self.itree]), 1 << lpf)
        fine_time = np.arange(0, dout, token_quantization, dtype=np.uint32)
        fine_tokens = (
            (np.uint32(extra_dm) << np.uint32(24))
            | (np.uint32(multiplet) << np.uint32(16))
            | (np.uint32(profile) << np.uint32(8))
            | fine_time
        ).astype(np.uint32, copy=False)

        idms = np.arange(self.ndm, dtype=np.int64)
        timestamp_rows = []
        dm_axis = None
        expected_pair = (int(self.subband.fmin), int(self.subband.fmax))
        for token in fine_tokens:
            tokens = np.full(self.ndm, token, dtype=np.uint32)
            decoded, physical = self._decode_physical(
                tokens, idms, np.zeros(self.ndm, dtype=np.int64))
            fmin = np.asarray(decoded[0], dtype=np.int64)
            fmax = np.asarray(decoded[1], dtype=np.int64)
            if not (np.all(fmin == expected_pair[0]) and np.all(fmax == expected_pair[1])):
                raise RuntimeError("representative argmax token decodes to the wrong sub-band")
            current_dm = np.asarray(physical[2], dtype=np.float64)
            if dm_axis is None:
                dm_axis = current_dm
            elif not np.array_equal(current_dm, dm_axis):
                raise RuntimeError("decoded DM unexpectedly depends on fine time")
            timestamp_rows.append(
                np.asarray(physical[3], dtype=np.float64) * self.time_sample_s)

        dm_differences = np.diff(dm_axis)
        if not np.all(dm_differences > 0):
            raise RuntimeError("decoded DM coordinate is not strictly increasing")

        # One corner verifies the affine coarse-time stride used below.
        token = np.asarray([fine_tokens[0]], dtype=np.uint32)
        idm = np.asarray([self.ndm // 2], dtype=np.int64)
        _, physical0 = self._decode_physical(token, idm, np.asarray([0], dtype=np.int64))
        _, physical1 = self._decode_physical(token, idm, np.asarray([1], dtype=np.int64))
        observed_stride = float(physical1[3][0] - physical0[3][0]) * self.time_sample_s
        if not np.isclose(observed_stride, self.time_step_s, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError(
                f"decoded time stride {observed_stride} disagrees with {self.time_step_s}"
            )

        return {
            "multiplet": multiplet,
            "extra_dm": extra_dm,
            "profile": profile,
            "profile_width_s": profile_width_s,
            "fine_tokens": fine_tokens,
            "dm_axis": dm_axis,
            "timestamp_by_fine_s": np.stack(timestamp_rows),
        }

    def _normalize_injected_snrs(self, toas, injected_snrs):
        """Return one checked amplitude per TOA, using the constructor default."""
        toas = tuple(float(value) for value in toas)
        if not toas or not all(np.isfinite(value) for value in toas):
            raise ValueError("analytic benchmark TOAs must be nonempty and finite")
        if injected_snrs is None:
            snrs = (self.injected_snr,) * len(toas)
        else:
            try:
                snrs = tuple(float(value) for value in injected_snrs)
            except TypeError as exc:
                raise ValueError(
                    "injected_snrs must provide one value per TOA"
                ) from exc
            if len(snrs) != len(toas):
                raise ValueError(
                    f"injected_snrs has length {len(snrs)}, expected {len(toas)}"
                )
        if not all(np.isfinite(value) and value > 0 for value in snrs):
            raise ValueError("every injected S/N must be finite and positive")
        return toas, snrs

    @staticmethod
    def _template_key(toas, injected_snrs, time_chunk_index):
        return (int(time_chunk_index), tuple(toas), tuple(injected_snrs))

    def common_chunk_index(self, toas):
        """Return the chunk containing all reference-frequency TOAs."""
        chunks = {
            int(math.floor(float(toa) / self.chunk_duration_s)) for toa in toas
        }
        if len(chunks) != 1:
            raise ValueError(f"analytic benchmark bursts span chunks {sorted(chunks)}")
        chunk = chunks.pop()
        if chunk < 0:
            raise ValueError("analytic benchmark TOAs must be non-negative")
        return chunk

    def signal_template(self, toas, *, injected_snrs=None,
                        time_chunk_index=None):
        toas, injected_snrs = self._normalize_injected_snrs(
            toas, injected_snrs)
        if time_chunk_index is None:
            time_chunk_index = self.common_chunk_index(toas)
        key = self._template_key(toas, injected_snrs, time_chunk_index)
        if key not in self._signal_cache:
            signal, argmax = make_analytic_signal_template(
                self.xp,
                dm_axis=self.dm_axis,
                timestamp_by_fine_s=self.timestamp_by_fine_s,
                fine_tokens=self.fine_tokens,
                nt=self.nt,
                time_step_s=self.time_step_s,
                time_chunk_start_s=time_chunk_index * self.chunk_duration_s,
                injected_dm=self.injected_dm,
                injected_toas_s=toas,
                injected_snrs=injected_snrs,
                injected_width_s=self.injected_width_s,
                freq_lo_mhz=self.subband.freq_lo_MHz,
                freq_hi_mhz=self.subband.freq_hi_MHz,
                reference_freq_mhz=self.reference_freq_mhz,
                profile_width_s=self.profile_width_s,
            )
            self._signal_cache[key] = signal
            self._argmax_cache[key] = argmax
        return self._signal_cache[key], self._argmax_cache[key]

    def _background_seed(self, trial):
        # Separate sub-bands deterministically while keeping every separation within
        # one (sub-band, trial) paired on exactly the same background realization.
        return self.base_seed + 1_000_003 * int(self.subband.band_index) + int(trial)

    def background(self, trial):
        trial = int(trial)
        if trial not in self._background_cache:
            self._background_cache[trial] = make_correlated_background(
                self.xp, (self.ndm, self.nt), seed=self._background_seed(trial),
                parameters=self.background_parameters,
            )
        return self._background_cache[trial]

    def generate_map(self, toas, *, trial, injected_snrs=None,
                     time_chunk_index=None):
        """Return GPU-resident ``(out_max, out_argmax, chunk_index)`` analogues."""
        toas, injected_snrs = self._normalize_injected_snrs(
            toas, injected_snrs)
        if time_chunk_index is None:
            time_chunk_index = self.common_chunk_index(toas)
        signal, argmax = self.signal_template(
            toas, injected_snrs=injected_snrs,
            time_chunk_index=time_chunk_index)

        snr_map = signal + self.background(trial) - self.background_parameters.location

        return snr_map.astype(self.xp.float32, copy=False), argmax, int(time_chunk_index)

    def synchronize(self):
        cuda = getattr(self.xp, "cuda", None)
        if cuda is not None:
            cuda.get_current_stream().synchronize()

    def precompute(self, toa_pairs, trials):
        """Populate all deterministic and stochastic caches before timed units."""
        start = time.perf_counter()
        for toas in toa_pairs:
            self.signal_template(toas)
        for trial in trials:
            self.background(trial)
        self.synchronize()
        self._precompute_wall_time_s += time.perf_counter() - start
        return self._precompute_wall_time_s

    def pulse_support(self, toa_s, *, sigma_radius=6.0):
        """Approximate the raw-stream support used only for output diagnostics."""
        delays = [
            K_DM * self.injected_dm
            * (float(freq) ** -2 - self.reference_freq_mhz ** -2)
            for freq in (self.subband.freq_lo_MHz, self.subband.freq_hi_MHz)
        ]
        start_s = float(toa_s) + min(delays) - sigma_radius * self.injected_width_s
        end_s = float(toa_s) + max(delays) + sigma_radius * self.injected_width_s
        return [
            int(math.floor(start_s / self.time_sample_s)),
            int(math.ceil(end_s / self.time_sample_s)) + 1,
        ]

    def preflight(self, toas, *, nchunks):
        supports = [self.pulse_support(toa) for toa in toas]
        stream_samples = int(nchunks) * self.nt_in
        bad = [support for support in supports
               if support[0] < 0 or support[1] > stream_samples]
        if bad:
            raise ValueError(
                f"analytic raw supports {bad} exceed simulated span [0,{stream_samples})"
            )
        return {
            "requested_toas_s": [float(value) for value in toas],
            "toa_native_samples": [float(value) / self.time_sample_s for value in toas],
            "stored_toa_difference_s": float(toas[1]) - float(toas[0]),
            "pulse_supports": supports,
            "active_channels_verified_for_both_bursts": True,
        }

    def diagnostics(self):
        signal_bytes = sum(int(value.nbytes) for value in self._signal_cache.values())
        argmax_bytes = sum(int(value.nbytes) for value in self._argmax_cache.values())
        background_bytes = sum(int(value.nbytes) for value in self._background_cache.values())
        return {
            "model": FAST_MODEL_NOTE,
            "tree": self.itree,
            "subband_id": self.subband.subband_id,
            "shape": [self.ndm, self.nt],
            "representative_multiplet": self.representative_multiplet,
            "representative_extra_dm": self.extra_dm,
            "dcores": self.dcores.tolist(),
            "argmax_encoding": ARGMAX_ENCODING,
            "representative_profile": self.profile,
            "representative_profile_width_ms": 1.0e3 * self.profile_width_s,
            "fine_time_states": int(self.fine_tokens_cpu.size),
            "dm_step": self.dm_step,
            "time_step_ms": 1.0e3 * self.time_step_s,
            "base_seed": self.base_seed,
            "background": asdict(self.background_parameters),
            "cached_signal_templates": len(self._signal_cache),
            "cached_backgrounds": len(self._background_cache),
            "cache_bytes": signal_bytes + argmax_bytes + background_bytes,
            "precompute_wall_time_s": self._precompute_wall_time_s,
        }
