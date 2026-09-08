"""Full-band peakfinding and seamless offline-stream regressions."""

import numpy as np

from ..Peakfinders import (
    EdgeFlag,
    OfflinePeakExtractor,
    PeakFinderGeometry,
    build_full_band_bowtie,
    concatenate_raw_candidates,
    extract_candidates,
    make_startup_valid_mask,
)
from .test_gpu_argmax_decoder import _make_plan


def _test_geometry(cp, *, ndm=5, ntime=7, footprint=None, steady=None):
    """Make a small scientifically typed geometry without a C++ plan."""

    if footprint is None:
        footprint = np.ones((3, 3), dtype=np.bool_)
    footprint = cp.asarray(footprint, dtype=cp.bool_)
    if steady is None:
        steady = np.zeros(ndm, dtype=np.int64)
    return PeakFinderGeometry(
        tree=0,
        ndm=ndm,
        ntime=ntime,
        dm_step=1.0,
        time_step_s=1.0,
        freq_low_mhz=300.0,
        freq_high_mhz=1500.0,
        reference_freq_mhz=300.0,
        waist_bins=0,
        full_band_bowtie=footprint,
        dm_radius=int(footprint.shape[0] // 2),
        time_radius=int(footprint.shape[1] // 2),
        requested_time_radius=int(footprint.shape[1] // 2),
        token_multiplets=1,
        token_profiles=1,
        token_dout=1,
        profile_dt=cp.asarray([1], dtype=cp.int64),
        steady_state_it0=cp.asarray(steady, dtype=cp.int64),
    )


def _legacy_v1_2_full_band(cp, geometry, dm_reach, waist_bins):
    """Evaluate the archived v1.2 full-band footprint formula verbatim."""

    dm_offsets = (
        cp.arange(-dm_reach, dm_reach + 1, dtype=cp.float64)
        * geometry.dm_step
    )
    slope_lo = 4148.808 * (
        geometry.reference_freq_mhz**-2 - geometry.freq_low_mhz**-2
    )
    slope_hi = 4148.808 * (
        geometry.reference_freq_mhz**-2 - geometry.freq_high_mhz**-2
    )
    edge_a = dm_offsets * slope_lo / geometry.time_step_s
    edge_b = dm_offsets * slope_hi / geometry.time_step_s
    lower = cp.minimum(edge_a, edge_b)
    upper = cp.maximum(edge_a, edge_b)
    upper[dm_offsets > 0] += waist_bins
    lower[dm_offsets < 0] -= waist_bins
    lower[dm_offsets == 0] = -waist_bins
    upper[dm_offsets == 0] = waist_bins
    reach = int(cp.ceil(cp.maximum(-lower.min(), upper.max())).item())
    offsets = cp.arange(-reach, reach + 1, dtype=cp.int32)
    return (
        (offsets[None, :] >= lower[:, None])
        & (offsets[None, :] <= upper[:, None])
    )


def _naive_peak_coordinates(snr, token, footprint, threshold):
    """CPU oracle for the archived clipped-footprint maximum semantics."""

    snr = np.asarray(snr)
    token = np.asarray(token)
    footprint = np.asarray(footprint)
    rd, rt = np.asarray(footprint.shape) // 2
    answer = []
    for ibeam in range(snr.shape[0]):
        for idm in range(snr.shape[1]):
            for itime in range(snr.shape[2]):
                value = snr[ibeam, idm, itime]
                if not np.isfinite(value) or value < threshold:
                    continue
                if token[ibeam, idm, itime] == np.uint32(0xffffffff):
                    continue
                competitors = []
                for jd, jt in np.argwhere(footprint):
                    kd = idm + int(jd) - rd
                    kt = itime + int(jt) - rt
                    if not (0 <= kd < snr.shape[1] and 0 <= kt < snr.shape[2]):
                        continue
                    other = snr[ibeam, kd, kt]
                    if (np.isfinite(other)
                            and token[ibeam, kd, kt] != np.uint32(0xffffffff)):
                        competitors.append(other)
                if value == max(competitors):
                    answer.append((ibeam, idm, itime))
    return answer


def _coordinates(cp, candidates, beam_ids):
    """Return host coordinates only at the test reporting boundary."""

    beam_to_index = {beam: i for i, beam in enumerate(beam_ids)}
    return [
        (beam_to_index[int(beam)], int(dm), int(time))
        for beam, dm, time in zip(
            cp.asnumpy(candidates.beam_id),
            cp.asnumpy(candidates.idm),
            cp.asnumpy(candidates.itime),
        )
    ]


def test_direct_full_band_matches_archived_for_every_real_tree(
        cuda_device_id=0):
    """Every tree matches the full-band formula inside the CHIME horizon."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        plan = _make_plan("chord_sb2_et.yml")
        saw_crop = False
        saw_unchanged = False
        for itree in range(int(plan.ntrees)):
            geometry = PeakFinderGeometry.from_plan(
                plan, itree, dm_reach=4, waist_bins=1
            )
            full_expected = _legacy_v1_2_full_band(cp, geometry, 4, 1)
            expected = full_expected
            while expected.shape[1] > geometry.ntime:
                expected = expected[:, 1:-1]
            assert np.array_equal(
                cp.asnumpy(geometry.full_band_bowtie),
                cp.asnumpy(expected),
            )
            assert geometry.full_band_bowtie.ndim == 2
            assert all(size % 2 for size in geometry.full_band_bowtie.shape)
            assert geometry.full_band_bowtie.shape[1] <= geometry.ntime
            assert 2 * geometry.time_radius < geometry.ntime
            assert geometry.requested_time_radius == (
                full_expected.shape[1] // 2
            )
            assert geometry.time_radius <= geometry.requested_time_radius
            saw_crop |= geometry.time_radius < geometry.requested_time_radius
            saw_unchanged |= (
                geometry.time_radius == geometry.requested_time_radius
            )
            assert bool(geometry.full_band_bowtie[
                geometry.dm_radius, geometry.time_radius
            ].item())

            cpu_direct = build_full_band_bowtie(
                dm_reach=4,
                dm_step=geometry.dm_step,
                time_step_s=geometry.time_step_s,
                freq_low_mhz=geometry.freq_low_mhz,
                freq_high_mhz=geometry.freq_high_mhz,
                reference_freq_mhz=geometry.reference_freq_mhz,
                waist_bins=1,
            )
            assert np.array_equal(cp.asnumpy(full_expected), cpu_direct)
        assert saw_crop and saw_unchanged


def test_cupyx_filter_matches_archived_semantics(cuda_device_id=0):
    """Interior, clipped edges, beams, invalid values, and views are exact."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        beam_ids = (11, 22)
        geometry = _test_geometry(cp, ndm=4, ntime=6)
        host_snr = np.zeros((2, 4, 12), dtype=np.float32)
        host_token = np.zeros((2, 4, 12), dtype=np.uint32)
        # Use every second column so the public path sees non-contiguous views.
        snr = cp.asarray(host_snr)[:, :, ::2]
        token = cp.asarray(host_token)[:, :, ::2]
        snr[0, 0, 0] = 10.0
        snr[0, 2, 2] = 8.0
        snr[0, 2, 3] = cp.nan
        snr[0, 3, 5] = cp.inf
        snr[1, 2, 2] = 9.0
        snr[1, 3, 5] = 7.0
        token[0, 1, 1] = cp.uint32(0xffffffff)
        snr[0, 1, 1] = 100.0

        expected = _naive_peak_coordinates(
            cp.asnumpy(snr),
            cp.asnumpy(token),
            cp.asnumpy(geometry.full_band_bowtie),
            5.0,
        )
        actual = extract_candidates(
            snr,
            token,
            geometry,
            threshold=5.0,
            beam_ids=beam_ids,
            source_chunk_index=4,
            assume_steady_state=True,
        )
        assert sorted(_coordinates(cp, actual, beam_ids)) == sorted(expected)
        assert np.all(np.isfinite(cp.asnumpy(actual.snr)))
        assert set(cp.asnumpy(actual.beam_id).tolist()) == {11, 22}
        assert np.any(
            cp.asnumpy(actual.edge_flags)
            & int(EdgeFlag.ACQUISITION_LEFT | EdgeFlag.DM_LOW)
        )


def test_candidate_counts_plateaus_and_order(cuda_device_id=0):
    """Zero, one, many, and equal plateaus use inclusive deterministic rules."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _test_geometry(cp, ndm=3, ntime=7)
        token = cp.zeros((1, 3, 7), dtype=cp.uint32)
        snr = cp.zeros((1, 3, 7), dtype=cp.float16)

        empty = extract_candidates(
            snr, token, geometry, threshold=5.0, beam_ids=(3,),
            source_chunk_index=0, assume_steady_state=True,
        )
        assert len(empty) == 0

        snr[0, 1, 3] = 5.0
        one = extract_candidates(
            snr, token, geometry, threshold=5.0, beam_ids=(3,),
            source_chunk_index=0, assume_steady_state=True,
        )
        assert len(one) == 1 and one.snr.dtype == cp.float32

        snr[0, 1, 4] = 5.0
        snr[0, 1, 0] = 7.0
        many_a = extract_candidates(
            snr, token, geometry, threshold=5.0, beam_ids=(3,),
            source_chunk_index=0, assume_steady_state=True,
        )
        many_b = extract_candidates(
            snr, token, geometry, threshold=5.0, beam_ids=(3,),
            source_chunk_index=0, assume_steady_state=True,
        )
        assert cp.asnumpy(many_a.itime).tolist() == [0, 3, 4]
        assert np.array_equal(
            cp.asnumpy(many_a.itime), cp.asnumpy(many_b.itime)
        )
        assert np.array_equal(
            cp.asnumpy(many_a.snr), cp.asnumpy(many_b.snr)
        )


def test_streaming_matches_concatenated_and_flushes(cuda_device_id=0):
    """A radius larger than half a chunk still emits every centre once."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        footprint = np.ones((1, 7), dtype=np.bool_)
        geometry = _test_geometry(
            cp, ndm=1, ntime=4, footprint=footprint
        )
        chunks = [
            cp.full((1, 1, 4), 10.0, dtype=cp.float32)
            for _ in range(3)
        ]
        tokens = [
            cp.zeros((1, 1, 4), dtype=cp.uint32)
            for _ in range(3)
        ]
        whole = extract_candidates(
            cp.concatenate(chunks, axis=2),
            cp.concatenate(tokens, axis=2),
            geometry,
            threshold=10.0,
            beam_ids=(9,),
            source_chunk_index=0,
            assume_steady_state=True,
        )

        extractor = OfflinePeakExtractor(
            geometry, threshold=10.0, beam_ids=(9,),
            assume_steady_state=True,
        )
        parts = []
        emitted_sizes = []
        for source, (snr, argmax) in enumerate(zip(chunks, tokens)):
            part = extractor.process_chunk(snr, argmax, source)
            parts.append(part)
            emitted_sizes.append(len(part))
        delayed = extractor.flush()
        parts.append(delayed)
        streamed = concatenate_raw_candidates(parts)

        expected_chunk = cp.asnumpy(whole.itime) // geometry.ntime
        expected_itime = cp.asnumpy(whole.itime) % geometry.ntime
        assert emitted_sizes == [1, 4, 4]
        assert len(delayed) == 3
        assert len(streamed) == len(whole) == 12
        assert np.array_equal(
            cp.asnumpy(streamed.source_chunk_index), expected_chunk
        )
        assert np.array_equal(cp.asnumpy(streamed.itime), expected_itime)
        assert np.array_equal(
            cp.asnumpy(streamed.edge_flags), cp.asnumpy(whole.edge_flags)
        )
        seam = (
            (cp.asnumpy(streamed.source_chunk_index) < 2)
            & (cp.asnumpy(streamed.itime) == 3)
        )
        acquisition_bits = int(
            EdgeFlag.ACQUISITION_LEFT | EdgeFlag.ACQUISITION_RIGHT
        )
        assert not np.any(cp.asnumpy(streamed.edge_flags)[seam] & acquisition_bits)


def test_startup_validity_and_missing_provenance(cuda_device_id=0):
    """The plan mask labels candidates; missing provenance stays explicit."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        plan = _make_plan()
        geometry = PeakFinderGeometry.from_plan(
            plan, 0, dm_reach=1, waist_bins=0
        )
        source_chunk = 7
        producer_start = 5
        actual = make_startup_valid_mask(
            geometry, source_chunk, producer_start
        )
        elapsed = (source_chunk - producer_start) * geometry.ntime
        expected = (
            np.arange(geometry.ntime, dtype=np.int64)[None, :] + elapsed
            >= np.asarray(plan.compute_steady_state_it0(0))[:, None]
        )
        assert np.array_equal(cp.asnumpy(actual), expected)

        small = _test_geometry(cp, ndm=1, ntime=3)
        snr = cp.zeros((1, 1, 3), dtype=cp.float32)
        token = cp.zeros((1, 1, 3), dtype=cp.uint32)
        try:
            extract_candidates(
                snr, token, small, threshold=0.0, beam_ids=(1,),
                source_chunk_index=0,
            )
        except ValueError as exc:
            assert "startup provenance is missing" in str(exc)
        else:
            raise AssertionError("missing startup provenance was accepted")

        assumed = extract_candidates(
            snr, token, small, threshold=0.0, beam_ids=(1,),
            source_chunk_index=0, assume_steady_state=True,
        )
        assert not np.any(
            cp.asnumpy(assumed.edge_flags)
            & int(EdgeFlag.STARTUP_INCOMPLETE)
        )


def test_startup_incomplete_is_a_label_not_a_search_veto(cuda_device_id=0):
    """One maximum filter lets an incomplete peak suppress its valid neighbor."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _test_geometry(
            cp,
            ndm=1,
            ntime=5,
            footprint=np.ones((1, 3), dtype=np.bool_),
        )
        snr = cp.zeros((1, 1, 5), dtype=cp.float32)
        token = cp.zeros((1, 1, 5), dtype=cp.uint32)
        snr[0, 0, 2] = 30.0
        snr[0, 0, 3] = 29.0
        startup = cp.ones((1, 5), dtype=cp.bool_)
        startup[0, 2] = False

        incomplete = extract_candidates(
            snr,
            token,
            geometry,
            threshold=20.0,
            beam_ids=(17,),
            source_chunk_index=4,
            startup_valid_mask=startup,
        )
        assert len(incomplete) == 1
        assert cp.asnumpy(incomplete.itime).tolist() == [2]
        assert cp.asnumpy(incomplete.snr).tolist() == [30.0]
        assert cp.asnumpy(incomplete.argmax_token).tolist() == [0]
        assert cp.asnumpy(incomplete.edge_flags).tolist() == [
            int(EdgeFlag.STARTUP_INCOMPLETE)
        ]

        startup[0, 2] = True
        complete = extract_candidates(
            snr,
            token,
            geometry,
            threshold=20.0,
            beam_ids=(17,),
            source_chunk_index=4,
            startup_valid_mask=startup,
        )
        for name in (
                "beam_id", "source_chunk_index", "tree", "idm", "itime",
                "snr", "argmax_token"):
            assert np.array_equal(
                cp.asnumpy(getattr(incomplete, name)),
                cp.asnumpy(getattr(complete, name)),
            )
        assert cp.asnumpy(complete.edge_flags).tolist() == [0]


def test_authoritative_steady_state_preserves_ordinary_results(
        cuda_device_id=0):
    """An all-valid authoritative mask changes no values, order, or old flags."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _test_geometry(cp, ndm=3, ntime=5)
        snr = cp.zeros((1, 3, 5), dtype=cp.float32)
        token = cp.zeros((1, 3, 5), dtype=cp.uint32)
        snr[0, 0, 0] = 12.0
        snr[0, 2, 4] = 11.0
        assumed = extract_candidates(
            snr, token, geometry, threshold=10.0, beam_ids=(2,),
            source_chunk_index=8, assume_steady_state=True,
        )
        authoritative = extract_candidates(
            snr, token, geometry, threshold=10.0, beam_ids=(2,),
            source_chunk_index=8,
            startup_valid_mask=cp.ones((3, 5), dtype=cp.bool_),
        )
        for name in (
                "beam_id", "source_chunk_index", "tree", "idm", "itime",
                "snr", "argmax_token", "edge_flags"):
            assert np.array_equal(
                cp.asnumpy(getattr(assumed, name)),
                cp.asnumpy(getattr(authoritative, name)),
            )


def test_streaming_halo_size_is_exact_and_requires_two(cuda_device_id=0):
    """Configured halo multiples are retained exactly above the safe minimum."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _test_geometry(
            cp,
            ndm=1,
            ntime=5,
            footprint=np.ones((1, 3), dtype=np.bool_),
        )
        for invalid in (True, 1, 2.0):
            try:
                OfflinePeakExtractor(
                    geometry,
                    threshold=10.0,
                    beam_ids=(5,),
                    assume_steady_state=True,
                    halo_size=invalid,
                )
            except (TypeError, ValueError) as exc:
                assert "halo_size" in str(exc)
            else:
                raise AssertionError(f"invalid halo_size {invalid!r} accepted")

        snr = cp.zeros((1, 1, 5), dtype=cp.float32)
        token = cp.zeros((1, 1, 5), dtype=cp.uint32)
        default = OfflinePeakExtractor(
            geometry,
            threshold=10.0,
            beam_ids=(5,),
            assume_steady_state=True,
        )
        default.process_chunk(snr, token, 0)
        assert default._tail_snr.shape[2] == 2 * geometry.time_radius

        wider = OfflinePeakExtractor(
            geometry,
            threshold=10.0,
            beam_ids=(5,),
            assume_steady_state=True,
            halo_size=4,
        )
        wider.process_chunk(snr, token, 0)
        assert wider._tail_snr.shape[2] == 4 * geometry.time_radius


def test_streaming_rejects_geometry_requiring_i_plus_2(cuda_device_id=0):
    """A footprint wider than one chunk cannot satisfy one-chunk latency."""

    import cupy as cp

    with cp.cuda.Device(cuda_device_id):
        geometry = _test_geometry(
            cp,
            ndm=1,
            ntime=4,
            footprint=np.ones((1, 11), dtype=np.bool_),
        )
        assert geometry.time_radius == 5 > geometry.ntime
        try:
            OfflinePeakExtractor(
                geometry,
                threshold=10.0,
                beam_ids=(5,),
                assume_steady_state=True,
                halo_size=2,
            )
        except ValueError as exc:
            assert "one-chunk streaming horizon" in str(exc)
        else:
            raise AssertionError("i+2-dependent peak geometry was accepted")
