#!/usr/bin/env python3
"""Write simulated acquisition frames containing one or more deterministic FRB pulses."""

import argparse
import os
from dataclasses import dataclass

import numpy as np

from pirate_frb.core import AssembledFrame, XEngineMetadata
from pirate_frb.simpulse import SinglePulse, dispersion_delay


def acq_filename(acqdir, beam_id, time_chunk_index):
    """Return the filename used for one acquisition frame."""
    return os.path.join(acqdir, f"frame_b{beam_id}_t{time_chunk_index}.asdf")


def _float_list(values):
    if np.isscalar(values):
        return [float(values)]
    return [float(x) for x in values]


def _broadcast_burst_parameter(values, nbursts, option_name):
    values = _float_list(values)
    if len(values) == 1:
        return values * nbursts
    if len(values) == nbursts:
        return values
    raise ValueError(
        f"{option_name} expects either one value or {nbursts} values "
        f"(one per --toa); got {len(values)}"
    )


def _normalize_burst_parameters(*, toa, arrival_sec, width_ms, snr):
    """Validate burst counts and broadcast scalar width/SNR values."""
    if toa is not None and arrival_sec is not None:
        raise ValueError("--toa and legacy --arrival-sec are mutually exclusive")

    requested_toas = None if toa is None else _float_list(toa)
    if requested_toas is not None and not requested_toas:
        raise ValueError("--toa requires at least one value")

    if arrival_sec is not None and not np.isscalar(arrival_sec):
        raise ValueError("--arrival-sec accepts exactly one value")

    nbursts = len(requested_toas) if requested_toas is not None else 1
    widths = _broadcast_burst_parameter(width_ms, nbursts, "--width-ms")
    snrs = _broadcast_burst_parameter(snr, nbursts, "--snr")
    return requested_toas, widths, snrs


def _toa_to_infinite_arrivals(requested_toas, dm, reference_frequency_MHz):
    """Convert bottom-of-band dedispersed TOAs to SinglePulse infinite-frequency times."""
    delay_sec = float(dispersion_delay(dm, reference_frequency_MHz))
    arrivals = [float(toa) - delay_sec for toa in requested_toas]
    return arrivals, delay_sec


def _dedispersed_reference_frequency_MHz(xmd):
    """Return PIRATE's decoded timestamp reference: the full-band low edge."""
    return float(np.asarray(xmd.get_channel_freq_edges(), dtype=np.float64)[0])


def _pulse_observation_status(pulse, total_samples):
    if pulse.it_end <= 0 or pulse.it_start >= total_samples:
        return "completely outside"
    if pulse.it_start < 0 or pulse.it_end > total_samples:
        return "clipped"
    return "fully inside"


@dataclass
class SimulatedAcquisition:
    """Prepared single-beam simulation that yields frames without writing them."""

    metadata_yaml: str
    xmd: object
    beam_id: int
    nchunks: int
    ntime: int
    pulses: list
    burst_summaries: list
    time_sample_ms: float
    reference_frequency_MHz: float
    time_input: str

    @property
    def stream_sec(self):
        return self.nchunks * self.ntime * self.time_sample_ms / 1.0e3

    def iter_frames(self):
        """Yield independently owned frames in increasing chunk order."""
        for time_chunk_index in range(self.nchunks):
            frame = AssembledFrame.make_uninitialized(
                self.xmd,
                ntime=self.ntime,
                beam_id=self.beam_id,
                time_chunk_index=time_chunk_index,
            )
            frame.randomize_many(
                normalize=True,
                gaussian=True,
                pulses=self.pulses,
                dt_sp=time_chunk_index * self.ntime,
            )
            yield frame

    def print_summary(self, outdir=None):
        print(f"metadata: {self.metadata_yaml}")
        if outdir is not None:
            print(f"output directory: {outdir}")
        print(f"beam: {self.beam_id}")
        print(f"nfreq: {self.xmd.get_total_nfreq()}")
        print(f"nchunks: {self.nchunks}")
        print(f"ntime per chunk: {self.ntime}")
        print(f"time resolution: {self.time_sample_ms:.6f} ms")
        print(f"total duration: {self.stream_sec:.3f} s")
        print(f"burst time input: {self.time_input}")
        for summary in self.burst_summaries:
            print(f"burst {summary['index']}:")
            print(f"  DM: {summary['dm']} pc cm^-3")
            print(f"  requested dedispersed TOA: {summary['requested_toa']:.9f} s")
            print(f"  reference frequency: {self.reference_frequency_MHz:.9f} MHz")
            print(f"  dispersion delay: {summary['dispersion_delay']:.9f} s")
            print(
                "  internal arrival at infinite frequency: "
                f"{summary['internal_arrival']:.9f} s"
            )
            print(f"  S/N: {summary['snr']}")
            print(f"  width: {summary['width_ms']} ms")
            print(
                f"  sample range: [{summary['sample_start']}, "
                f"{summary['sample_end']})"
            )
            print(
                f"  chunk range: [{summary['first_chunk']}, "
                f"{summary['last_chunk']}]"
            )
            print(f"  observation status: {summary['status']}")


def create_simulated_acquisition(
    metadata_yaml,
    *,
    nchunks,
    ntime,
    dm,
    snr,
    width_ms,
    toa=None,
    arrival_sec=None,
    sm=0.0,
    spectral_index=0.0,
    subband_lo_MHz=0.0,
    subband_hi_MHz=1.0e9,
):
    """Prepare PIRATE simulation frames for in-memory or file-backed use."""
    requested_toas, widths, snrs = _normalize_burst_parameters(
        toa=toa,
        arrival_sec=arrival_sec,
        width_ms=width_ms,
        snr=snr,
    )

    if ntime <= 0 or ntime % 256:
        raise RuntimeError(f"ntime={ntime} must be a positive multiple of 256")
    if nchunks < 1:
        raise RuntimeError(f"nchunks={nchunks} must be at least 1")

    xmd = XEngineMetadata.from_yaml_file(metadata_yaml)
    xmd.validate()
    beam_ids = list(xmd.beam_ids)
    if not beam_ids:
        raise RuntimeError(f"{metadata_yaml} contains no beam IDs")

    # This offline generator intentionally remains single-beam. Match the
    # projection performed by AssembledFrame.from_asdf so these same frames can
    # be passed directly to OfflineDedisperser without a disk round trip.
    beam_id = beam_ids[0]
    beam_index = beam_ids.index(beam_id)
    xmd.beam_ids = [beam_id]
    xmd.beam_positions_x = [xmd.beam_positions_x[beam_index]]
    xmd.beam_positions_y = [xmd.beam_positions_y[beam_index]]
    xmd.freq_channels = []
    xmd.validate()

    time_sample_ms = xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.0e6
    total_samples = nchunks * ntime
    stream_sec = total_samples * time_sample_ms / 1.0e3
    freq_edges_MHz = np.asarray(xmd.get_channel_freq_edges(), dtype=np.float64)
    freq_variances = np.asarray(xmd.get_channel_variances(), dtype=np.float64)
    reference_frequency_MHz = _dedispersed_reference_frequency_MHz(xmd)

    if requested_toas is None:
        if arrival_sec is None:
            arrival_sec = 0.25 * stream_sec
        infinite_arrivals = [float(arrival_sec)]
        delay_sec = float(dispersion_delay(dm, reference_frequency_MHz))
        requested_toas = [infinite_arrivals[0] + delay_sec]
        time_input = "legacy infinite-frequency --arrival-sec"
    else:
        infinite_arrivals, delay_sec = _toa_to_infinite_arrivals(
            requested_toas, dm, reference_frequency_MHz
        )
        time_input = "dedispersed --toa at the full-band low edge"

    pulses = []
    burst_summaries = []
    for iburst, (requested_toa, internal_arrival, burst_width, burst_snr) in enumerate(
        zip(requested_toas, infinite_arrivals, widths, snrs)
    ):
        pulse = SinglePulse(
            dm=dm,
            sm=sm,
            intrinsic_width=1.0e-3 * burst_width,
            spectral_index=spectral_index,
            undispersed_arrival_time_sec=internal_arrival,
            time_sample_ms=time_sample_ms,
            snr=burst_snr,
            freq_edges_MHz=freq_edges_MHz,
            freq_variances=freq_variances,
            subband_freq_lo_MHz=subband_lo_MHz,
            subband_freq_hi_MHz=subband_hi_MHz,
        )
        pulses.append(pulse)
        burst_summaries.append(
            {
                "index": iburst,
                "dm": float(dm),
                "requested_toa": float(requested_toa),
                "dispersion_delay": delay_sec,
                "internal_arrival": float(internal_arrival),
                "snr": float(burst_snr),
                "width_ms": float(burst_width),
                "sample_start": int(pulse.it_start),
                "sample_end": int(pulse.it_end),
                "first_chunk": int(pulse.it_start // ntime),
                "last_chunk": int((pulse.it_end - 1) // ntime),
                "status": _pulse_observation_status(pulse, total_samples),
            }
        )

    return SimulatedAcquisition(
        metadata_yaml=str(metadata_yaml),
        xmd=xmd,
        beam_id=beam_id,
        nchunks=nchunks,
        ntime=ntime,
        pulses=pulses,
        burst_summaries=burst_summaries,
        time_sample_ms=time_sample_ms,
        reference_frequency_MHz=reference_frequency_MHz,
        time_input=time_input,
    )


def make_simulated_acq(
    metadata_yaml,
    outdir,
    *,
    nchunks,
    ntime,
    dm,
    snr,
    width_ms,
    toa=None,
    arrival_sec=None,
    sm=0.0,
    spectral_index=0.0,
    subband_lo_MHz=0.0,
    subband_hi_MHz=1.0e9,
):
    """Write frames produced by :func:`create_simulated_acquisition`."""
    simulation = create_simulated_acquisition(
        metadata_yaml,
        nchunks=nchunks,
        ntime=ntime,
        dm=dm,
        snr=snr,
        width_ms=width_ms,
        toa=toa,
        arrival_sec=arrival_sec,
        sm=sm,
        spectral_index=spectral_index,
        subband_lo_MHz=subband_lo_MHz,
        subband_hi_MHz=subband_hi_MHz,
    )
    simulation.print_summary(outdir)

    os.makedirs(outdir, exist_ok=True)
    for time_chunk_index, frame in enumerate(simulation.iter_frames()):
        filename = acq_filename(outdir, simulation.beam_id, time_chunk_index)
        frame.write_asdf(filename)
        print(f"wrote chunk {time_chunk_index + 1}/{nchunks}: {filename}")

    print(f"Finished: wrote {nchunks} frames to {outdir}")


def _make_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate Gaussian-noise acquisition frames containing deterministic FRBs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("metadata_yaml", help="X-engine metadata YAML file")
    parser.add_argument("outdir", help="directory in which to write the acquisition frames")
    parser.add_argument("--nchunks", type=int, default=10, help="number of time chunks")
    parser.add_argument(
        "--ntime", type=int, default=2048,
        help="time samples per chunk; must be a multiple of 256",
    )
    parser.add_argument(
        "--dm", type=float, default=100.0,
        help="shared dispersion measure in pc cm^-3",
    )
    parser.add_argument(
        "--snr", type=float, nargs="+", default=[30.0],
        help="one shared S/N value or one value per --toa",
    )
    parser.add_argument(
        "--width-ms", type=float, nargs="+", default=[1.0],
        help="one shared intrinsic width or one value per --toa",
    )
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--toa", type=float, nargs="+",
        help="dedispersed arrival time(s) at the lowest edge of the full observing band",
    )
    time_group.add_argument(
        "--arrival-sec", type=float, default=argparse.SUPPRESS,
        help=("legacy: one arrival time extrapolated to infinite frequency; "
              "if neither time option is given, 10.0 seconds is used"),
    )
    parser.add_argument("--sm", type=float, default=0.0, help="shared scattering measure")
    parser.add_argument(
        "--spectral-index", type=float, default=0.0, help="shared spectral index"
    )
    parser.add_argument(
        "--subband-lo-MHz", type=float, default=0.0,
        help="lowest frequency containing the FRBs",
    )
    parser.add_argument(
        "--subband-hi-MHz", type=float, default=1.0e9,
        help="highest frequency containing the FRBs",
    )
    return parser


def main(argv=None):
    parser = _make_arg_parser()
    args = parser.parse_args(argv)
    args.arrival_sec = getattr(args, "arrival_sec", None)
    if args.toa is None and args.arrival_sec is None:
        args.arrival_sec = 10.0

    try:
        _normalize_burst_parameters(
            toa=args.toa,
            arrival_sec=args.arrival_sec,
            width_ms=args.width_ms,
            snr=args.snr,
        )
    except ValueError as exc:
        parser.error(str(exc))

    make_simulated_acq(
        args.metadata_yaml,
        args.outdir,
        nchunks=args.nchunks,
        ntime=args.ntime,
        dm=args.dm,
        snr=args.snr,
        width_ms=args.width_ms,
        toa=args.toa,
        arrival_sec=args.arrival_sec,
        sm=args.sm,
        spectral_index=args.spectral_index,
        subband_lo_MHz=args.subband_lo_MHz,
        subband_hi_MHz=args.subband_hi_MHz,
    )


if __name__ == "__main__":
    main()
