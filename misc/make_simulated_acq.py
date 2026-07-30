#!/usr/bin/env python3
"""
Ad hoc script: write a simulated acqdir of Gaussian noise plus one injected FRB.

Generates a single-beam sequence of AssembledFrames (one per time chunk), fills
each with calibrated Gaussian noise plus a simulated FRB whose parameters you
specify, and writes them as '.asdf' files in the acqdir layout that
pirate_frb.Acquisition expects. The output can be fed straight to
'pirate_frb run_offline_dedisperser'.

The frame geometry -- frequency channelization, timekeeping, and per-zone noise
variance -- comes from an xengine_metadata yaml (e.g. configs/xengine_metadata.yml),
so the simulated stream matches the real one. If the yaml defines several beams,
only the first is written; edit the yaml's beam_ids to simulate a different one.

Contrast with SimulatedFrameFactory, which also produces noise+FRB frames but
draws each pulse's parameters from distributions, and streams frames to a
consumer rather than to disk. Use this script when you want a specified pulse
and files on disk; use the factory when you want random pulses at rate.

Usage:
   python misc/make_simulated_acq.py METADATA_YAML OUTDIR [options]
   python misc/make_simulated_acq.py --help
"""

import os
import argparse

import numpy as np

from pirate_frb.core import AssembledFrame, XEngineMetadata
from pirate_frb.simpulse import SinglePulse


def acq_filename(acqdir, beam_id, time_chunk_index):
    """Filename of one frame in an acqdir, in the server's fixed naming scheme.

    Must stay in sync with make_acq_relpath() in src_lib/FileWriter.cpp and with
    _FRAME_RE in pirate_frb/Acquisition.py, which is what reads these back.
    """
    return os.path.join(acqdir, f'frame_b{beam_id}_t{time_chunk_index}.asdf')


def make_simulated_acq(metadata_yaml, outdir, *, nchunks, ntime, dm, snr,
                       width_ms, arrival_sec=None, sm=0.0, spectral_index=0.0,
                       subband_lo_MHz=0.0, subband_hi_MHz=1.0e9):
    """Write an acqdir of noise+FRB frames; see the module docstring.

    One FRB is injected into the stream, at 'arrival_sec' after the start. Since
    a pulse is dispersed across the band it will usually span several chunks; the
    span actually written is printed.

    'arrival_sec' is the UNDISPERSED arrival time (as freq -> infinity), so the
    pulse lands later than this -- at high dm, much later. None places it 1/4 of
    the way into the stream. 'ntime' must be a positive multiple of 256.
    Narrowing (subband_lo_MHz, subband_hi_MHz) also makes pulse construction
    cheaper, which matters at CHORD-scale channel counts.
    """
    if (ntime <= 0) or (ntime % 256):
        raise RuntimeError(f'make_simulated_acq: ntime={ntime} must be a positive multiple of 256')
    if nchunks < 1:
        raise RuntimeError(f'make_simulated_acq: nchunks={nchunks} must be >= 1')

    xmd = XEngineMetadata.from_yaml_file(metadata_yaml)
    xmd.validate()

    beam_ids = list(xmd.beam_ids)
    if not beam_ids:
        raise RuntimeError(f'{metadata_yaml}: xengine_metadata has no beam_ids')

    # Single-beam acquisition: if the yaml defines several beams we write only the
    # first. The extra beams cost us nothing on disk -- each .asdf carries just its
    # own beam's metadata (from_asdf() projects beam_ids down to length 1).
    beam_id = beam_ids[0]

    # An AssembledFrame's metadata must be "frequency-scrubbed": a non-empty
    # freq_channels names the channels that ONE X-engine sender contributes,
    # which is meaningless for a frame holding the whole band. The example yaml
    # sets it (to document the field), so clear it rather than quietly writing
    # frames that violate the invariant.
    xmd.freq_channels = []

    # The pulse's time grid must match the frame's ACTUAL sample duration, which is
    # derived from the timekeeping fields rather than stated directly. randomize()
    # throws if they disagree.
    time_sample_ms = xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample / 1.0e6
    stream_sec = nchunks * ntime * time_sample_ms / 1.0e3

    if arrival_sec is None:
        arrival_sec = 0.25 * stream_sec

    nfreq = xmd.get_total_nfreq()
    print(f'make_simulated_acq: {metadata_yaml} -> {outdir}')
    print(f'  nfreq={nfreq}, beam={beam_id}, nchunks={nchunks}, ntime={ntime}, '
          f'dt={time_sample_ms:.5f} ms ({stream_sec:.3f} sec of data)')

    # Constructing the pulse does per-channel inverse FFTs, so this is the slow
    # step at CHORD-scale nfreq -- do it once and reuse it for every chunk.
    # The metadata accessors return python lists, but SinglePulse's array args go
    # through the C++ Array converter, which needs numpy.
    sp = SinglePulse(dm=dm, sm=sm, intrinsic_width=1.0e-3 * width_ms,
                     spectral_index=spectral_index,
                     undispersed_arrival_time_sec=arrival_sec,
                     time_sample_ms=time_sample_ms, snr=snr,
                     freq_edges_MHz=np.asarray(xmd.get_channel_freq_edges()),
                     freq_variances=np.asarray(xmd.get_channel_variances()),
                     subband_freq_lo_MHz=subband_lo_MHz,
                     subband_freq_hi_MHz=subband_hi_MHz)

    # sp's sample indices are absolute (its time grid is zero-based at the start of
    # the stream), so it_start/it_end say which chunks the pulse actually touches.
    chunk_lo, chunk_hi = sp.it_start // ntime, (sp.it_end - 1) // ntime
    print(f'  frb: dm={dm}, snr={snr}, width={width_ms} ms, arrival={arrival_sec:.3f} sec')
    print(f'  frb spans samples [{sp.it_start},{sp.it_end}) = chunks [{chunk_lo},{chunk_hi}]')

    if (chunk_hi < 0) or (chunk_lo >= nchunks):
        print(f'  WARNING: the frb lies entirely outside the {nchunks} chunks being written '
              f'(adjust --arrival-sec, --nchunks, or --dm)')
    elif (chunk_lo < 0) or (chunk_hi >= nchunks):
        print('  WARNING: the frb is clipped by the ends of the stream')

    os.makedirs(outdir, exist_ok=True)

    for tci in range(nchunks):
        # Map this frame's local time axis onto the pulse's stream-absolute one.
        # (Same bookkeeping SimulatedFrameFactory does for its random pulses, so
        # successive chunks receive successive time-slices of the one pulse.)
        dt_sp = tci * ntime

        # Each frame owns tens of MB of pinned host memory at CHORD-scale nfreq,
        # so keep exactly one alive: rebinding 'frame' frees the previous one.
        # Don't accumulate them in a list.
        frame = AssembledFrame.make_uninitialized(
            xmd, ntime=ntime, beam_id=beam_id, time_chunk_index=tci)

        # normalize=True and gaussian=True are preconditions of pulse injection,
        # and are what make the noise match metadata.noise_variance.
        frame.randomize(normalize=True, gaussian=True, sp=sp, dt_sp=dt_sp)

        frame.write_asdf(acq_filename(outdir, beam_id, tci))
        print(f'  wrote chunk {tci+1}/{nchunks}')

    print(f'make_simulated_acq: wrote {nchunks} frames to {outdir}')


def main():
    parser = argparse.ArgumentParser(
        description='Write a simulated acqdir: Gaussian noise + one injected FRB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('metadata_yaml',
                        help='xengine metadata yaml defining the frame geometry '
                             '(e.g. configs/xengine_metadata.yml)')
    parser.add_argument('outdir',
                        help='output acqdir of frame_b(BEAM)_t(CHUNK).asdf files (created if needed)')
    parser.add_argument('--nchunks', type=int, default=8,
                        help='number of time chunks')
    parser.add_argument('--ntime', type=int, default=2048,
                        help='time samples per chunk, must be a multiple of 256')
    parser.add_argument('--dm', type=float, default=100.0,
                        help='frb dispersion measure in pc cm^-3')
    parser.add_argument('--snr', type=float, default=30.0,
                        help='frb signal-to-noise')
    parser.add_argument('--width-ms', type=float, default=1.0,
                        help='frb intrinsic width in ms')
    parser.add_argument('--arrival-sec', type=float, default=None,
                        help='undispersed arrival time in sec from stream start '
                             '(default: 1/4 of the way in)')
    parser.add_argument('--sm', type=float, default=0.0,
                        help='frb scattering measure')
    parser.add_argument('--spectral-index', type=float, default=0.0,
                        help='frb spectral index')
    parser.add_argument('--subband-lo-MHz', type=float, default=0.0,
                        help='restrict the frb to channels above this')
    parser.add_argument('--subband-hi-MHz', type=float, default=1.0e9,
                        help='restrict the frb to channels below this')

    args = parser.parse_args()

    make_simulated_acq(args.metadata_yaml, args.outdir,
                       nchunks=args.nchunks, ntime=args.ntime,
                       dm=args.dm, snr=args.snr, width_ms=args.width_ms,
                       arrival_sec=args.arrival_sec, sm=args.sm,
                       spectral_index=args.spectral_index,
                       subband_lo_MHz=args.subband_lo_MHz,
                       subband_hi_MHz=args.subband_hi_MHz)


if __name__ == '__main__':
    main()
