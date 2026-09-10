#!/usr/bin/env python3
"""Spot test: decoding a chimefrb data file, pirate vs ch_frb_io.

The central check of pirate's chimefrb reader.  It covers three things at once:

  - pirate's msgpack parser agrees with ch_frb_io's;
  - pirate's decode kernels agree with ch_frb_io's assembled_chunk::decode();
  - and, for free, pirate's test WRITER produces files the real reader accepts.  That last
    one matters more than it looks: the unit tests in
    pirate_frb/chimefrb/test_assembled_chunk.py use that writer to generate every input
    they check, so if it were malformed they would all be checking fiction.  Here the bytes
    it produces are handed to ch_frb_io's own msgpack adaptor, which throws if they are not
    a well-formed assembled_chunk.

Both of ch_frb_io's decode paths are run.  assembled_chunk::make() returns a
fast_assembled_chunk (AVX2) when nt_per_packet==16 and nupfreq is even, and a plain
assembled_chunk otherwise; production CHIME data hits the fast path, so that is the
semantics pirate has to match.  They are separate implementations, so running both is how
we find out whether they agree with each other rather than assuming it.

Run me directly, or through misc/chimefrb/run_spot_tests.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness

from pirate_frb import chimefrb
from pirate_frb.chimefrb import test_assembled_chunk as cfrb_test

HERE = os.path.dirname(os.path.abspath(__file__))

SEED = 137

# The old reader hardwires nfreq_coarse = nt_per_assembled_chunk = 1024 and validates every
# file against them, so this cannot use the small chunks pirate's unit tests use.
#
# nupfreq=1 is the smallest legal input (1 MB of data, 8 MB of output) and is the default,
# per the harness's keep-the-input-small rule.  nupfreq=16 is the real CHIME geometry --
# 16 MB in, 128 MB out -- and is opt-in via CFRB_FULL_SIZE=1.  Note nupfreq=1 is ODD, so
# ch_frb_io's fast path is unavailable for it; a second config with nupfreq=2 is used to
# exercise both paths cheaply.
NUPFREQ = 16 if os.environ.get("CFRB_FULL_SIZE") else 1


def make_chunk(nupfreq, rng):
    """A random chunk in the real CHIME geometry, built with pirate's test writer.

    nrfifreq is 1024 (== nfreq_coarse), which is what real files carry and which satisfies
    ch_frb_io's requirement that nrfifreq divide nfreq.
    """

    nfreq_coarse, nt_coarse, nt_per_packet = 1024, 64, 16
    nt = nt_coarse * nt_per_packet
    nfreq = nfreq_coarse * nupfreq
    nrfifreq = nfreq_coarse

    scales = rng.uniform(0.1, 900.0, size=(nfreq_coarse, nt_coarse)).astype(np.float32)
    offsets = rng.uniform(-7300.0, 7300.0, size=(nfreq_coarse, nt_coarse)).astype(np.float32)

    # Zero an order-one fraction of blocks: real files have ~38% (blocks where no packet
    # arrived), and the encoder writes scale=offset=0 with all-zero data for those.
    zero = rng.uniform(size=scales.shape) < 0.35
    scales[zero] = 0.0
    offsets[zero] = 0.0

    data = rng.integers(0, 256, size=(nfreq, nt), dtype=np.uint8)
    data = data.reshape(nfreq_coarse, nupfreq, nt_coarse, nt_per_packet)
    data[zero[:, None, :, None].repeat(nupfreq, 1).repeat(nt_per_packet, 3)] = 0
    data = data.reshape(nfreq, nt)

    # The sentinels (0 and 255) are what decode_weights() tests, so plant plenty of both.
    hit = rng.uniform(size=data.shape) < 0.25
    data[hit] = np.where(rng.uniform(size=int(hit.sum())) < 0.5, 0, 255).astype(np.uint8)

    rfi_mask = rng.integers(0, 256, size=(nrfifreq, nt // 8), dtype=np.uint8)

    return cfrb_test.Chunk(
        beam_id=1000, nupfreq=nupfreq, nt_per_packet=nt_per_packet,
        fpga_counts_per_sample=384, binning=1, nfreq_coarse=nfreq_coarse,
        nt_coarse=nt_coarse, fpga_begin=42*nt*384, frame0_nano=1234567890,
        nrfifreq=nrfifreq, has_rfi_mask=True,
        scales=scales, offsets=offsets, data=data, rfi_mask=rfi_mask)


def main():
    t = harness.Test("assembled_chunk_decode")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    rng = np.random.default_rng(SEED)

    # nupfreq=1 exercises the reference path; nupfreq=2 additionally makes the fast path
    # legal (it needs nupfreq even). Under CFRB_FULL_SIZE, one config at the real geometry.
    configs = [(NUPFREQ, "reference")] if NUPFREQ != 1 else [(1, "reference"), (2, "reference"), (2, "fast")]
    if NUPFREQ != 1:
        configs.append((NUPFREQ, "fast"))

    for nupfreq, force in configs:
        chunk = make_chunk(nupfreq, rng)
        payload = chunk.to_msgpack(rng=rng)
        t.note("nupfreq=%d, force=%s, %d bytes" % (nupfreq, force, len(payload)))

        # dtype=np.uint8 is not optional: the driver reads a uint8 array.
        old = harness.run_driver(HERE, np.frombuffer(payload, dtype=np.uint8),
                                 params={"force": force}, dtype=np.uint8)
        old_i, old_w = old[0], old[1]

        with cfrb_test.TempChunkFile(payload) as fn:
            c = chimefrb.AssembledChunk.from_msgpack(fn)
            # apply_rfimask=False on both: ch_frb_io's decode() ignores the rfi_mask, so
            # comparing with masking on would be comparing two different quantities.
            new_i = np.asarray(c.decode_intensity(apply_rfimask=False))
            new_w = np.asarray(c.decode_weights(apply_rfimask=False))
            ref = cfrb_test.read_msgpack(fn)

        atol = cfrb_test.intensity_tolerance(ref)
        t.check_allclose("intensity (nupfreq=%d, %s)" % (nupfreq, force),
                         new_i.ravel().astype(np.float64), old_i.ravel().astype(np.float64),
                         rtol=0.0, atol=atol,
                         why="ch_frb_io computes scale*x+offset as a multiply then an add "
                             "(possibly contracted into an FMA by whatever flags oldpipe was "
                             "built with); pirate uses an explicit FMA. The bound is "
                             "eps_f32*max(2|scale*x| + |offset|), which covers both cases. It "
                             "must be ABSOLUTE, not relative: offset = -128*scale + mean makes "
                             "the intensity scale*(x-128) + mean, so the two terms nearly "
                             "cancel for x near 128 and a relative bound on the result would "
                             "be meaningless")

        t.check_allclose("weights (nupfreq=%d, %s)" % (nupfreq, force),
                         new_w.ravel().astype(np.float64), old_w.ravel().astype(np.float64),
                         rtol=0.0, atol=0.0,
                         why="a comparison against 0 and 255 involves no arithmetic, so any "
                             "difference at all is a real disagreement")

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
