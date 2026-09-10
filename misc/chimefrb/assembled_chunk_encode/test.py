#!/usr/bin/env python3
"""Spot test: reading a msgpack file that ch_frb_io wrote.

pirate's AssembledChunk parses the msgpack format by hand.  The unit tests in
pirate_frb/chimefrb/test_assembled_chunk.py check that parser against a pure-python reader,
but both were written from the same reading of ch_frb_io, so a shared misreading would pass
them.  Worse, both are exercised only on bytes that pirate's own test writer produced.

msgpack allows several encodings for the same value -- an integer may be a fixint, uint8,
uint16, uint32 or uint64, and a binary blob may carry a 1-, 2- or 4-byte length -- so a
reader that has only ever seen one writer's choices can hardcode them silently.  Here the
bytes come from ch_frb_io's own packer.

Ground truth for the metadata is by construction: this file passes the chunk parameters to
the driver, so it knows what they should be without having to trust either reader.  The
four data arrays are checked by requiring pirate's C++ and python readers to agree on them.

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

# The old reader hardwires nfreq_coarse = nt_per_assembled_chunk = 1024 and validates every
# file against them, so unlike pirate's unit tests this cannot use a small chunk. nupfreq=1
# is the smallest legal input: 1 MB of data.
PARAMS = dict(beam_id=1000, nupfreq=1, nt_per_packet=16, fpga_counts_per_sample=384,
              nrfifreq=1024, ichunk=42, binning=1, frame0_nano=1234567890, seed=137)


def main():
    t = harness.Test("assembled_chunk_encode")

    manifest = harness.build_manifest()
    if manifest:
        for line in manifest.splitlines():
            if line.startswith("date:") or line.startswith("arch:"):
                t.note(line.strip())

    t.note("chunk from ch_frb_io's packer: " +
           ", ".join("%s=%s" % (k, v) for k, v in sorted(PARAMS.items())))

    # dtype=np.uint8 is not optional: the driver reads a uint8 array, and run_driver()
    # would otherwise write whatever dtype the dummy happened to have.
    dummy = np.zeros(1, dtype=np.uint8)
    payload = harness.run_driver(HERE, dummy, params=PARAMS, dtype=np.uint8)
    payload = np.ascontiguousarray(payload, dtype=np.uint8)
    t.note("packed %d bytes" % payload.size)

    with cfrb_test.TempChunkFile(payload.tobytes()) as fn:
        c = chimefrb.AssembledChunk.from_msgpack(fn)
        ref = cfrb_test.read_msgpack(fn)

    # 1. Metadata, against what we asked the driver to build.
    nfreq_coarse, nt_per_chunk = 1024, 1024
    want = {
        "beam_id": PARAMS["beam_id"],
        "nupfreq": PARAMS["nupfreq"],
        "nt_per_packet": PARAMS["nt_per_packet"],
        "fpga_counts_per_sample": PARAMS["fpga_counts_per_sample"],
        "nrfifreq": PARAMS["nrfifreq"],
        "binning": PARAMS["binning"],
        "frame0_nano": PARAMS["frame0_nano"],
        "version": 2,
        "compression": 0,
        "nfreq_coarse": nfreq_coarse,
        "nt_per_chunk": nt_per_chunk,
        "nt_coarse": nt_per_chunk // PARAMS["nt_per_packet"],
        "nscales": nfreq_coarse * (nt_per_chunk // PARAMS["nt_per_packet"]),
        "ndata": nfreq_coarse * PARAMS["nupfreq"] * nt_per_chunk,
        "has_rfi_mask": True,
        "fpga_begin": PARAMS["ichunk"] * nt_per_chunk * PARAMS["fpga_counts_per_sample"],
    }
    got = np.array([float(getattr(c, k)) for k in sorted(want)])
    exp = np.array([float(want[k]) for k in sorted(want)])
    t.check_allclose("metadata fields (%s)" % ", ".join(sorted(want)), got, exp,
                     rtol=0.0, atol=0.0,
                     why="every one of these is either a value this test passed to the "
                         "driver, or fixed by the format; nothing here is computed")

    # 2. The four arrays, C++ reader vs python reader on the same ch_frb_io-written bytes.
    #    Both are memcpy paths, so bit-exact.
    for name in ("scales", "offsets", "data", "rfi_mask"):
        a = np.asarray(getattr(c, name)).ravel().astype(np.float64)
        b = np.asarray(getattr(ref, name)).ravel().astype(np.float64)
        t.check_allclose("%s (%d elements)" % (name, a.size), a, b, rtol=0.0, atol=0.0,
                         why="memcpy path: any difference means one of the two readers "
                             "mislocated or misinterpreted the blob")

    return t.done()


if __name__ == "__main__":
    sys.exit(main())
