"""Parsed view of an acquisition directory written by a 'start_stream' RPC."""

import os
import re


# Frame files written by the server's fixed naming scheme
# '{acqdir}/frame_b(BEAM)_t(CHUNK).asdf' -- see make_acq_relpath() in
# src_lib/FileWriter.cpp, which must stay in sync with this regex.
_FRAME_RE = re.compile(r"^frame_b(\d+)_t(\d+)\.asdf$")

# Uncommitted temporary files from C++ make_tmp_filename() (file_utils.cpp): the
# final filename with a '.tmp' + make_random_hex_string(8) suffix appended.
_TMP_RE = re.compile(r"\.tmp[0-9a-fA-F]{8}$")


class Acquisition:
    """Helper class, to enumerate `frame_b{beam}_t{chunk}.asdf` files in a pirate acqdir.

    Typically, the acqdir will be created by either `pirate_frb rpc_start_stream`,
    or a triggered `WriteFiles` RPC. The constructor raises an exception if the
    acqdir contains a file that's not of the form `frame_b{beam}_t{chunk}.asdf`,
    or if any beam's time chunk indices are non-contiguous. Different beams may
    span different time-index ranges;

    Attributes
    ----------
    acqdir : str
        The acquisition directory that was parsed.
    beam_ids : list of int
        Sorted beam ids present in the acqdir.
    per_beam_filenames : dict[int, list of str]
        beam_id -> filenames (with the acqdir prefix, i.e. directly openable),
        sorted by time chunk.
    per_beam_chunk_indices : dict[int, list of int]
        beam_id -> its contiguous, sorted time-chunk indices (elementwise parallel
        to per_beam_filenames[beam_id]).
    """

    def __init__(self, acqdir):
        self.acqdir = acqdir

        per_beam = {}   # beam_id -> list of (time_chunk, full_filename)
        for name in sorted(os.listdir(acqdir)):
            m = _FRAME_RE.match(name)
            if m is not None:
                b, chunk = int(m.group(1)), int(m.group(2))
                per_beam.setdefault(b, []).append((chunk, os.path.join(acqdir, name)))
            elif _TMP_RE.search(name):
                print(f"Acquisition: ignoring temporary (uncommitted) file {name!r} in {acqdir}")
            else:
                raise RuntimeError(f"Acquisition: {acqdir} contains entry {name!r} which does not "
                                   f"match the expected pattern 'frame_b(BEAM)_t(CHUNK).asdf'")

        self.beam_ids = sorted(per_beam)
        self.per_beam_filenames = {}
        self.per_beam_chunk_indices = {}

        for b in self.beam_ids:
            entries = sorted(per_beam[b])   # list of (time_chunk, full_filename)
            chunks = [c for c, _ in entries]
            for i in range(1, len(chunks)):
                if chunks[i] != chunks[i-1] + 1:
                    raise RuntimeError(
                        f"Acquisition: beam {b} time indices are not contiguous in {acqdir}: "
                        f"chunk {chunks[i-1]} is followed by {chunks[i]} "
                        f"(expected {chunks[i-1] + 1})")
            self.per_beam_filenames[b] = [fn for _, fn in entries]
            self.per_beam_chunk_indices[b] = chunks
