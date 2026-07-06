"""Helpers for listing acquisition directories written by a 'start_stream' RPC."""

import os
import re


# Frame files written by FrbSearchClient's default pattern
# 'streams/{acq_name}/frame_b(BEAM)_t(CHUNK).asdf' (see FrbSearchClient.start_stream).
_FRAME_RE = re.compile(r"^frame_b(\d+)_t(\d+)\.asdf$")

# Uncommitted temporary files from C++ make_tmp_filename() (file_utils.cpp): the
# final filename with a '.tmp' + make_random_hex_string(8) suffix appended.
_TMP_RE = re.compile(r"\.tmp[0-9a-fA-F]{8}$")


def list_acqdir(acqdir, beam_id=None):
    """
    Parses and sorts filenames from an acqdir that was populated with a 'start_stream' RPC.

    By default (beam_id=None), returns a list of (beam_id, filename_list) pairs, where beam_ids
    are sorted, and per-beam filenames are sorted by time chunk.

    If beam_id is specified, then returns the filename_list for the specific beam.

    The filename_pattern is assumed to be 'frame_b(BEAM)_t(CHUNK).asdf'. This is the
    default in FrbSearchClient. Raises an exception if the acqdir contains files that don't match
    the pattern. (Exception: filenames which look like they're from C++ make_tmp_filename()
    are ignored, with a message printed to stdout.)

    For each beam, the time (chunk) indices must be contiguous -- a gap or duplicate
    raises an exception. (Different beams may span different time-index ranges.)

    Returned filenames include the 'acqdir' prefix (i.e. are directly openable).
    """
    per_beam = {}   # beam_id -> list of (time_chunk, full_filename)

    for name in sorted(os.listdir(acqdir)):
        m = _FRAME_RE.match(name)
        if m is not None:
            b, chunk = int(m.group(1)), int(m.group(2))
            per_beam.setdefault(b, []).append((chunk, os.path.join(acqdir, name)))
        elif _TMP_RE.search(name):
            print(f"list_acqdir: ignoring temporary (uncommitted) file {name!r} in {acqdir}")
        else:
            raise RuntimeError(f"list_acqdir: {acqdir} contains entry {name!r} which does not match "
                               f"the expected pattern 'frame_b(BEAM)_t(CHUNK).asdf'")

    # Sort each beam's files by time chunk (dropping the chunk sort key), after
    # checking that the beam's time indices are contiguous (no gap or duplicate).
    result = []
    for b in sorted(per_beam):
        entries = sorted(per_beam[b])   # list of (time_chunk, full_filename)
        chunks = [c for c, _ in entries]
        for i in range(1, len(chunks)):
            if chunks[i] != chunks[i-1] + 1:
                raise RuntimeError(
                    f"list_acqdir: beam {b} time indices are not contiguous in {acqdir}: "
                    f"chunk {chunks[i-1]} is followed by {chunks[i]} "
                    f"(expected {chunks[i-1] + 1})")
        result.append((b, [fn for _, fn in entries]))

    if beam_id is None:
        return result

    for b, filenames in result:
        if b == beam_id:
            return filenames
    raise RuntimeError(f"list_acqdir: beam_id={beam_id} not found in {acqdir} "
                       f"(beams present: {sorted(per_beam)})")
