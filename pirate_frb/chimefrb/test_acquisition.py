"""Tests for the acqdir scanner behind 'pirate_frb cfrb check_acq' (acquisition.py).

Dispatched from ``python -m pirate_frb test --cfrb``.

Two halves, because the module has two separable jobs:

- split_acquisition() is a pure function on a list of Entry, so it is tested on synthetic
  entries with no files at all. That is where the fiddly cases live: a break can have SEVERAL
  reasons at once, and a file that is present but unreadable is not the same thing as a
  missing chunk index -- the second must not be reported as a gap.
- the rest (finding the files, reading their metadata, the widths in the report) is tested
  end to end on a directory of real .msg files, one of them deliberately truncated.

The reasons are compared as exact strings. They are the product: a scan whose break line says
the wrong thing is worse than no scan, because the point of the tool is to make a factual
claim about an acquisition.
"""

import os
import shutil

import numpy as np

from ..utils import atomic_print
from .acquisition import (ACQ_FIELDS, Entry, check_acq, format_report, list_chunk_files,
                          read_metadata, split_acquisition, split_into_dirs,
                          subacquisition_dirs)
from .test_assembled_chunk import _SCRATCH_DIR, make_random_chunk
from .testutils import default_rng as _default_rng


# A metadata tuple in ACQ_FIELDS order, with the fields a case does not care about fixed.
def _meta(beam_id=0, nupfreq=16, nt_per_packet=16, fpga_counts_per_sample=384, nrfifreq=1024):
    return (beam_id, nupfreq, nt_per_packet, fpga_counts_per_sample, nrfifreq)


def _entries(*spec):
    """Build an Entry list from (ichunk, meta_or_error) pairs; a str means unreadable."""

    out = []
    for (ichunk, m) in spec:
        if isinstance(m, str):
            out.append(Entry(ichunk, f'chunk_{ichunk:08d}.msg', error=m))
        else:
            out.append(Entry(ichunk, f'chunk_{ichunk:08d}.msg', meta=m))
    return out


def _summary(entries):
    """(list of (first, last) per subacquisition, list of reason lists) -- the shape a case
    asserts on."""

    (subacqs, gaps) = split_acquisition(entries)
    return ([(s.first, s.last) for s in subacqs], gaps)


def check_split():
    """split_acquisition() on synthetic entries. Runs on iteration 0 only (deterministic)."""

    # One clean acquisition: no breaks, and the empty leading/trailing slots still present.
    got = _summary(_entries((10, _meta()), (11, _meta()), (12, _meta())))
    assert got == ([(10, 12)], [[], []]), got

    # A pure gap.
    got = _summary(_entries((10, _meta()), (14, _meta())))
    assert got == ([(10, 10), (14, 14)],
                   [[], ['gap of 3 chunks (00000011..00000013 are missing)'], []]), got

    # A one-chunk gap: the wording is singular on both halves.
    got = _summary(_entries((10, _meta()), (12, _meta())))
    assert got == ([(10, 10), (12, 12)],
                   [[], ['gap of 1 chunk (00000011..00000011 is missing)'], []]), got

    # A pure field change.
    got = _summary(_entries((10, _meta()), (11, _meta(beam_id=1000))))
    assert got == ([(10, 10), (11, 11)],
                   [[], ['beam_id changed from 0 to 1000'], []]), got

    # Two fields at once, reported in ACQ_FIELDS order.
    got = _summary(_entries((10, _meta()), (11, _meta(beam_id=7, nrfifreq=64))))
    assert got == ([(10, 10), (11, 11)],
                   [[], ['beam_id changed from 0 to 7',
                         'nrfifreq changed from 1024 to 64'], []]), got

    # A gap AND a field change: one break, two reasons, gap first.
    got = _summary(_entries((10, _meta()), (20, _meta(beam_id=1000))))
    assert got == ([(10, 10), (20, 20)],
                   [[], ['gap of 9 chunks (00000011..00000019 are missing)',
                         'beam_id changed from 0 to 1000'], []]), got

    # An UNREADABLE file in the middle. It breaks the acquisition, and -- the point of this
    # case -- it is NOT also reported as a gap, because the chunk index is not missing.
    got = _summary(_entries((10, _meta()), (11, 'boom'), (12, _meta())))
    assert got == ([(10, 10), (12, 12)],
                   [[], ['chunk_00000011.msg could not be read: boom'], []]), got

    # Several unreadable files coalesce into one reason...
    got = _summary(_entries((10, _meta()), (11, 'boom'), (12, 'bang'), (13, _meta())))
    assert got == ([(10, 10), (13, 13)],
                   [[], ['2 files could not be read (00000011..00000012),'
                         ' first error: boom'], []]), got

    # ... and a real gap alongside them counts only the indices that have no file at all
    # (11 is present, just unreadable). The missing indices are then not contiguous, so the
    # reason names the chunks that bracket the gap rather than a range that would be a lie.
    got = _summary(_entries((10, _meta()), (11, 'boom'), (20, _meta())))
    assert got == ([(10, 10), (20, 20)],
                   [[], ['gap of 8 chunks between 00000010 and 00000020',
                         'chunk_00000011.msg could not be read: boom'], []]), got

    # Unreadable files at the two ends land in the leading and trailing slots, which is the
    # only thing those slots are for.
    got = _summary(_entries((10, 'boom'), (11, _meta()), (12, 'bang')))
    assert got == ([(11, 11)],
                   [['chunk_00000010.msg could not be read: boom'],
                    ['chunk_00000012.msg could not be read: bang']]), got

    # Nothing readable at all: no subacquisitions, and the reason is still reported.
    got = _summary(_entries((10, 'boom')))
    assert got == ([], [['chunk_00000010.msg could not be read: boom']]), got

    # A multi-line C++ error is quoted as its first line only. (One subacquisition here: the
    # unreadable file ends it and has nothing after it, so its reason is the trailing slot.)
    got = _summary(_entries((10, _meta()), (11, 'first line\nsecond line')))
    assert got == ([(10, 10)],
                   [[], ['chunk_00000011.msg could not be read: first line']]), got

    # An empty directory cannot reach split_acquisition() through check_acq(), but the
    # function should still be total.
    assert _summary(_entries()) == ([], [[]])


def check_report_shape():
    """format_report()'s two summary lines. Runs on iteration 0 only."""

    ent = _entries((10, _meta()), (11, _meta()))
    (subacqs, gaps) = split_acquisition(ent)
    r = format_report('/acq', ent, subacqs, gaps)
    assert r.startswith('/acq: 2 files, chunks 00000010..00000011\nVALID: 1 subacquisition\n'), r
    assert r.endswith('\n')

    ent = _entries((10, _meta()), (11, _meta(beam_id=3)))
    (subacqs, gaps) = split_acquisition(ent)
    r = format_report('/acq', ent, subacqs, gaps)
    assert 'INVALID: 2 subacquisitions\n' in r, r
    # The break line is indented under the two rows it separates, and the columns line up.
    lines = r.splitlines()
    k = [i for (i, s) in enumerate(lines) if s.startswith('    beam_id changed')][0]
    assert lines[k-1].startswith('00000010') and lines[k+1].startswith('00000011'), lines
    assert len(lines[k-1]) == len(lines[k+1]) == len(lines[k-2]), lines   # header included

    # Every field is a column, named.
    assert all(f in lines[k-2] for f in ACQ_FIELDS), lines[k-2]


def _expect_raise_value(what, f):
    """Like test_assembled_chunk._expect_raise, but for the ValueError that
    list_chunk_files() raises: an unusable acqdir is a python-side error, not a C++ one."""

    try:
        f()
    except ValueError:
        return
    raise AssertionError(f'expected {what} to raise')


class _TempAcqDir:
    """Context manager: a directory of chunk_NNNNNNNN.msg files, removed on exit.

    Alongside test_assembled_chunk.py's files, in tmpfs and pid-tagged, since other agents
    run 'pirate_frb test' on this machine at the same time.
    """

    def __init__(self, tag):
        import tempfile
        d = _SCRATCH_DIR if _SCRATCH_DIR is not None else tempfile.gettempdir()
        self.dirname = os.path.join(d, f'pirate_acq_{os.getpid()}_{tag}')

    def __enter__(self):
        os.makedirs(self.dirname, exist_ok=True)
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.dirname, ignore_errors=True)
        return False

    def write(self, ichunk, payload):
        with open(os.path.join(self.dirname, f'chunk_{ichunk:08d}.msg'), 'wb') as f:
            f.write(payload)


def check_end_to_end(rng, verbose):
    """The whole path on real files: two beam_ids, a gap, and a truncated file.

    Uses ONE random chunk serialized twice, differing only in beam_id, so that the only
    metadata difference in the directory is the one the report is supposed to name. (beam_id
    is a plain attribute of the reference Chunk, not used to size any array, so it can be set
    after the draw.)
    """

    c = make_random_chunk()
    c.beam_id = 11
    payload_a = c.to_msgpack(rng=rng)
    c.beam_id = 22
    payload_b = c.to_msgpack(rng=rng)

    #  chunk  100 101 102     105 106      107        110
    #  what    A   A   A   |   A   A   |  truncated |  B
    #                     gap        unreadable    gap+beam change
    with _TempAcqDir('e2e') as acq:
        d = acq.dirname
        for i in (100, 101, 102, 105, 106):
            acq.write(i, payload_a)
        acq.write(107, payload_a[:len(payload_a)//2])      # truncated: parses, then runs out
        acq.write(110, payload_b)

        pairs = list_chunk_files(d)
        assert [i for (i, _) in pairs] == [100, 101, 102, 105, 106, 107, 110], pairs

        entries = read_metadata(d, pairs, nthreads=3)
        assert [e.ichunk for e in entries] == [i for (i, _) in pairs]
        assert [e.error is None for e in entries] == [True]*5 + [False, True], \
            [(e.ichunk, e.error) for e in entries]

        (subacqs, gaps) = split_acquisition(entries)
        assert [(s.first, s.last, s.nchunks) for s in subacqs] \
            == [(100, 102, 3), (105, 106, 2), (110, 110, 1)], \
            [(s.first, s.last) for s in subacqs]
        assert [s.meta[0] for s in subacqs] == [11, 11, 22]

        assert gaps[0] == [] and gaps[3] == [], gaps
        assert gaps[1] == ['gap of 2 chunks (00000103..00000104 are missing)'], gaps[1]
        # The truncated file and the two missing indices (108, 109) are separate reasons,
        # and the beam_id change rides along with them on the same break.
        assert len(gaps[2]) == 3, gaps[2]
        assert gaps[2][0] == 'gap of 2 chunks between 00000106 and 00000110', gaps[2]
        assert gaps[2][1].startswith('chunk_00000107.msg could not be read: '), gaps[2]
        assert gaps[2][2] == 'beam_id changed from 11 to 22', gaps[2]

        # And the whole thing through the front door.
        (report, entries2, subacqs2) = check_acq(d, nthreads=3)
        assert [(s.first, s.last) for s in subacqs2] == [(100, 102), (105, 106), (110, 110)]
        assert f'{d}: 7 files, chunks 00000100..00000110' in report, report
        assert 'INVALID: 3 subacquisitions' in report, report

        if verbose:
            atomic_print('    test_acquisition: end-to-end report:\n'
                         + ''.join(f'      {s}\n' for s in report.splitlines()))

    # A directory with no chunk files, and a path that is not a directory.
    with _TempAcqDir('empty') as acq:
        _expect_raise_value('an empty acqdir', lambda: list_chunk_files(acq.dirname))
        _expect_raise_value('a nonexistent acqdir',
                            lambda: list_chunk_files(os.path.join(acq.dirname, 'nope')))


def check_split_into_dirs(rng, verbose):
    """The -s/--split path: files really move, and a refusal really moves nothing."""

    c = make_random_chunk()
    c.beam_id = 11
    payload_a = c.to_msgpack(rng=rng)
    c.beam_id = 22
    payload_b = c.to_msgpack(rng=rng)

    #  100 101 | 104        | 107 108      + a junk file that is not a chunk at all
    #   A   A  |  A(gap)    |  B(beam)
    with _TempAcqDir('split') as acq:
        d = acq.dirname
        for i in (100, 101):
            acq.write(i, payload_a)
        acq.write(104, payload_a)
        for i in (107, 108):
            acq.write(i, payload_b)
        acq.write(102, payload_a[:len(payload_a)//2])          # unreadable
        with open(os.path.join(d, 'README.txt'), 'w') as f:
            f.write('not a chunk\n')

        (_, entries, subacqs) = check_acq(d, nthreads=3)
        assert [(s.first, s.last) for s in subacqs] == [(100, 101), (104, 104), (107, 108)], \
            [(s.first, s.last) for s in subacqs]

        dirs = subacquisition_dirs(d, len(subacqs))
        assert dirs == [d + '_sub1', d + '_sub2', d + '_sub3'], dirs

        # A target directory in the way is refused, and NOTHING moves -- the point of doing
        # the check before the first rename().
        os.mkdir(dirs[1])
        _expect_raise_value('a split into an existing directory',
                            lambda: split_into_dirs(d, entries, subacqs))
        assert not os.path.exists(dirs[0]), 'the refused split created a directory'
        assert not os.path.exists(dirs[2]), 'the refused split created a directory'
        assert len(os.listdir(d)) == 7, os.listdir(d)
        os.rmdir(dirs[1])

        moved = split_into_dirs(d, entries, subacqs)
        assert moved == [(dirs[0], 2), (dirs[1], 1), (dirs[2], 2)], moved

        assert sorted(os.listdir(dirs[0])) == ['chunk_00000100.msg', 'chunk_00000101.msg']
        assert sorted(os.listdir(dirs[1])) == ['chunk_00000104.msg']
        assert sorted(os.listdir(dirs[2])) == ['chunk_00000107.msg', 'chunk_00000108.msg']

        # What belongs to no subacquisition stays put: the unreadable chunk and the junk.
        assert sorted(os.listdir(d)) == ['README.txt', 'chunk_00000102.msg'], os.listdir(d)

        # Each piece is now a valid acquisition in its own right -- which is the whole point.
        for (k, sub) in enumerate(dirs):
            (report, _, subacqs2) = check_acq(sub, nthreads=2)
            assert len(subacqs2) == 1, report
            assert 'VALID: 1 subacquisition' in report, report
            assert subacqs2[0].meta == subacqs[k].meta

        for sub in dirs:
            shutil.rmtree(sub)

        if verbose:
            atomic_print('    test_acquisition: split into 3 directories, verified, ok')

    # Fewer than two subacquisitions: a no-op that creates nothing.
    with _TempAcqDir('nosplit') as acq:
        acq.write(100, payload_a)
        (_, entries, subacqs) = check_acq(acq.dirname, nthreads=1)
        assert len(subacqs) == 1
        assert split_into_dirs(acq.dirname, entries, subacqs) == []
        assert not os.path.exists(acq.dirname + '_sub1')


def test_acquisition(iteration=0, rng=None, verbose=False):
    rng = _default_rng(rng)

    if iteration == 0:
        check_split()
        check_report_shape()

    check_end_to_end(rng, verbose)
    check_split_into_dirs(rng, verbose)
