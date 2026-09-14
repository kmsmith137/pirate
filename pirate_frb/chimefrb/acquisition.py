"""Surveying a directory of chimefrb data files: is it one acquisition, or several?

An "acqdir" is a directory of ``chunk_NNNNNNNN.msg`` files written by the old CHIME L1
server. The number in the name is the chunk index, and the files are supposed to form a
single acquisition: consecutive indices, all with the same
(beam_id, nupfreq, nt_per_packet, fpga_counts_per_sample, nrfifreq).

SOME OF OUR ACQUISITIONS DO NOT. There are gaps, and there are directories where the
beam_id changes partway through. Anything reading a whole acqdir as one stream -- an
AssembledChunkReader over the sorted file list, a SlabAllocator sized from the first file --
is wrong on those. So this module splits an acqdir into its LARGEST VALID
SUBACQUISITIONS: maximal runs of consecutive chunks that agree on all five fields.

The command-line front end is 'pirate_frb cfrb check_acq <acqdir>', and its -s/--split flag
turns the survey into a repair: each subacquisition is moved into a directory of its own.

The scan reads metadata only (``AssembledChunk.from_msgpack(..., metadata_only=True)``), so
it costs four small reads per file rather than a 17 MB body, and the reads go through a
thread pool: the pybind11 binding releases the GIL, so python threads really do overlap.
Measured on local NVMe, one file costs 0.32 ms cold and 0.04 ms warm single-threaded, and 16
threads bring the cold case to 0.035 ms -- a 9x speedup, hence the default. A 13770-file
acquisition is then about half a second of work, less than the interpreter takes to start.

This is the use case ``AssembledChunk``'s metadata_only flag exists for, and the reason a
scan does NOT want ``AssembledChunkReader`` (see the note in its header: what that class
parallelizes is the body read, which a scan never does).
"""

import os
import re
import collections
import concurrent.futures

from . import AssembledChunk


# The filename form the L1 server wrote. The digit count is not pinned to 8: a directory
# whose indices outgrew 8 digits should still be readable, and the report widens its columns
# to whatever it finds.
CHUNK_FILENAME_RE = re.compile(r'chunk_(\d+)\.msg\Z')

# The metadata that must be constant across an acquisition, in the order the report prints
# them. These five and no others: they are what an AssembledChunkReader's caller assumes is
# uniform, and (unlike fpga_begin, or the array contents) they have no business changing
# from one chunk of one stream to the next.
ACQ_FIELDS = ('beam_id', 'nupfreq', 'nt_per_packet', 'fpga_counts_per_sample', 'nrfifreq')

# How much of a read error to quote in the report. The C++ messages can be a paragraph.
ERROR_CHARS = 120


class Entry:
    """One file: its chunk index, its name, and either its metadata or the error reading it.

    'meta' is a tuple in ACQ_FIELDS order, or None if the file could not be read; 'error' is
    the other way round. An unreadable file is not dropped, because it is not the same thing
    as a missing one -- the report distinguishes them.
    """

    def __init__(self, ichunk, name, meta=None, error=None):
        self.ichunk = ichunk
        self.name = name
        self.meta = meta
        self.error = error


class Subacquisition:
    """A maximal run of consecutive chunks agreeing on all of ACQ_FIELDS."""

    def __init__(self, first, last, meta):
        self.first = first
        self.last = last
        self.meta = meta

    @property
    def nchunks(self):
        return self.last - self.first + 1


def list_chunk_files(acqdir):
    """Sorted (ichunk, name) pairs for the chunk_NNNNNNNN.msg files in 'acqdir'.

    Sorted by chunk INDEX, not by filename, so that a directory with mixed digit widths
    still comes out in stream order. Raises ValueError if there are none, since every caller
    wants to say so in its own words.
    """

    if not os.path.isdir(acqdir):
        raise ValueError(f'{acqdir}: not a directory')

    pairs = []
    for name in os.listdir(acqdir):
        m = CHUNK_FILENAME_RE.fullmatch(name)
        if m:
            pairs.append((int(m.group(1)), name))

    if not pairs:
        raise ValueError(f'{acqdir}: no chunk_NNNNNNNN.msg files here')

    pairs.sort()
    return pairs


def read_metadata(acqdir, pairs, nthreads=16):
    """Read every file's metadata, in parallel; return a list of Entry in chunk-index order.

    A file that fails to parse becomes an Entry with an 'error' rather than raising: the
    whole point of the scan is to report what a directory contains, and one bad file should
    not hide the other 13769.
    """

    def read_one(pair):
        (ichunk, name) = pair
        try:
            c = AssembledChunk.from_msgpack(os.path.join(acqdir, name), metadata_only=True)
        except Exception as e:
            return Entry(ichunk, name, error=str(e))
        return Entry(ichunk, name, meta=tuple(getattr(c, f) for f in ACQ_FIELDS))

    nthreads = max(1, min(int(nthreads), len(pairs)))
    with concurrent.futures.ThreadPoolExecutor(nthreads) as pool:
        return list(pool.map(read_one, pairs))


def _plural(n, noun):
    return f'{n} {noun}' if n == 1 else f'{n} {noun}s'


def _short_error(entry):
    """The useful part of one file's read error: first line, de-prefixed, truncated.

    The C++ messages are 'chimefrb: <full path>: <what went wrong>', and the reason string
    already names the file -- so the prefix is not merely redundant, it is long enough to
    push the actual error past the truncation below.
    """

    err = entry.error.splitlines()[0]
    err = err.removeprefix('chimefrb: ')

    (head, sep, tail) = err.partition(': ')
    if sep and head.endswith(entry.name):
        err = tail

    return err if len(err) <= ERROR_CHARS else err[:ERROR_CHARS - 3] + '...'


def _unreadable_reason(entries, width):
    """One reason string for a run of consecutive unreadable files."""

    err = _short_error(entries[0])

    if len(entries) == 1:
        return f'{entries[0].name} could not be read: {err}'
    return (f'{len(entries)} files could not be read ({entries[0].ichunk:0{width}d}'
            f'..{entries[-1].ichunk:0{width}d}), first error: {err}')


def split_acquisition(entries, width=8):
    """Split 'entries' (in chunk-index order) into subacquisitions.

    Returns (subacqs, gaps), where 'gaps' has ONE MORE element than 'subacqs': gaps[k] is the
    list of reasons separating subacqs[k-1] from subacqs[k], with gaps[0] whatever precedes
    the first subacquisition and gaps[-1] whatever follows the last. (Those two ends are only
    ever non-empty when a directory begins or ends with unreadable files.) A reason list is
    a list because one break can have several causes at once -- a gap AND a beam_id change.

    'width' is the zero-padding used for chunk indices inside the reason strings.
    """

    subacqs = []
    gaps = [[]]
    cur = None          # [first, last, meta] of the open subacquisition
    prev = None         # the last READABLE entry, whatever came after it
    unreadable = []     # the run of unreadable entries since 'prev'

    for e in entries:
        # The common case: this file continues the open subacquisition.
        if (e.error is None) and (cur is not None) \
                and (e.ichunk == cur[1] + 1) and (e.meta == cur[2]):
            cur[1] = e.ichunk
            prev = e
            continue

        # Anything else ends it.
        if cur is not None:
            subacqs.append(Subacquisition(cur[0], cur[1], cur[2]))
            gaps.append([])
            cur = None

        if e.error is not None:
            unreadable.append(e)
            continue

        # A readable file that did not continue a subacquisition starts a new one, and
        # everything since 'prev' is why.
        if prev is not None:
            # A missing chunk index is not the same as a file that is there but unreadable,
            # so the unreadable ones are subtracted out and reported on their own terms.
            nmissing = (e.ichunk - prev.ichunk - 1) - len(unreadable)
            if nmissing > 0 and not unreadable:
                # Nothing else lies between the two readable chunks, so the whole span is
                # missing and can be named exactly.
                gaps[-1].append(f'gap of {_plural(nmissing, "chunk")}'
                                f' ({prev.ichunk+1:0{width}d}..{e.ichunk-1:0{width}d}'
                                f' {"is" if nmissing == 1 else "are"} missing)')
            elif nmissing > 0:
                # Unreadable files are interleaved, so the missing indices need not be
                # contiguous. Name the chunks that bracket the gap instead of a range that
                # would wrongly include the files that ARE there.
                gaps[-1].append(f'gap of {_plural(nmissing, "chunk")} between'
                                f' {prev.ichunk:0{width}d} and {e.ichunk:0{width}d}')

        if unreadable:
            gaps[-1].append(_unreadable_reason(unreadable, width))

        if prev is not None:
            for (i, f) in enumerate(ACQ_FIELDS):
                if prev.meta[i] != e.meta[i]:
                    gaps[-1].append(f'{f} changed from {prev.meta[i]} to {e.meta[i]}')

        unreadable = []
        cur = [e.ichunk, e.ichunk, e.meta]
        prev = e

    if cur is not None:
        subacqs.append(Subacquisition(cur[0], cur[1], cur[2]))
        gaps.append([])
    if unreadable:
        gaps[-1].append(_unreadable_reason(unreadable, width))

    return (subacqs, gaps)


def format_report(acqdir, entries, subacqs, gaps, width=8):
    """The report 'pirate_frb cfrb check_acq' prints, as a string ending in a newline.

    A table, one row per subacquisition, with the reasons for each break on an indented line
    between the rows it separates. The five metadata fields are columns rather than
    'key=value' pairs so that the field which changed is obvious at a glance -- which is the
    whole job of this output.
    """

    lines = []
    nfiles = len(entries)
    span = (f'chunks {entries[0].ichunk:0{width}d}..{entries[-1].ichunk:0{width}d}'
            if entries else 'no chunks')
    lines.append(f'{acqdir}: {_plural(nfiles, "file")}, {span}')

    if len(subacqs) == 1 and not any(gaps):
        lines.append(f'VALID: 1 subacquisition')
    else:
        lines.append(f'INVALID: {_plural(len(subacqs), "subacquisition")}')
    lines.append('')

    # Column widths, header included, so that a wide value cannot break the alignment.
    header = ('first', 'last', 'nchunks') + ACQ_FIELDS
    rows = [(f'{s.first:0{width}d}', f'{s.last:0{width}d}', str(s.nchunks))
            + tuple(str(v) for v in s.meta) for s in subacqs]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for (i, h) in enumerate(header)]

    def row(cells):
        return '  '.join(c.rjust(w) for (c, w) in zip(cells, widths)).rstrip()

    if rows:
        lines.append(row(header))

    for (k, r) in enumerate(rows):
        lines += [f'    {g}' for g in gaps[k]]
        lines.append(row(r))
    lines += [f'    {g}' for g in gaps[-1]]

    return '\n'.join(lines) + '\n'


def check_acq(acqdir, nthreads=16):
    """The whole scan: list, read, split, format.

    Returns (report_string, entries, subacqs) -- the last two so that a caller who wants to
    act on the result (split_into_dirs()) does not have to scan the directory twice.
    """

    pairs = list_chunk_files(acqdir)
    width = max(8, len(str(pairs[-1][0])))
    entries = read_metadata(acqdir, pairs, nthreads=nthreads)
    (subacqs, gaps) = split_acquisition(entries, width=width)
    report = format_report(acqdir, entries, subacqs, gaps, width=width)
    return (report, entries, subacqs)


def subacquisition_dirs(acqdir, nsubacqs):
    """The directory each subacquisition would be split into: '<acqdir>_sub1', '_sub2', ...

    Siblings of the acqdir rather than children, so that the result is a set of directories
    that look exactly like the acqdir did, and so that the files can be moved with rename()
    rather than copied.
    """

    return [f'{acqdir}_sub{k+1}' for k in range(nsubacqs)]


def split_into_dirs(acqdir, entries, subacqs):
    """Move each subacquisition's files into its own directory. Returns [(dirname, nfiles)].

    A no-op returning [] if there are fewer than two subacquisitions: a valid acquisition is
    already what the split would produce, and moving it would only rename it.

    Files that are NOT part of a subacquisition -- the unreadable ones, and anything whose
    name is not chunk_NNNNNNNN.msg -- are left where they are, along with the acqdir itself.
    The caller is better placed than we are to decide what those are worth.

    Raises ValueError, having moved NOTHING, if a target directory already exists or if the
    acqdir is a mount point (which would make rename() an EXDEV failure partway through).
    The checks are up front because a half-finished split is much worse than a refused one.
    """

    if len(subacqs) < 2:
        return []

    dirs = subacquisition_dirs(acqdir, len(subacqs))

    existing = [d for d in dirs if os.path.exists(d)]
    if existing:
        raise ValueError(f'{existing[0]} already exists (and {len(existing)} of the'
                         f' {len(dirs)} target directories do). Refusing to split into a'
                         f' directory that is already there -- move or remove it first.')

    parent = os.path.dirname(acqdir) or '.'
    if os.stat(acqdir).st_dev != os.stat(parent).st_dev:
        raise ValueError(f'{acqdir} is a mount point: its files are on a different filesystem'
                         f' from {parent}, where the split directories would go, so they'
                         f' cannot be moved with rename(). Split by hand, or point at a'
                         f' directory inside the mount.')

    # Which subacquisition each file belongs to. Unreadable entries belong to none.
    targets = []
    for e in entries:
        if e.error is not None:
            continue
        k = [j for (j, s) in enumerate(subacqs) if s.first <= e.ichunk <= s.last]
        assert len(k) == 1, f'{e.name}: expected exactly one subacquisition, got {k}'
        targets.append((e.name, dirs[k[0]]))

    for d in dirs:
        os.mkdir(d)

    nmoved = collections.Counter()
    for (i, (name, d)) in enumerate(targets):
        try:
            os.rename(os.path.join(acqdir, name), os.path.join(d, name))
        except OSError as e:
            raise ValueError(f'{name}: {e}. THE SPLIT IS INCOMPLETE: {i} of {len(targets)}'
                             f' files were already moved into'
                             f' {", ".join(dirs)}.') from e
        nmoved[d] += 1

    return [(d, nmoved[d]) for d in dirs]
