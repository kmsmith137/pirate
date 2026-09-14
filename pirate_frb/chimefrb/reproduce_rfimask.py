"""The 'pirate_frb cfrb reproduce_rfimask' subcommand: rerun an acquisition's RFI chain on
the GPU, with ChimePreDedisperser, and compare the mask with the one the CHIME L1 server
saved in real time.

The GPU counterpart of misc/chimefrb/01-reproduce-rfimask.py, which does the same with the
old code, and whose file rules and report format this follows so that the two read side by
side: the files must be consecutive, they are trimmed at either end to whole blocks aligned
the way the L1 server aligned them, and the result is the fraction of mask samples in each
of the four (saved, new) x (masked, unmasked) combinations.
"""

import os
import queue
import re
import sys
import threading
import time

import numpy as np

from ..utils import atomic_print


# The L1 server's 'intensity_prescale' (ch_frb_l1/l1_configs/l1_production_8beam.yaml), for
# the reminder at the end of the report.
REALTIME_PRESCALE = 1.0e-4

_FILENAME_RE = re.compile(r'^chunk_(\d{8})\.msg$')


def list_chunk_files(acqdir):
    """Sorted (name, ichunk) pairs for the chunk_NNNNNNNN.msg files in 'acqdir'. Raises
    ValueError if there are none, or if any ichunk is skipped: a gap would silently shift
    every later block boundary, so it is refused rather than papered over."""

    if not os.path.isdir(acqdir):
        raise ValueError(f'{acqdir} is not a directory')

    pairs = []
    for name in sorted(os.listdir(acqdir)):
        m = _FILENAME_RE.match(name)
        if m:
            pairs.append((name, int(m.group(1))))
    if not pairs:
        raise ValueError(f'no chunk_NNNNNNNN.msg files in {acqdir}')

    for ((n0, i0), (n1, i1)) in zip(pairs[:-1], pairs[1:]):
        if i1 != i0 + 1:
            raise ValueError(f'files are not consecutive: {n0} is followed by {n1}'
                             f' ({i1 - i0 - 1} chunk(s) missing)')
    return pairs


def trim_to_blocks(pairs, chunks_per_block, nfiles=0):
    """Drop the files at either end that do not belong to a complete block of
    'chunks_per_block' chunks aligned to ichunk % chunks_per_block == 0 (the L1 server's
    alignment), and print what was dropped. 'nfiles', if nonzero, then keeps only the first
    that many (a multiple of chunks_per_block)."""

    n = chunks_per_block
    starts = [k for (k, (_, i)) in enumerate(pairs) if i % n == 0]
    ends = [k for (k, (_, i)) in enumerate(pairs) if i % n == n - 1]
    if (not starts) or (not ends) or (ends[-1] < starts[0]):
        raise ValueError(f'no complete block: need {n} consecutive files, the first with ichunk % {n} == 0')

    (k0, k1) = (starts[0], ends[-1] + 1)
    for (where, dropped) in (('start', pairs[:k0]), ('end', pairs[k1:])):
        if dropped:
            atomic_print(f'Trimmed {len(dropped)} file(s) at the {where}, to align with the'
                         f' {n}-chunk blocks: ' + ' '.join(name for (name, _) in dropped))

    kept = pairs[k0:k1]
    if nfiles:
        if nfiles % n:
            raise ValueError(f'--nfiles must be a multiple of chunks_per_block = {n}')
        kept = kept[:nfiles]
    return kept


def _unpack(packed):
    """(nrfifreq, nt/8) packed bytes -> (nrfifreq, nt) bool, LSB-first, True = unmasked."""
    return np.unpackbits(packed, axis=1, bitorder='little').astype(bool)


def reproduce_rfimask(acqdir, yaml_file, prescale=1.0, ntime=4096, nfiles=0, nthreads=4,
                      gpu=0, pinned=False, verbose=False):
    """Run the comparison; returns the exit status (0 identical, 2 different)."""

    from . import AssembledChunk, AssembledChunkReader, ChimePreDedisperser, Pipeline
    from ..core import SlabAllocator

    pairs = list_chunk_files(acqdir)

    # The first file's header fixes the chunk length and channel count, which the chain is
    # built for; every later file must match (ChimePreDedisperser checks).
    md = AssembledChunk.from_msgpack(os.path.join(acqdir, pairs[0][0]), metadata_only=True)
    if ntime % md.nt:
        raise ValueError(f'--ntime {ntime} is not a multiple of the chunk length {md.nt}')
    chunks_per_block = ntime // md.nt

    chain = Pipeline.read_yaml_file(yaml_file, nbeams=1, nfreq=md.nfreq, ntime=ntime)
    if chain.get_mask_extractor() is None:
        raise ValueError(f'{yaml_file}: the chain contains no RfiMaskExtractor, so there is no'
                         f' mask to compare; convert the legacy json with "pirate_frb cfrb'
                         f' json2yaml", which places one at the last mask_counter')

    pairs = trim_to_blocks(pairs, chunks_per_block, nfiles)
    names = [name for (name, _) in pairs]
    paths = [os.path.join(acqdir, name) for name in names]
    atomic_print(f'Using {len(names)} files ({len(names) // chunks_per_block} blocks of {ntime}'
                 f' samples): {names[0]} .. {names[-1]}')
    atomic_print(f'Chain: {yaml_file} at (nbeams, nfreq, ntime) = (1, {md.nfreq}, {ntime}),'
                 f' {chunks_per_block} chunks of {md.nt} samples per block')

    cpd = ChimePreDedisperser(chain, prescale, cuda_device_id=gpu)

    # The producer: files -> chunks -> the driver, passing each file's saved mask to the
    # consumer through a queue (the chunk itself is dropped once put_chunk() returns).
    saved_masks = queue.Queue()
    producer_error = []

    def producer():
        try:
            allocator = SlabAllocator('af_rhost') if pinned else None
            with AssembledChunkReader(paths, nthreads, allocator) as reader:
                for chunk in reader:
                    if not chunk.has_rfi_mask:
                        raise ValueError(f'{chunk.filename}: no RFI mask saved (nrfifreq={chunk.nrfifreq});'
                                         f' the L1 server that wrote this acquisition was not saving masks')
                    if chunk.nrfifreq != cpd.nrfifreq:
                        raise ValueError(f'{chunk.filename}: the saved mask has {chunk.nrfifreq} channels,'
                                         f' but the chain\'s extractor runs at {cpd.nrfifreq}')
                    saved_masks.put(np.array(chunk.rfi_mask))
                    cpd.put_chunk(chunk)
            cpd.finish()
        except BaseException as e:
            producer_error.append(e)
            cpd.stop(e)

    t0 = time.time()
    thread = threading.Thread(target=producer, name='reproduce_rfimask_producer')
    thread.start()

    # The consumer: one mask per file, against the saved one. counts[saved][new], 1 = unmasked.
    counts = np.zeros((2, 2), dtype=np.int64)
    disagreements = []
    nfiles_done = 0
    try:
        while True:
            new = cpd.get_rfimask()
            if new is None:
                break
            saved = saved_masks.get()
            (s, n) = (_unpack(saved), _unpack(new))
            n11 = int(np.count_nonzero(s & n))
            n10 = int(np.count_nonzero(s & ~n))
            n01 = int(np.count_nonzero(~s & n))
            n00 = s.size - n11 - n10 - n01
            counts += np.array([[n00, n01], [n10, n11]], dtype=np.int64)

            name = names[nfiles_done]
            if n10 + n01:
                nrows = int(np.count_nonzero((s != n).all(axis=1)))
                disagreements.append((name, n10 + n01, nrows))
            nfiles_done += 1
            if verbose:
                atomic_print(f'    {name}: {n10 + n01} samples disagree, unmasked fraction saved'
                             f' {s.mean():.4f} new {n.mean():.4f}')
            elif nfiles_done % 200 == 0:
                atomic_print(f'    {nfiles_done} / {len(names)} files, {time.time() - t0:.0f} s')
    finally:
        thread.join()
    run_time = time.time() - t0

    if producer_error:
        raise producer_error[0]
    if nfiles_done != len(names):
        raise RuntimeError(f'got {nfiles_done} masks for {len(names)} files')

    total = float(counts.sum())
    lines = [f'Compared {len(names)} files ({len(names) // chunks_per_block} blocks), {int(total)} mask'
             f' samples, run time {run_time:.0f} s']
    for (label, c) in (('Unmasked in both saved files and new run  ', counts[1, 1]),
                       ('Masked in both saved files and new run    ', counts[0, 0]),
                       ('Unmasked in saved files, masked in new run', counts[1, 0]),
                       ('Masked in saved files, unmasked in new run', counts[0, 1])):
        lines.append(f'  {label} = {100.0 * c / total:11.6f}%  ({c} samples)')

    if not disagreements:
        lines.append('  The masks are identical.')
    else:
        lines.append(f'  The masks differ in {len(disagreements)} of {len(names)} files.')
        listed = disagreements if len(disagreements) <= 10 else sorted(disagreements, key=lambda x: -x[1])[:10]
        for (name, nsamples, nrows) in listed:
            lines.append(f'    {name}: {nsamples} samples, of which {nrows * md.nt} whole {md.nt}-sample rows')
        if len(disagreements) > 10:
            lines.append('    (the 10 worst; -v lists every file)')
        lines.append('  (Whole rows point at std_dev_clipper decisions, scattered samples at intensity_clippers or detrenders.)')

    lines.append('')
    lines.append(f'prescale used in this run: {prescale:g}  (the real-time pipeline used'
                 f' intensity_prescale = {REALTIME_PRESCALE:g})')
    atomic_print('\n'.join(lines))

    return 0 if not disagreements else 2
