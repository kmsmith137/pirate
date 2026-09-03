"""Concurrency test for the serialized-output funnel (atomic_print / AtomicPrint).

Runs C++ threads (via test_atomic_print()) and python threads concurrently, all
writing to the SAME file descriptor, then checks the resulting file for spliced,
missing, or duplicated lines.

The point is that a line emitted through the funnel is unsplittable no matter
who else is writing: the process-global mutex serializes C++ and python callers
against each other, and the single write(2) per message means even a separate
process sharing the fd cannot interleave. Both writer populations pad their
lines to the same drawn length, so any splice shows up as a malformed line
rather than being hidden by short-write luck.

WHAT ONE CALL SAMPLES IS ONE SCHEDULE, which is why every parameter is drawn.
There is no fixed point to enumerate here: the property under test is that NO
interleaving splices a line, and the interleavings a run visits depend on how
many threads of each kind are contending and on how long each message is.
"""

import os
import re
import tempfile
import threading

import numpy as np

from ..utils import atomic_print, test_atomic_print


# Lines look like "py t=<thread> i=<iter> www..." or, from the C++ side,
# "cpp t=<thread> i=<iter> <style> xxx...". Anything else is a splice.
_LINE_RE = re.compile(r'^(py t=\d+ i=\d+ w+|cpp t=\d+ i=\d+ (oneliner x+|block[12] y+|direct z+))$')


def test_atomic_out(nthreads=None, nlines=None, cpp_nthreads=None, cpp_nlines=None,
                    line_pad=None):
    """Hammer the output funnel from python and C++ threads at once.

    Every argument defaults to a draw; pass one to pin it when reproducing a failure.

    LINE LENGTH IS THE AXIS THAT MATTERS MOST. The funnel's promise is one write(2) per
    message, and a short write -- the only way a message can be split at all -- becomes a
    possibility only for long messages: PIPE_BUF (4096 on Linux) is where a pipe stops
    guaranteeing atomicity. So line_pad is drawn LOG-UNIFORMLY over a range that straddles
    it, putting about a fifth of draws above.

    The two thread counts include their degenerate values on purpose: nthreads == 1 is the
    uncontended python path, and cpp_nthreads == 0 leaves python writing alone.
    """
    if line_pad is None:
        line_pad = int(round(float(np.exp(np.random.uniform(np.log(16), np.log(16384))))))
    if nthreads is None:
        nthreads = int(np.random.randint(1, 13))
    if cpp_nthreads is None:
        cpp_nthreads = int(np.random.randint(0, 9))

    # ONE BUDGET ACROSS BOTH POPULATIONS, on lines and on bytes: the per-message cost is a
    # lock plus a write, so short lines are bounded by the count and long ones by the total
    # size. Without it the four draws multiply and a 16 kB line meets a four-figure line
    # count. Each side gets half; a C++ thread emits four lines per three iterations.
    max_lines = min(4000, max(8, 4*1000*1000 // (line_pad + 32)))
    if nlines is None:
        nlines = int(np.random.randint(1, 1 + max(1, max_lines // (2 * nthreads))))
    if cpp_nlines is None:
        cpp_nlines = int(np.random.randint(1, 1 + max(1, (3 * max_lines)
                                                      // (8 * max(cpp_nthreads, 1)))))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'atomic_out.txt')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

        try:
            def py_worker(t):
                for i in range(nlines):
                    atomic_print(f"py t={t} i={i} " + "w" * line_pad, fd=fd)

            workers = [threading.Thread(target=py_worker, args=(t,)) for t in range(nthreads)]
            # The C++ storm spawns its own threads; run it alongside the python
            # ones so both populations are contending for the same mutex. At
            # cpp_nthreads == 0 there is no storm: test_atomic_print() requires a
            # positive thread count, and the case being sampled is python alone.
            cpp = (threading.Thread(target=test_atomic_print,
                                    args=(fd, cpp_nthreads, cpp_nlines, line_pad))
                   if cpp_nthreads else None)

            for w in workers:
                w.start()
            if cpp is not None:
                cpp.start()
            for w in workers:
                w.join()
            if cpp is not None:
                cpp.join()
        finally:
            os.close(fd)

        with open(path) as f:
            text = f.read()

        # Every emitted message ends in '\n', so the final split element is ''.
        assert text.endswith('\n'), "test_atomic_out: output does not end with a newline"
        lines = text.split('\n')[:-1]

        malformed = [l for l in lines if not _LINE_RE.match(l)]
        assert not malformed, \
            (f"test_atomic_out: {len(malformed)} malformed line(s); first: {malformed[0][:200]!r}")

        # Exact counts. Per C++ thread, test_atomic_print() emits 1 line when
        # (i%3)==0, 2 when (i%3)==1, and 1 when (i%3)==2.
        expected_py = nthreads * nlines
        n0 = (cpp_nlines + 2) // 3          # i%3 == 0
        n1 = (cpp_nlines + 1) // 3          # i%3 == 1  (two lines each)
        n2 = cpp_nlines // 3                # i%3 == 2
        expected_cpp = cpp_nthreads * (n0 + 2*n1 + n2)

        got_py = sum(1 for l in lines if l.startswith('py '))
        got_cpp = sum(1 for l in lines if l.startswith('cpp '))
        assert got_py == expected_py, f"test_atomic_out: got {got_py} python lines, expected {expected_py}"
        assert got_cpp == expected_cpp, f"test_atomic_out: got {got_cpp} C++ lines, expected {expected_cpp}"

        # Each python (thread, iteration) pair must appear exactly once -- catches
        # a duplicated or dropped write that still happened to be well-formed.
        py_ids = [l.split(' w')[0] for l in lines if l.startswith('py ')]
        assert len(set(py_ids)) == expected_py, \
            f"test_atomic_out: python lines not unique ({len(set(py_ids))} distinct of {expected_py})"

        # The two-line block form must stay contiguous: block2 immediately
        # follows its own block1. This is the property a naive per-line lock
        # would NOT give us.
        for i, l in enumerate(lines):
            if ' block1 ' in l:
                tag = l.split(' block1 ')[0]
                assert i + 1 < len(lines) and lines[i+1].startswith(tag + ' block2 '), \
                    f"test_atomic_out: block form was split apart at line {i}: {l[:80]!r}"

    atomic_print(f"    test_atomic_out: pass  ({len(lines)} lines of {line_pad+16} chars"
                 f" from {nthreads} python + {cpp_nthreads} C++ threads, no splices)")
