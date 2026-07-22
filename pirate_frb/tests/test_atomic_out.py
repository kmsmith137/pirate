"""Concurrency test for the serialized-output funnel (atomic_print / AtomicPrint).

Runs C++ threads (via test_atomic_print()) and python threads concurrently, all
writing to the SAME file descriptor, then checks the resulting file for spliced,
missing, or duplicated lines.

The point is that a line emitted through the funnel is unsplittable no matter
who else is writing: the process-global mutex serializes C++ and python callers
against each other, and the single write(2) per message means even a separate
process sharing the fd cannot interleave. Both writer populations here emit long
lines (a few hundred chars), so any splice shows up as a malformed line rather
than being hidden by short-write luck.
"""

import os
import re
import tempfile
import threading

from ..utils import atomic_print, test_atomic_print


# Lines look like "py t=<thread> i=<iter> www..." or, from the C++ side,
# "cpp t=<thread> i=<iter> <style> xxx...". Anything else is a splice.
_LINE_RE = re.compile(r'^(py t=\d+ i=\d+ w+|cpp t=\d+ i=\d+ (oneliner x+|block[12] y+|direct z+))$')


def test_atomic_out(nthreads=8, nlines=200, cpp_nthreads=8, cpp_nlines=100):
    """Hammer the output funnel from python and C++ threads at once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'atomic_out.txt')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)

        try:
            def py_worker(t):
                for i in range(nlines):
                    atomic_print(f"py t={t} i={i} " + "w" * 200, fd=fd)

            workers = [threading.Thread(target=py_worker, args=(t,)) for t in range(nthreads)]
            # The C++ storm spawns its own threads; run it alongside the python
            # ones so both populations are contending for the same mutex.
            cpp = threading.Thread(target=test_atomic_print, args=(fd, cpp_nthreads, cpp_nlines))

            for w in workers:
                w.start()
            cpp.start()
            for w in workers:
                w.join()
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

    atomic_print(f"    test_atomic_out: pass  ({len(lines)} lines from "
                 f"{nthreads} python + {cpp_nthreads} C++ threads, no splices)")
