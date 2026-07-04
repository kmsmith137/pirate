#!/usr/bin/env python3
"""Wrap protoc-generated .cc files in `#pragma push_macro("NDEBUG")` blocks.

conda-forge builds libabseil with `-DNDEBUG`. Several abseil headers
(notably <absl/synchronization/mutex.h>) inline certain member functions
-- e.g. absl::Mutex::Dtor() -- ONLY when NDEBUG is defined at the include
site. Any TU that includes grpc++/protobuf headers WITHOUT NDEBUG emits
an undefined reference to `absl::...::Mutex::Dtor()` that the abseil DSO
does not export, and libpirate.so then fails to load at import time.

pirate compiles WITHOUT `-DNDEBUG` globally (we don't want to lose device-
side `assert()` in CUDA kernels or stdlib asserts elsewhere), so we wrap
the abseil-touching include sites instead. For hand-written source we
put the `#pragma push_macro`/`pop_macro` block around the grpc/protobuf
includes directly. For protoc-generated .cc files, we run this script
after protoc to prepend/append the same wrap around the whole file.

We only wrap .grpc.pb.cc files (which pull in grpcpp headers and
instantiate types with absl::Mutex members). The plain .pb.cc files don't
reference Mutex::Dtor -- their undefined absl symbols are all exported
by libabseil.so and resolve at load time.

See notes/build.md for the fuller story.
"""

import argparse
import pathlib
import sys


PROLOGUE = """// AUTO-INSERTED by grpc/wrap_ndebug.py; see notes/build.md.
#pragma push_macro("NDEBUG")
#ifndef NDEBUG
#  define NDEBUG
#endif
"""

EPILOGUE = """
// AUTO-INSERTED by grpc/wrap_ndebug.py.
#pragma pop_macro("NDEBUG")
"""

# Idempotency marker so re-running is a no-op (protoc regenerates on each
# proto edit, so this script re-runs on each edit; if for some reason a
# file was already wrapped, don't wrap it twice).
SENTINEL = 'AUTO-INSERTED by grpc/wrap_ndebug.py'


def wrap_file(path: pathlib.Path) -> None:
    text = path.read_text()
    if SENTINEL in text:
        return
    path.write_text(PROLOGUE + text + EPILOGUE)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'files',
        nargs='+',
        type=pathlib.Path,
        help='.cc files to wrap in place (typically the .grpc.pb.cc stubs '
             'produced by protoc for one .proto)',
    )
    args = ap.parse_args()

    for f in args.files:
        if not f.is_file():
            print(f'{sys.argv[0]}: {f}: no such file', file=sys.stderr)
            return 1
        wrap_file(f)
    return 0


if __name__ == '__main__':
    sys.exit(main())
