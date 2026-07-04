#!/usr/bin/env python3
"""Post-process protoc-generated Python stubs so they can live under a package.

By default `python -m grpc_tools.protoc` emits `import <stem>_pb2` at the top
of each `_pb2_grpc.py` -- as if the generated modules will sit on
`sys.path`. Inside a subpackage (here: `pirate_frb.rpc.grpc`) that import
fails, so we rewrite each such line to `from . import <stem>_pb2 [as ...]`.

This replaces the equivalent invocation of `protoletariat`:

    protol --create-package --in-place --python-out <dir> \\
        protoc --proto-path grpc <name>.proto

We swapped protoletariat out because its conda-forge package pins
`protobuf < 6`, which in turn pins `libgrpc <= 1.71`. See notes/build.md.

The Makefile handles __init__.py separately (a plain `touch`), so this script
does not create it -- it only rewrites imports in the .py files listed on
the command line, in place.
"""

import argparse
import pathlib
import re
import sys


# Match a top-of-line `import <stem>_pb2` or `import <stem>_pb2 as <alias>`.
# Anchored to the start of a line via re.MULTILINE, matched non-greedily so
# it never spans multiple statements.
_IMPORT_RE = re.compile(
    r'^(import\s+\w+_pb2(?:\s+as\s+\w+)?)\s*$',
    re.MULTILINE,
)


def rewrite_file(path: pathlib.Path) -> None:
    text = path.read_text()
    new_text = _IMPORT_RE.sub(r'from . \1', text)
    if new_text != text:
        path.write_text(new_text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'files',
        nargs='+',
        type=pathlib.Path,
        help='.py files to rewrite in place (typically the _pb2.py + '
             '_pb2_grpc.py pair produced by grpc_tools.protoc for one .proto)',
    )
    args = ap.parse_args()

    for f in args.files:
        if not f.is_file():
            print(f'{sys.argv[0]}: {f}: no such file', file=sys.stderr)
            return 1
        rewrite_file(f)
    return 0


if __name__ == '__main__':
    sys.exit(main())
