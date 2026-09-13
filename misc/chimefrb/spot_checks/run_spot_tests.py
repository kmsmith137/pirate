#!/usr/bin/env python3
"""Run the chimefrb spot tests: pirate vs the old CHIME pipeline.

    misc/chimefrb/spot_checks/run_spot_tests.py             # all tests
    misc/chimefrb/spot_checks/run_spot_tests.py -l          # list them
    misc/chimefrb/spot_checks/run_spot_tests.py <name> ...  # just these

A test is any subdirectory of misc/chimefrb/spot_checks containing a test.py, which this
runs as a separate process.  Exit status 0 means the comparison held.  Each test.py is also
runnable on its own, which is the easier way to debug one.

These are deliberately NOT dispatched from 'python -m pirate_frb test'.  They need a
built copy of the old CHIME pipeline (misc/chimefrb/build_oldpipe.sh), which the
normal test suite must never require.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # misc/chimefrb/spot_checks
OLDPIPE = os.path.join(os.path.dirname(HERE), "oldpipe")   # the build lives beside us, not under us
SKIP = {"__pycache__"}


def discover():
    names = []
    for e in sorted(os.listdir(HERE)):
        if e in SKIP or e.startswith("."):
            continue
        if os.path.exists(os.path.join(HERE, e, "test.py")):
            names.append(e)
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tests", nargs="*", help="tests to run (default: all)")
    p.add_argument("-l", "--list", action="store_true", help="list tests and exit")
    args = p.parse_args()

    available = discover()

    if args.list:
        for n in available:
            print(n)
        return 0

    if not available:
        print("no spot tests found in %s" % HERE, file=sys.stderr)
        return 1

    todo = args.tests or available
    for n in todo:
        if n not in available:
            print("no such spot test: %s (have: %s)" % (n, ", ".join(available)), file=sys.stderr)
            return 1

    if not os.path.exists(os.path.join(OLDPIPE, "driver_flags.json")):
        print("no old-pipeline build found.\n"
              "Run misc/chimefrb/build_oldpipe.sh first (about a minute).", file=sys.stderr)
        return 1

    failed = []
    for n in todo:
        rc = subprocess.run([sys.executable, os.path.join(HERE, n, "test.py")]).returncode
        if rc != 0:
            failed.append(n)
        print()

    print("%d/%d spot tests passed" % (len(todo) - len(failed), len(todo)))
    if failed:
        print("failed: %s" % ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
