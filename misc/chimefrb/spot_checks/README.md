# Spot checks: pirate vs the old CHIME pipeline

A spot check runs one piece of the old CHIME FRB search and compares it against its
pirate equivalent.  They need a built `oldpipe/`, which the build script one directory
up produces -- see `../README.md`, and `notes/chimefrb.md` for how they fit into the
port as a whole.

    misc/chimefrb/spot_checks/run_spot_tests.py             # all
    misc/chimefrb/spot_checks/run_spot_tests.py -l          # list them
    misc/chimefrb/spot_checks/run_spot_tests.py <name>      # just one

Each test is a directory holding exactly two files:

    <name>/driver.cpp   the OLD side
    <name>/test.py      the NEW side, and the comparison

`dispersion_delay/` is the worked example.  Read it before adding another; it is short
on purpose.

The driver is a standalone program, usually reading one array and writing one:

    driver <in_1.npy> ... <in_m.npy> <out_1.npy> ... <out_n.npy> [key=value ...]

linking only old libraries, silent on stdout, exit status its only signal.  It does not
know pirate exists, what it is compared against, or what the tolerance is.  `test.py`
owns all of that -- generating the input, running pirate, comparing, reporting -- and
`harness.py` gives it `run_driver()`, which compiles the driver on demand and does the
file exchange, plus `Test.check_allclose()`, which reports the disagreement it actually
measured rather than just pass/fail.

`.npy` is the wire format because numpy reads and writes it for free on the new side,
`npy.hpp` handles it in about 200 lines on the old side, and it is self-describing, so
an intermediate array can be archived and inspected when a comparison surprises you.

Two rules the layout depends on:

NOTHING BINARY IS COMMITTED.  A test generates its input from a seed written down in
`test.py`, and regenerates the old side's output every run.  So a spot test always
needs a built `oldpipe/`, and the whole comparison reproduces from source.  Compiled
drivers and scratch arrays live in `oldpipe/spot_tests/<name>/`, which is gitignored
and is discarded whenever the old pipeline is rebuilt -- correct, since a driver links
against it.

SPOT TESTS ARE NOT PART OF THE PIRATE TEST SUITE.  They are not dispatched from
`pirate_frb/__main__.py` and never will be: they need the old pipeline built, which
`python -m pirate_frb test` must never require.  `run_spot_tests.py` is the only entry
point, and each `test.py` also runs directly, which is the easier way to debug one.

State an explicit tolerance in every test, and say WHY in the same place.  These are
rarely arbitrary -- in `dispersion_delay` it is the 4.8e-7 offset between the two
codes' dispersion constants, which is a fact about the two codes and not a fudge
factor.  A tolerance with no stated reason cannot be reviewed later.

Note also that the old code's own tests seed from OS entropy without printing the seed
(see the known issue above), so a green run of the old side is weaker evidence than it
looks.  Re-run before believing a disagreement.
