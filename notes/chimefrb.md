`pirate` is a new real-time FRB search, under development for the CHORD radio telescope,
that will supersede the "chimefrb" search code that we wrote a few years ago.

The chimefrb code is a collection of 11 repos, that are not in the pirate repo, but should
be available here (paths are relative to `pirate`):

```
  ../../extern/bonsai            -> dedispersion transform [dstn-sps]
  ../../extern/ch_frb_io         -> file and networking IO code [kms_chord]
  ../../extern/ch_frb_l1         -> top-level FRB search server for CHIME [dstn-master]
  ../../extern/ch_frb_rfi        -> helper functions for constructing RFI transform chains
  ../../extern/pyclops           -> python linkage (low-budget homegrown pybind11)
  ../../extern/rf_kernels        -> x86 compute kernels (not gpu kernels)
  ../../extern/rf_pipelines      -> high-level framework for organizing RFI flagging, plotting, etc. [dstn-master]
  ../../extern/simd_helpers      -> x86 inline helpers
  ../../extern/simpulse          -> python2 simulation code (vendored into `pirate` with modifications)
  ../../extern/sp_hdf5           -> higher-level interface for libhdf5 [kms_1_10]
  ../../extern/spshuff           -> supports a compressed format [apr_slow_pulsar]
```

## CHIMEFRB "porting" tasks

Some tasks will be "chimefrb-porting" tasks: "porting" some of the old code's functionality into pirate.
As a concrete example, our first chimefrb-porting task was writing a `pirate::AssembledChunk` class which
can read data files (in "msgpack" format) that were created using the old chimefrb code, so that we can
process old data files with pirate.

Our main use case is comparing chimefrb code to its equivalent (or moral equivalent) in the new code.
This has a lot of short-term value for development, but we may not want to keep it long-term.
For this reason, I'd like to avoid "polluting" the pirate design with refactoring that only makes
sense in order to accommodate chimefrb code.

Therefore, when doing chimefrb-porting tasks (e.g. the msgpack reader task mentioned above), please use the following rules:

  - Put new C++ code in namespace `pirate::chimefrb`, and in the following files:
    ```
      pirate/include/chimefrb/*.hpp
      pirate/src_lib/chimefrb/*.{cpp,cu}
      pirate/src_pybind11/pirate_pybind11_chimefrb.cpp
    ```

  - Put new python code in a subpackage `pirate_frb.chimefrb`, and import C++ code into this subpackage somewhere.

  - Avoid modifying existing pirate code in order to accommodate chimefrb code. Building parallel functionality 

  - Adhere to `pirate` code conventions, not chimefrb style conventions. Feel free to change details of the chimefrb
    code if it facilitates porting. For example, the chimefrb C++/python bindings were written using a homegrown library
    `pyclops`. When porting, it would make more sense to write pybind11 bindings instead (since pirate uses pybind11),
    rather than "port" pyclops into `pirate::chimefrb`. Preserving chimefrb code verbatim is NOT a goal -- feel free
    to make modifications if it facilitates porting, but don't modify the chimefrb code unnecessarily.
    In some cases, this may be a judgement call, so please ask/discuss if helpful.

  - We only intend to port a subset of the chimefrb functionality into pirate, so please keep it minimal, and "port"
    only chimefrb functionality which is needed to accomplish the task. In some cases, this may be a judgement call,
    so please ask/discuss if helpful.

  - In chimefrb, we often ended up with "reference" code which is either written in python or unoptimized c++,
    "production" avx2 code, and a unit test which verifies equivalence. In pirate, we often have something similar,
    but the details are different. For the reference code, either python or unoptimized C++ is possible, but python
    is usually preferred. For the "production" code, we might want a cuda kernel, an avx512 kernel,
    or just well-optimized non-simd C++ code, depending on context and how timings work out.
    Let me know if it isn't clear from context which of these possibilities you should be writing.

It will usually be obvious whether a task is a "chimefrb-porting" task or not (and I'll try to remember to indicate
explicitly), but if you're unsure, just ask.

## Appendix A: building the chimefrb code

The 11 repos above do not build out of the box on a modern system, and their `master`
branches do not form a working set. All of that is handled by:

```
misc/chimefrb/build_oldpipe.sh          # all 11 repos, in dependency order
misc/chimefrb/build_oldpipe.sh bonsai   # or just one
```

This takes about a minute from scratch. It never touches `../../extern`: each repo is
rsynced into `misc/chimefrb/oldpipe/src/`, patched there if needed, and installed into
`misc/chimefrb/oldpipe/prefix`. The whole `oldpipe/` directory is gitignored and
per-worktree, so it is always safe to delete and rebuild, and agents working in
parallel worktrees cannot collide.

To use the result:

```
. misc/chimefrb/oldpipe/env.sh
oldpython -c 'import rf_pipelines'    # 'oldpython' is the python 2.7 in the conda env
```

The build needs a conda env named `chimefrb`, which SHOULD ALREADY EXIST. It supplies
python 2.7 and the old C/C++ libraries (the sandbox cannot create it: miniforge3 is
mounted read-only and the conda channels are not on the egress allowlist). It was
created with:

```
conda create -n chimefrb --override-channels -c conda-forge \
  python=2.7 numpy 'cython<3' \
  'hdf5=1.10' h5py \
  fftw jsoncpp zeromq cppzmq 'msgpack-c<4' libpng yaml-cpp libcurl lz4-c \
  pkg-config \
  matplotlib pillow pyyaml pyzmq msgpack-python
```

If the build fails, `misc/chimefrb/README.md` has the details: which branch each repo
must be on and why, what the local patches are for, and the two known test failures.
Read it before changing anything -- several of the constraints look arbitrary and are
not (the `hdf5=1.10` pin in particular is load-bearing in both directions).

## Appendix B: spot checks against the chimefrb code

A "spot check" runs one piece of the old code and compares it against its pirate
equivalent. These are one-off correctness checks used during development, not part of
the pirate test suite.

```
misc/chimefrb/run_spot_tests.py             # all
misc/chimefrb/run_spot_tests.py -l          # list them
misc/chimefrb/run_spot_tests.py <name>      # just one
```

They are deliberately NOT dispatched from `pirate_frb/__main__.py`, and should not be:
they need `oldpipe/` built, which `python -m pirate_frb test` must never require.

### How one is put together

Each spot check is a directory under `misc/chimefrb/` holding exactly two files:

```
<name>/driver.cpp    the OLD side
<name>/test.py       the NEW side, and the comparison
```

The driver is a standalone program with one job:

```
driver <input.npy> <output.npy> [key=value ...]
```

It links only chimefrb libraries, prints nothing on success, and signals failure by
exit status. It knows nothing about pirate, about what it is being compared against, or
about what counts as agreement. `test.py` owns all of that: it generates the input, runs
the pirate side, and decides the tolerance.

`.npy` is the exchange format because numpy reads and writes it for free on the pirate
side, and `misc/chimefrb/npy.hpp` handles it on the chimefrb side. `misc/chimefrb/harness.py`
supplies `run_driver()`, which compiles the driver on demand and moves the arrays
across, and `Test.check_allclose()`, which reports the disagreement it actually
measured rather than just pass/fail.

Read `misc/chimefrb/dispersion_delay/` before writing a new one. It is short on purpose,
and it is a real comparison: bonsai and pirate use dispersion constants that differ by
4.8e-7, so it is an example of a spot check where exact agreement is the wrong
expectation.

### Adding a new spot check

1. Make a directory `misc/chimefrb/<name>/` and copy the two files from
   `dispersion_delay/` as a starting point.
2. In `driver.cpp`, include `"../npy.hpp"` plus whatever chimefrb headers you need,
   read one array, compute, write one array. Do not print on success.
3. In `test.py`, generate the input, call `harness.run_driver()`, compute the pirate
   answer, and compare with `Test.check_allclose()`.
4. Run it. `run_spot_tests.py` picks it up automatically; `test.py` also runs directly,
   which is easier when debugging.

Three rules worth following:

  - NOTHING BINARY GETS COMMITTED. Generate the input from a seed written down in
    `test.py`, and let the driver regenerate its output every run. Compiled drivers and
    scratch arrays go under `oldpipe/spot_tests/<name>/`, which is gitignored and is
    discarded whenever the old pipeline is rebuilt (correct, since a driver links
    against it).

  - STATE AN EXPLICIT TOLERANCE, AND SAY WHY IN THE SAME PLACE. These are rarely
    arbitrary. In `dispersion_delay` the tolerance IS a fact about the two codes, not a
    fudge factor. A tolerance with no stated reason cannot be reviewed later.

  - KEEP THE INPUT SMALL. Nothing is cached between runs, so a large array costs
    runtime every time and usually buys no extra coverage.

One caution when interpreting a result: the chimefrb unit tests seed their RNGs from OS
entropy and do not print the seed, and at least one of them (`rf_kernels
test-intensity-clipper`) fails a few percent of the time. A single green run of the old
code is weaker evidence than it looks. Re-run before believing a disagreement.
