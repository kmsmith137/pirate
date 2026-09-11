# Building and running the old CHIME pipeline

Spot tests compare pirate against the CHIME FRB search that preceded it -- the 11
repos in `~/ch/extern` (see `notes/chimefrb.md`).  This directory holds the build
script, the source patches it needs, and the spot tests themselves.

## Layout

    build_oldpipe.sh    builds ~/ch/extern/* into ./oldpipe/prefix
    patches/            source fixes, applied to the working copy only
    npy.hpp             .npy read/write for a driver
    harness.py          shared helpers for a test.py
    run_spot_tests.py   the runner
    <test>/             one spot test: driver.cpp (old side) + test.py (new side)
    oldpipe/            the build, plus compiled drivers -- gitignored

`~/ch/extern` holds pristine reference clones and is never modified.  `build_oldpipe.sh`
rsyncs each repo into `oldpipe/src/`, applies the patches, and builds there, so a working
copy can always be thrown away and rebuilt.

`oldpipe/` sits inside this directory, so each worktree gets its own -- agents working in
parallel in different worktrees cannot trip over each other's build.  It is about 100 MB
and takes a minute to make, and `.gitignore` covers it.

## Prerequisites

A conda env named `chimefrb`, holding python 2.7 and the old C/C++ libraries.  The
sandbox cannot create it -- `/home/kmsmith/miniforge3` is mounted read-only and the
conda channels are not on the egress allowlist -- so it is created on the host:

    conda create -n chimefrb --override-channels -c conda-forge \
      python=2.7 numpy 'cython<3' \
      'hdf5=1.10' h5py \
      fftw jsoncpp zeromq cppzmq 'msgpack-c<4' libpng yaml-cpp libcurl lz4-c \
      pkg-config \
      matplotlib pillow pyyaml pyzmq msgpack-python

Set `CHIMEFRB_ENV` if it lives somewhere other than
`/home/kmsmith/miniforge3/envs/chimefrb`.

The `hdf5=1.10` pin matters in both directions.  conda-forge's hdf5 1.8 builds predate
its move to the C++11 string ABI -- they export `std::basic_string` where jsoncpp and
yaml-cpp export `std::__cxx11::basic_string`, and no binary here can link both.  And
`sp_hdf5` only reaches a 1.10+ C++ API via its patch below.  Taking hdf5 from the system
instead is not a way out: that copy is new-ABI, but it needs the system libcurl, which
shares the env's `libcurl.so.4` SONAME while carrying a different symbol-version node
(`CURL_OPENSSL_4`), so the two collide at link and load time.

Compilers come from the system (gcc 13), not the env: the env deliberately ships no
compiler, so only `python` and `cython` are taken from it.

## Which branch each extern repo must be on

The old code's `master` branches do not form a working set: the slow-pulsar work lives
on side branches, and `rf_pipelines` will not compile against `bonsai` or `ch_frb_io`
on `master`.  The combination that builds:

    simd_helpers    master
    sp_hdf5         kms_1_10         cb7fa99  2020-01-17
    rf_kernels      master
    simpulse        master
    spshuff         apr_slow_pulsar  a1094ab  2022-12-01
    pyclops         master
    bonsai          dstn-sps         6c5b283  2020-11-11
    ch_frb_io       kms_chord        210d57c  (dstn-master + 4 fixes)
    rf_pipelines    dstn-master      8a483b3  2026-03-23
    ch_frb_rfi      master
    ch_frb_l1       dstn-master      0306859  2026-06-19

Each of these is a local maximum: no other branch in the same repo strictly contains
it.  Four of them are NOT the obvious choice, so the reasons:

`ch_frb_io` on `kms_chord`, which is `dstn-master` plus the four fixes described
under Patches below.  `rf_pipelines` and `ch_frb_l1` on `dstn-master` rather than
`kms_sps`.
`dstn-master` strictly contains `kms_sps` in all three (their merge base IS the
`kms_sps` tip, and `dstn-master` merged it forward), and adds 7, 23 and 53 commits of
maintenance running into 2026.  `kms_sps` is a 2022 snapshot of the same line.

`bonsai` on `dstn-sps` because `ch_frb_l1`'s `l1-rpc.cpp` calls
`bonsai::dedisperser::get_weights_and_variances()`, which exists on only two branches,
both at commit 6c5b283 (`dstn-sps` and `apr_slow_pulsar`).

`sp_hdf5` on `kms_1_10` because `master`'s C++ wrappers only compile against HDF5 1.8
(see the `hdf5=1.10` discussion above).  `kms_1_10` makes them version-conditional --
`_attr_holder` / `_group_holder` typedefs switching on `H5_VERS_MINOR` -- so both 1.8
and 1.10 work.  Do NOT use `kms_1_12`, which is a superset but assumes the 1.12
five-argument `H5Oget_info_by_name()` and fails against the env's 1.10.5.

`spshuff` has NO `spshuff.hpp` on `master`, so master cannot work.  Four branches carry
a byte-identical copy of it (sha256 4738986...) -- `apr_slow_pulsar`, `ci-actions`,
`setup-cxx-args`, `fix-downsampling` -- and the encoding constants (`edges5`,
`dequant5`, `codes`, `lens`) match on every branch that has the header, so the choice
does not affect the on-wire format.  Since that header is the only thing this build
consumes, the branches are interchangeable here.

Where they are not interchangeable is spshuff's own python package, which we do not
use (it is py3-only, and this env is python 2.7).  `apr_slow_pulsar` moved the pybind
`encode()` entry point off `quantize_naive5_reference()` onto the faster
`quantize_naive5()`, and added downsampling to `spshuff/l1_io.py`; `ci-actions` instead
carries three 2022-03-03 commits that were the version running at site.  Neither
contains the other.  If a spot test ever drives spshuff through python rather than
through the header, that difference has to be pinned down.

Note `ch_frb_io.hpp` hard-codes `version = 5` with a comment saying it must agree with
spshuff.

## Building

    misc/chimefrb/build_oldpipe.sh              # all repos, in dependency order
    misc/chimefrb/build_oldpipe.sh bonsai       # or just one

Takes about a minute from scratch.  Then:

    . misc/chimefrb/oldpipe/env.sh
    oldpython -c 'import rf_pipelines'

`oldpipe/BUILD_MANIFEST.txt` records the extern revisions, compiler and flags that
produced a build; `oldpipe/log/<repo>.log` has the output.  `oldpipe/driver_flags.json`
carries the compiler flags a spot-test driver needs.

The build is pinned to `-march=haswell`, not `-march=native`.  Golden files have to
reproduce on a machine other than the one that made them, and CHIME ran on Haswell-era
Xeons anyway.  Combined with `-ffast-math`, expect agreement with pirate at the 1e-5
level for float32, not bit-for-bit -- assert an explicit tolerance.

## Patches

Each file in `patches/` is applied to one repo by name (`<repo>-NN-<topic>.patch`).

  bonsai-01-cstdint             add <cstdint>  -- older libstdc++ pulled this in
  ch_frb_l1-01-cassert          add <cassert>     transitively; gcc 13 does not

Both are one-line additions of a header.  Anything larger should become a commit on
the repo's own branch instead, the way the four `ch_frb_io` fixes did -- a
write_msgpack_file() buffer overflow, a bad offset in test_packet_offsets(), a
bitshuffle fallback that could not detect a missing plugin, and a test that violated
downsample()'s own precondition -- all now on `kms_chord`.

Add a patch only when the alternative is worse.  A spot test wants the old code's
numerics unchanged, so a patch that alters anything but a build error needs saying so
out loud, in the test's manifest.

## Known issue: rf_kernels test-intensity-clipper fails intermittently

`test-intensity-clipper` fails about 2 runs in 25.  A captured failure:

    assertion 'w_fast[i] <= w_ref2[i]' failed (test-intensity-clipper.cpp:442)
    test_intensity_clipper(nfreq=848, nt_chunk=1792, istride=2874, wstride=2321,
                           axis=AXIS_NONE, Df=16, Dt=16, niter=1, sigma=1.04236,
                           iter_sigma=1.65865, two_pass=1)
        failed at ifreq=0, it=480
        at (ifreq,it) = (0,480): i_in=0.914107, w_fast=0.96884, w_ref1=0, w_ref2=0

Both reference implementations mask the sample (weight 0) and the fast AVX2 kernel
keeps it at weight 0.969, so this is a real disagreement between the fast and
reference clippers, not a tolerance that is slightly too tight.  Root cause not
investigated; it predates anything here and is unrelated to the CHORD work.

What makes it awkward is that the test seeds from OS entropy and does not print the
seed:

    std::random_device rd;
    std::mt19937 rng(rd());

so an individual failure cannot be replayed.  Every rf_kernels and bonsai test seeds
this way.

For spot tests the lesson is procedural: do not treat a single green run of the old
code as authoritative, and re-run before believing a disagreement.

## Optional: bitshuffle

One thing is not built, because it is not in these repos: the bitshuffle HDF5 filter
plugin (`kiyo-masui/bitshuffle`, filter id 32008).  `ch_frb_io` loads it dynamically at
runtime through `$HDF5_PLUGIN_PATH` -- the vendored `ch_frb_io/bitshuffle/` is only the
core codec it uses for msgpack, not the HDF5 filter.

Without it, `ch_frb_io test-intensity-hdf5-file` aborts with `required filter 32008 is
not registered`, and bitshuffle-compressed intensity files can be neither read nor
written.  CHIME pathfinder acquisition data is generally bitshuffle-compressed, so a
spot test that wants to read real acquisition data will need it; nothing else does.
Everything else HDF5 works -- `bonsai-mkweight` writes a `bonsai_config.hdf5`, bonsai
writes trigger files, and h5py reads both, all against the same libhdf5 1.10.5.

To add it, clone `kiyo-masui/bitshuffle` into `~/ch/extern` and build its `--h5plugin`
target, then set `HDF5_PLUGIN_PATH`.

## Spot tests

    misc/chimefrb/run_spot_tests.py             # all
    misc/chimefrb/run_spot_tests.py -l          # list them
    misc/chimefrb/run_spot_tests.py <name>      # just one

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
