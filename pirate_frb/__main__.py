import os
import re
import sys
import time
import itertools
import shlex
import random
import textwrap
import argparse
import threading
import traceback

# BLAS THREAD COUNT, for 'pirate_frb test' only, and set HERE because OpenBLAS reads these
# once, when numpy is first imported -- which the next line does, transitively. The unit tests
# do thousands of SMALL linear-algebra calls (varmap's run_all() alone does ~280 SVDs), and an
# unbounded thread pool spends more time synchronizing than computing: measured on this
# 64-core host, varmap's run_all() takes 10.5 s at the default and 6.5 s at 4 threads. The
# PRODUCTION paths want the full pool -- varmap at CHORD scale factorizes matrices of billions
# of entries -- so this is scoped to the 'test' command, and it is a setdefault, so
# 'OMP_NUM_THREADS=64 pirate_frb test ...' still gets the old behaviour.
if (len(sys.argv) > 1) and (sys.argv[1] == 'test'):
    for _blas_var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_blas_var, '4')

import argcomplete
import ksgpu

from . import pirate_pybind11
from . import casm
from . import chime
from . import kernels
from . import loose_ends
from . import core
from . import tests
from . import varmap
from .fast_varmap import compute_detrender_free_varcoarse


from . import (
    DedispersionConfig,
    DedispersionPlan,
    GpuDedisperser,
)

from .Hardware import Hardware
from .Hwtest import Hwtest
from .HwtestSender import HwtestSender
from .yaml_utils import indent_dedispersion_plan_comments, align_inline_comments
from .utils import atomic_print, print_separator


############################################   seeding   ###########################################


# Master seed for every RNG the tests draw from. Override with 'pirate_frb test --seed N',
# or draw one from OS entropy with 'pirate_frb test -r'.
DEFAULT_SEED = 137


def seed_rngs(seed):
    """Seeds every RNG the tests draw from, from one master seed.

    THREE STREAMS, ONE NUMBER:

      - ksgpu::default_rng(), the C++ side. Note this covers
        DedispersionConfig::make_random(), which draws through ksgpu::rand_int(), and
        avx2_simulate_4bit_noise(), whose per-thread xoshiro is seeded from it on first use.
      - numpy's global RandomState (np.random.uniform() and friends).
      - python's stdlib 'random', which tests/test_network.py and tests/test_server.py use.

    The three are seeded from independent children of one SeedSequence rather than from the
    master directly, so the streams are uncorrelated but the whole run replays from a single
    pasteable integer.

    NOTHING DRAWS FROM A FOURTH, UNSEEDED STREAM, and that is a rule rather than an accident.
    A numpy Generator (np.random.default_rng) built with no argument seeds itself from OS
    entropy and is therefore outside all of this; the suites that want one -- varmap and the
    three detrending packages -- derive its seed from the global RandomState above, so
    successive calls still differ while the run as a whole replays. See varmap/tests.py's
    _rng() and detrending_testutils.default_rng(), which the three detrending suites share.

    SEEDED ONCE PER PROCESS, NOT PER TEST, and that is the point: iteration i of the 'test
    -n' loop draws different values from iteration j (so a long run explores the parameter
    space), while rerunning the same command replays the same sequence (so a failure at
    iteration 700 is reproducible). Seeding per test would give up the first property, and
    not seeding at all gives up the second.

    WHAT IT DOES NOT BUY is anything drawn on a thread other than this one:
    ksgpu::seed_default_rng() reseeds only the CALLING thread, so any C++ thread spawned
    later self-seeds from std::random_device; and in python a seeded RNG fixes the sequence
    of values drawn but not which thread draws which. That affects --net and --serv only, and
    test() prints exactly which parts of those two replay and which do not.
    """

    import numpy as np

    s_np, s_ks, s_py = np.random.SeedSequence(seed).spawn(3)
    np.random.seed(s_np.generate_state(4))
    ksgpu.seed_default_rng(int(s_ks.generate_state(1, dtype=np.uint32)[0]))
    random.seed(int(s_py.generate_state(1, dtype=np.uint32)[0]))


def draw_random_seed():
    """A master seed from OS entropy, for 'pirate_frb test -r'. Printed by the caller, since
    a randomized run that does not say what it drew cannot be replayed."""

    return int.from_bytes(os.urandom(4), 'little')


#########################################   test command  ##########################################


def parse_test(subparsers):
    help_text = "Run unit tests (use flags to select specific tests)"
    parser = subparsers.add_parser("test", help=help_text, description=help_text)
    parser.set_defaults(func=test)
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for tests (default 0)")
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument('-n', '--niter', type=int, default=100,
                            help="Number of unit test iterations (default 100)")
    stop_group.add_argument('-t', '--time', type=float, metavar='SECONDS',
                            help="Run for at least SECONDS instead of a fixed iteration count. The check is at the BOTTOM of the loop, so at least one FULL iteration always runs and the elapsed time will overshoot by up to one iteration. Note a -t run is not directly replayable, since the iteration count depends on machine speed; the count to replay with -n is printed at the end.")

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('-s', '--seed', type=int, default=DEFAULT_SEED, metavar='N',
                            help=f"Master RNG seed (default {DEFAULT_SEED}). Seeds the ksgpu (C++), numpy and stdlib-random generators. Replaying a run needs the same seed AND the same test flags AND the same -n, since the streams are shared and consumed in test order.")
    seed_group.add_argument('-r', '--randomize-seed', action='store_true',
                            help="Draw the master RNG seed from OS entropy instead of using the default. The seed is printed, so a failing run can be replayed with --seed.")

    parser.add_argument('--rt', action='store_true', help='Runs ReferenceTree and ReferenceLagbuf tests')
    parser.add_argument('--pfwr', action='store_true', help='Runs PfWeightReaderMicrokernel.test_random()')
    parser.add_argument('--pfom', action='store_true', help='Runs PfOutputMicrokernel.test_random()')
    parser.add_argument('--pfsq', action='store_true', help='Runs the PfSquare tests (GpuPfSquare + ReferencePfSquare)')
    parser.add_argument('--gldk', action='store_true', help='Runs GpuLaggedDownsamplingKernel.test_random()')
    parser.add_argument('--gddk', action='store_true', help='Runs GpuDedispersionKernel.test_random()')
    parser.add_argument('--gpfk', action='store_true', help='Runs GpuPeakFindingKernel.test_random()')
    parser.add_argument('--grck', action='store_true', help='Runs GpuRingbufCopyKernel.test_random()')
    parser.add_argument('--gtgk', action='store_true', help='Runs GpuTreeGriddingKernel.test_random()')
    parser.add_argument('--gdqk', action='store_true', help='Runs GpuDequantizationKernel.test_random()')
    parser.add_argument('--cdd2', action='store_true', help='Runs CoalescedDdKernel2.test_random()')
    parser.add_argument('--sbdd', action='store_true', help='Runs GpuSbDedispersionKernel.test_random()')
    parser.add_argument('--casm', action='store_true', help='Runs some casm tests')
    parser.add_argument('--zomb', action='store_true', help='Runs "zombie" tests (code that I wrote during protoyping that may never get used)')
    parser.add_argument('--dd', action='store_true', help='Runs GpuDedisperser.test_random()')
    parser.add_argument('--varmap', action='store_true', help="pirate_frb.varmap. Two halves, both run by this flag: everything checkable WITHOUT a dedisperser (the VarianceMap class, the covering-LP and basis machinery, and the analytic map of detrender_free.py against a hand-written oracle), and the brute-force sweep, which pushes a one-hot through the REAL dedisperser once per input channel and checks the analytic map against what comes out. Needs a DedispersionPlan and a GPU for the second half.")
    parser.add_argument('--chime', action='store_true', help='Runs test_chime_frb_{beamform,upchan}()')
    parser.add_argument('--net', action='store_true', help='Runs network/allocator tests (AssembledFrameAllocator, etc.)')
    parser.add_argument('--serv', action='store_true', help='Runs end-to-end FakeXEngine -> FrbServer -> GpuDedisperser -> FrbGrouper test')
    parser.add_argument('--sim', action='store_true', help='Runs avx2_simulate_4bit_noise() distribution test + AssembledFrame pulse-injection and pulse-invariants tests')
    parser.add_argument('--amax', action='store_true', help='Runs DedispersionPlan.decode_argmax() tests (black-box probe arrays)')
    parser.add_argument('--sb', action='store_true', help='Runs frequency-subband tests (C++/python parity of the two FrequencySubbands implementations, and the per-tree subband-set property)')
    parser.add_argument('--aout', action='store_true', help='Runs the serialized-output test (atomic_print/AtomicPrint, C++ and python threads)')
    parser.add_argument('--util', action='store_true', help='Runs test_utils() (integer/bit helpers in inlines.hpp, plus bit_reverse_slow())')
    parser.add_argument('--dt1d', action='store_true', help='Runs pirate_frb.detrending_1d tests (pure-numpy 1-d detrender)')
    parser.add_argument('--dt1k', action='store_true', help='Runs pirate_frb.detrending_1d_kalman tests (pure-numpy fixed-lag Kalman detrender)')
    parser.add_argument('--dts', action='store_true', help='Runs pirate_frb.detrending_spline tests (pure-numpy regularized spline detrender)')
    parser.add_argument('--dt2g', action='store_true', help='Runs pirate_frb.detrending_spline.tests.test_gpu_kernel() (Detrender2d GPU kernel vs the numpy reference)')


def rrange(registry_class):
    """Repeat-range helper for iterating over a kernel registry.

    This function is used to iterate over a kernel registry, for an appropriate number
    of iterations, so that every kernel is tested a few times. See usage in test() below.
    """

    n = registry_class.registry_size()

    if n == 0:
        atomic_print(f'{registry_class.__name__}: no kernels were registered, associated unit test will be skipped.')
        return
    
    for i in range((n+9)//10):
        yield i


def test(args):
    test_flags = [ 'rt', 'pfwr', 'pfom', 'pfsq', 'gldk', 'gddk', 'gpfk', 'grck', 'gtgk', 'gdqk', 'cdd2', 'sbdd', 'casm', 'chime', 'zomb', 'dd', 'varmap', 'net', 'serv', 'sim', 'amax', 'sb', 'aout', 'util', 'dt1d', 'dt1k', 'dts', 'dt2g' ]
    run_all_tests = not any(getattr(args,x) for x in test_flags)

    seed = draw_random_seed() if args.randomize_seed else args.seed
    seed_rngs(seed)
    atomic_print(f'RNG seed {seed} (replay with: --seed {seed}, the same test flags, and the'
                 f' same iteration count)')

    if run_all_tests or args.net or args.serv:
        # Said out loud rather than left to be rediscovered, and split into what replays and
        # what does not, because "not reproducible" is too blunt to act on -- a --net failure
        # IS worth rerunning at the same seed, since the config and the frame data come back.
        #
        # What replays: the config and every drawn parameter (python's global RandomState,
        # stdlib random, and ksgpu::default_rng() are all pinned on the main thread), and
        # --net's frame data -- randomize(normalize=False, gaussian=False) at
        # test_network.py:495,760 runs on the main thread and takes AssembledFrame::randomize's
        # seeded mt19937 branch, not avx2_simulate_4bit_noise().
        #
        # What does not, in rough order of how much it matters:
        #   - The SCRIPT --net issues. _maybe_issue_write() reads live rb_start/rb_processed/
        #     rb_streamed and can return without drawing, so how many values the seeded python
        #     RNG is asked for depends on server timing, and the stream diverges from turn one.
        #     Pre-drawing the script would fix it.
        #   - --serv's frame data, from FakeXEngine's two randomizer threads
        #     (test_server.py:489). ksgpu::seed_default_rng() reseeds only the CALLING thread,
        #     so a thread spawned later self-seeds from std::random_device.
        #   - The short-read pattern in Socket::_misbehave_maxbytes() (network_utils.cpp:611),
        #     drawn on the reader thread, same reason.
        atomic_print('NOTE: --net and --serv replay only PARTLY from the seed. The config, all'
                     ' drawn parameters, and --net\'s frame data do replay; the sequence of'
                     ' operations --net issues does not (it depends on live server state), nor'
                     ' does anything drawn on a spawned thread (--serv\'s frame data, the'
                     ' short-read pattern). Every other test replays in full.')

    ksgpu.set_cuda_device(args.gpu)
    from . import utils   # local import (utils pulls in heavier deps)

    t_start = time.time()

    for i in itertools.count():
        if args.time is not None:
            atomic_print(f'\nIteration {i+1} ({time.time()-t_start:.0f} of {args.time:g} s)'
                         f'\n\n')
        else:
            atomic_print(f'\nIteration {i+1}/{args.niter}\n\n')
        
        if run_all_tests or args.dt1d:
            from .detrending_1d import tests as dt1d_tests
            dt1d_tests.run_all()

        if run_all_tests or args.dt1k:
            from .detrending_1d_kalman import tests as dt1k_tests
            dt1k_tests.run_all(iteration=i)

        if run_all_tests or args.dts:
            from .detrending_spline import tests as dts_tests
            dts_tests.run_all(iteration=i)

        if run_all_tests or args.dt2g:
            from .detrending_spline import tests as dts_tests
            print('  detrending_spline: Detrender2d GPU kernel vs the numpy reference')
            dts_tests.test_gpu_kernel(None)

        if run_all_tests or args.rt:
            kernels.ReferenceLagbuf.test_random()
            kernels.ReferenceTree.test_basics()
            kernels.ReferenceTree.test_subbands()
        
        if run_all_tests or args.pfwr:
            for _ in rrange(kernels.PfWeightReaderMicrokernel):
                kernels.PfWeightReaderMicrokernel.test_random()
        
        if run_all_tests or args.pfom:
            for _ in rrange(kernels.PfOutputMicrokernel):
                kernels.PfOutputMicrokernel.test_random()
        
        if run_all_tests or args.pfsq:
            # test_vs_peak_finder() is the only link between the peak-finding kernels and
            # the squaring kernels, which --gpfk and GpuPfSquare.test_random() each cover only
            # one side of. What it uniquely protects is the STREAMING comparison -- chunk
            # boundaries, tpad history, batch ordering, and (dpf, m) row order. (The
            # peak-finder's h_p coefficients are also reached from the other side by
            # PfVarianceConvolver.test_kernels_match_reference(), one impulse at a time.)
            kernels.ReferencePfSquare.test_vs_peak_finder()
            kernels.GpuPfSquare.test_random()
        
        if run_all_tests or args.gldk:
            kernels.GpuLaggedDownsamplingKernel.test_random()
        
        if run_all_tests or args.gddk:
            for _ in rrange(kernels.GpuDedispersionKernel):
                kernels.GpuDedispersionKernel.test_random()
        
        if run_all_tests or args.gpfk:
            for _ in rrange(kernels.GpuPeakFindingKernel):
                kernels.GpuPeakFindingKernel.test_random()
        
        if run_all_tests or args.grck:
            kernels.GpuRingbufCopyKernel.test_random()
        
        if run_all_tests or args.gtgk:
            kernels.GpuTreeGriddingKernel.test_random()
        
        if run_all_tests or args.gdqk:
            kernels.GpuDequantizationKernel.test_random()

        if run_all_tests or args.sim:
            utils.test_avx2_simulate_4bit_noise()
            tests.test_pulse_injection()
            tests.test_pulse_invariants()

        if run_all_tests or args.cdd2:
            for _ in rrange(kernels.CoalescedDdKernel2):
                kernels.CoalescedDdKernel2.test_random()

        if run_all_tests or args.sbdd:
            for _ in rrange(kernels.GpuSbDedispersionKernel):
                kernels.GpuSbDedispersionKernel.test_random()
        
        if run_all_tests or args.casm:
            atomic_print("\n")
            if i == 0:
                # This test is slower than the others, but I don't think we need it more than once.
                casm.CasmReferenceBeamformer.test_interpolative_beamforming()
            
            casm.CasmBeamformer.test_microkernels()
            casm.CasmReferenceBeamformer.test_cuda_python_equivalence(linkage='pybind11')
            
        if run_all_tests or args.chime:
            # test_chime_frb_beamform()'s CPU reference is a brute-force 512-point DFT per
            # (time, freq, pol, ew), which is ~1.7 s on an average draw and the most
            # expensive thing under --chime by a wide margin. Every tenth iteration keeps it
            # randomized (its shape is drawn) without paying for it every time.
            if (i % 10) == 0:
                chime.test_chime_frb_beamform()
            chime.test_chime_frb_upchan()

        if run_all_tests or args.zomb:
            loose_ends.test_avx2_m64_outbuf()
            loose_ends.test_cpu_downsampler()
            loose_ends.test_gpu_downsample()
            loose_ends.test_gpu_transpose()
            loose_ends.test_gpu_reduce2()
            
        if run_all_tests or args.dd:
            if i == 0:
                # Catches errors in DedispersionConfig::make_random() or validate().
                tests.test_max_width_monotone()

            # BOTH OF THESE ARE RANDOMIZED, so they run every iteration rather than once:
            # pinned to i == 0, a 1000-iteration overnight run saw exactly the same draws as
            # a 5-iteration smoke test. Ten configs an iteration reaches 500 by '-n 50' and
            # keeps going.
            #
            # The loop is really two checks. make_random() re-derives validate()'s rules in
            # three places, so "make_random() never emits a config validate() rejects" is
            # the one with content; config.test() adds the frequency_to_index round trip on
            # top of it. gpu_valid alternates because the True path has a key chain to get
            # right (DedispersionConfig.cpp) and the False path is the only one this loop
            # ever exercised.
            tests.test_random_args_flags()
            for j in range(10):
                c = DedispersionConfig.make_random(max_toplevel_rank=8, max_early_triggers=4,
                                                   gpu_valid=bool(j % 2))
                c.test()

            for _ in rrange(kernels.CoalescedDdKernel2):
                GpuDedisperser.test_random()
        
        if run_all_tests or args.varmap:
            # run_tests() owns the cadences -- which group runs once per invocation, which
            # every iteration, which every tenth. That is a property of the tests, so it
            # lives with them rather than here.
            from .varmap import tests as varmap_tests
            varmap_tests.run_tests(i)

        if run_all_tests or args.amax:
            tests.test_decode_argmax()

        if run_all_tests or args.sb:
            # Each of these has a deterministic half that says the same thing every time --
            # the exhaustive pf_rank <= 3 sweep, and the shipped-config re-parses -- and a
            # randomized half. Run the deterministic halves once and the randomized halves
            # every iteration; see each test's docstring.
            tests.test_frequency_subbands_parity(sweep_low_ranks=(i == 0))
            tests.test_subband_property(shipped=(i == 0))

        if run_all_tests or args.aout:
            # NOT deterministic: it is a concurrency test, and one call samples one thread
            # schedule. It draws its own thread counts, line counts and line length, and
            # costs 5-20 ms, so it runs every iteration.
            tests.test_atomic_out()

        if run_all_tests or args.util:
            # Integer/bit helpers: one call exhausts the interesting inputs, so once
            # is enough (see notes/unit_tests.md, "exhaust the parameter space").
            if i == 0:
                utils.test_utils()

        if run_all_tests or args.net:
            # Every one of its seven tests draws nfreq, time_samples_per_chunk, the beam
            # ids and the consumer count, so a multi-iteration run covers more than a single
            # one; it is host memory only and costs ~50 ms.
            tests.test_assembled_frame_allocator()
            # test_slow_subscriber() DOES draw its parameters (NetworkTester), so pinning it
            # to i == 0 meant a 1000-iteration run tested one draw. It is ~1 s, which is
            # most of a --net iteration, so every tenth rather than every one.
            if (i % 10) == 0:
                tests.test_slow_subscriber()
            tests.test_assembled_frame_asdf()
            tests.test_network()

        if run_all_tests or args.serv:
            tests.test_server()

        # AT THE BOTTOM OF THE LOOP, so that '-t' always runs at least one FULL iteration.
        # Two things follow, and both are deliberate: 'test -t 1' is a smoke test rather than
        # a no-op, and the elapsed time overshoots the budget by up to one iteration -- which
        # for a slow flag combination can be a lot, so -t bounds the START of the last
        # iteration, not the end of the run.
        if args.time is not None:
            if (time.time() - t_start) >= args.time:
                break
        elif (i + 1) >= args.niter:
            break

    if args.time is not None:
        # A -t run cannot be replayed with -t: the iteration count it reached depends on how
        # fast this machine is. Print the count, so that the seed printed above is actually
        # usable.
        atomic_print(f'\nRan {i+1} iterations in {time.time()-t_start:.1f} s.'
                     f' Replay with: --seed {seed} -n {i+1}\n')


######################################   dev test_simpulse command  #####################################


def parse_test_simpulse(subparsers):
    help_text = "Run simpulse tests (pulse-upsampling self-consistency) and write example plots to cwd"
    parser = subparsers.add_parser("test_simpulse", help=help_text, description=help_text)
    parser.set_defaults(func=test_simpulse)
    parser.add_argument('-n', '--niter', type=int, default=100, help="Number of upsampling-test iterations (default 100)")


def test_simpulse(args):
    # Import lazily so matplotlib (needed by plot_pulses) is only required for this command.
    from .simpulse import test_pulse_upsampling, plot_pulses
    test_pulse_upsampling.run_tests(args.niter)
    plot_pulses.make_plots()


####################################   varmap subcommands  ##########################################


def parse_varmap(subparsers):
    """The 'varmap' group: utils for working with asdf-serialized variance maps.

    'bf' and 'df' COMPUTE a map; 'mc' CHECKS one that already exists.

    The two that compute are not two speeds of one algorithm -- they produce different
    objects. 'bf' sweeps the real dedisperser and returns A_true itself, carrying no
    domination certificate until get_distance() scores it; 'df' is analytic and
    SVD-truncated, and its map DOMINATES A_true by construction (is_admissible=True). Only
    'bf' can take a detrender.
    """
    help_text = "Subcommand for working with asdf-serialized variance maps (see varmap --help)"
    sub = _add_group(subparsers, "varmap", help_text)
    parse_varmap_bf(sub)
    parse_varmap_df(sub)
    parse_varmap_mc(sub)


def _add_varmap_common_args(parser):
    """The three arguments both subcommands take, so the two cannot drift apart.

    NOTE '-g/--gpu' is NOT here -- it is bf-only. See parse_varmap_df().
    """
    parser.add_argument('config_file', help="Path to dedispersion YAML config file")
    parser.add_argument('-o', '--output', required=True, metavar='PATH',
                        help="Output .asdf file (required)")
    parser.add_argument('-L', '--coarse-grain', type=int, default=None, metavar='L',
                        help="Coarse-grain at rank L rather than writing the dense fine map."
                             " This is what makes a large config reachable: at CHORD tree 0"
                             " the dense map is 1.2 TiB and the coarse map is 344 GiB at L=4."
                             " Legal range is R <= L <= r per tree. Omit to write the dense"
                             " fine map, which is only viable at subscale.")


########################################   varmap bf command  #######################################


def parse_varmap_bf(subparsers):
    help_text = "Compute the variance map A by brute force (sweep), and write it to an ASDF file"
    parser = subparsers.add_parser("bf", help=help_text, description=help_text)
    parser.set_defaults(func=varmap_bf)
    _add_varmap_common_args(parser)
    parser.add_argument('detrender_file', nargs='?', default=None,
                        help="Path to Detrender2dParams YAML file (omit with --no-detrender)")
    parser.add_argument('--no-detrender', action='store_true',
                        help="Run with no Detrender2d; 'detrender_file' must then be omitted")
    parser.add_argument('--cpu', action='store_true',
                        help="Force the CPU sweep (default: GPU)")
    parser.add_argument('--channels', default=None, metavar='SPEC',
                        help="Sweep only these input channels, as a comma-separated list of"
                             " indices or LO:HI[:STEP] slices. The result is a PARTIAL map"
                             " whose other columns are zero, written with no y_true so that"
                             " nothing downstream can score it. For timing a sweep before"
                             " committing to the whole thing.")
    parser.add_argument('--scratch-dir', default=None, metavar='DIR',
                        help="Back the arrays of matrix size with memmaps under DIR instead"
                             " of RAM. The fallback for a config whose accumulator does not"
                             " fit; DIR must survive until the output file is written.")
    parser.add_argument('--no-guard-chunk', action='store_true',
                        help="Skip the per-pass guard chunk. The guard is what proves no part"
                             " of the impulse response was truncated, and an undersized sweep"
                             " silently UNDERESTIMATES A, so only use this on a config you"
                             " have already validated.")
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use (default 0)")


########################################   varmap df command  #######################################


def parse_varmap_df(subparsers):
    """The analytic, detrender-free map.

    NO '-g/--gpu' FLAG, deliberately: this path runs no GPU kernel, so a device flag would
    offer a choice that does not exist. It does still need a CUDA context -- see varmap_df().
    """
    help_text = ("Compute the variance map A by the fast detrender-free algorithm, and write"
                 " it to an ASDF file")
    parser = subparsers.add_parser("df", help=help_text, description=help_text)
    parser.set_defaults(func=varmap_df)
    _add_varmap_common_args(parser)
    parser.add_argument('-e', '--epsilon', type=float, default=None, metavar='EPS',
                        help="Relative singular-value threshold, applied per group. Omit for"
                             " the per-group default max(1e-11, 16 * max(nrow,ncol) * eps_f64),"
                             " which is the float64 noise floor on singular values at that"
                             " group's size.")
    parser.add_argument('-m', '--max-bytes', type=_parse_size, default=None, metavar='SIZE',
                        help="Ceiling on the lifted Q, which is the only large allocation"
                             " here (32.0 GiB for a fine map at chime_sb2_et.yml, 6.9 GiB at"
                             " L=4), doubled when svd-optimization is on (-s 1 or 2), since"
                             " the SVD allocates a second array of the same shape. Accepts a"
                             " K/M/G/T suffix, e.g. '64G'. Omit for no limit; the size is"
                             " reported before the allocation either way, so a run that is"
                             " about to fail says why rather than being killed by the OOM"
                             " reaper.")
    parser.add_argument('-s', '--svd-optimization-level', type=int, default=2, metavar='N',
                        choices=(0, 1, 2),
                        help="How much of the factorization to rebuild at its true rank:"
                             " 0 = none, 1 = the base tree only, 2 = the base tree and every"
                             " higher tree again after the row restriction (default)."
                             " Exact, not an approximation. The base pass is where the rank"
                             " goes -- measured 24%% off at toy.yml, 54-57%% at CHIME and"
                             " 59-60%% at CHORD -- and the second pass adds 0.8-3.6%%. Use 0"
                             " if you are short of memory or in a hurry: at"
                             " chord_sb2_et.yml the map builds in 26 s and the base"
                             " optimization takes 854 s.")
    parser.add_argument('--debug', action='store_true',
                        help="Turn on SdPlan's O(subbands) planning-pass cross-checks. Too"
                             " expensive to leave on at production scale.")

    # Accepted only so that varmap_df() can reject them with a message naming 'varmap bf'.
    # Without these, argparse reports 'unrecognized arguments: det.yml' against the TOP-LEVEL
    # usage line, which tells a user neither what went wrong nor where to go. SUPPRESS keeps
    # them out of --help, so they are not offered as options.
    parser.add_argument('detrender_file', nargs='?', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--no-detrender', action='store_true', help=argparse.SUPPRESS)


def _parse_size(s):
    """A '--max-bytes' argument as an integer number of bytes.

    Accepts a bare integer, or a decimal with a binary K/M/G/T suffix ('64G' = 64*2^30,
    '1.5T'). Case-insensitive. Raises argparse.ArgumentTypeError on anything else, so
    argparse reports it as a bad argument rather than a traceback.
    """
    t = str(s).strip().upper()
    mult = 1
    if t and t[-1] in 'KMGT':
        mult = 1 << (10 * (1 + 'KMGT'.index(t[-1])))
        t = t[:-1]
    try:
        v = float(t)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{s!r} is not a size: expected an integer number of bytes, optionally with a"
            " K/M/G/T suffix (e.g. 1048576, '512M', '64G', '1.5T')")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"{s!r} must be positive")
    return int(v * mult)


def _parse_channel_spec(spec, nfreq):
    """A 'varmap bf --channels' argument as a sorted list of input channel indices. Accepts a
    comma-separated mix of bare indices and LO:HI[:STEP] slices."""

    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            f = part.split(':')
            if len(f) > 3:
                raise RuntimeError(f"varmap bf: --channels: '{part}' has too many colons")
            lo = int(f[0]) if f[0] else 0
            hi = int(f[1]) if (len(f) > 1 and f[1]) else nfreq
            step = int(f[2]) if (len(f) > 2 and f[2]) else 1
            out.extend(range(lo, hi, step))
        else:
            out.append(int(part))
    if not out:
        raise RuntimeError(f"varmap bf: --channels={spec!r} selects no channels")
    return sorted(set(out))


# Config keys that this tool overrides, with the value it forces, because the sweep requires
# them (see varmap.brute_force._SweepGeometry). Each is safe to override because none can
# change A: the analytic route (varmap.detrender_free) computes the same matrix and never
# reads any downsampling factor.
#
# The peak-finder's Dcore is not among them, and could not be: it is a property of a
# peak-finding kernel, not of the config. The sweep never sees it -- it ends in a PfSquare,
# which evaluates h_p at every time sample.
def _varmap_bf_override_config(config, nbeams=1):
    overrides = []

    def _set(obj, field, want, label):
        got = getattr(obj, field)
        if int(got) != int(want):
            setattr(obj, field, want)
            overrides.append(f'{label}: {int(got)} -> {int(want)}')

    _set(config, 'beams_per_gpu', nbeams, 'beams_per_gpu')
    _set(config, 'beams_per_batch', nbeams, 'beams_per_batch')
    _set(config, 'num_active_batches', 1, 'num_active_batches')

    return overrides


def varmap_bf(args):
    """Sweep, and write a pirate_frb.varmap file.

    NOTE THE OUTPUT FORMAT: this writes varmap/asdf_io.py's format. An older, incompatible
    variance-map format existed; the reader refuses such a file by name rather than
    misreading it, and nothing can read one any more.
    """

    from .kernels import Detrender2dParams

    # ---- Argument-level rejections. The detrender arguments can contradict each other or
    # be absent, and both are worth catching before a config is even loaded.
    if args.no_detrender and (args.detrender_file is not None):
        raise RuntimeError("varmap bf: --no-detrender was given together with"
                           f" '{args.detrender_file}'. These say opposite things; pass one or"
                           " the other.")
    if (not args.no_detrender) and (args.detrender_file is None):
        raise RuntimeError("varmap bf: no detrender specified. Pass a Detrender2dParams"
                           " yaml file, or --no-detrender to run without one.")

    config = DedispersionConfig.from_yaml(args.config_file)
    detrender = Detrender2dParams.from_yaml(args.detrender_file) if args.detrender_file else None

    # ---- Config-level rejections. Collected rather than raised one at a time, so that a
    # user editing a config does not discover the requirements one run at a time.
    #
    # Only things the SWEEP cannot check for itself belong here. Device capabilities do not:
    # _SweepGeometry and _GpuSweep check their own, and their messages name the kernel or the
    # quantity at fault. A blanket rejection here drifts as those capabilities change -- this
    # block used to refuse early triggers and num_primary_trees > 1 outright, which by the
    # time it was removed was refusing configs that both sweeps handle (early triggers) and
    # configs the CPU sweep handles (multiple primary trees).
    errs = []

    if detrender is not None:
        # The two files carry three quantities in common. Beam counts are overridden rather
        # than checked (they cannot change A); the other two must agree, since a mismatch
        # means the pair of files does not describe one computation.
        nfreq = int(config.get_total_nfreq())
        if int(detrender.nfreq) != nfreq:
            errs.append(f"{args.detrender_file}: nfreq = {int(detrender.nfreq)}, but"
                        f" {args.config_file} has sum(zone_nfreq) = {nfreq}")
        if int(detrender.T) != int(config.time_samples_per_chunk):
            errs.append(f"{args.detrender_file}: time_samples_per_chunk ="
                        f" {int(detrender.T)}, but {args.config_file} has"
                        f" {int(config.time_samples_per_chunk)}")

    if errs:
        raise RuntimeError("varmap bf: the config is not usable by this tool:\n  - "
                           + "\n  - ".join(errs))

    # ---- Overrides. One beam always: the beam axis carries passes, not beams, and
    # measurement showed that batching does not speed up a full sweep.
    overrides = _varmap_bf_override_config(config, nbeams=1)
    if detrender is not None and int(detrender.M) != 1:
        overrides.append(f'detrender num_beams: {int(detrender.M)} -> 1')
        detrender.M = 1

    for o in overrides:
        atomic_print(f"varmap bf: overriding {o}")

    config.validate()

    channels = (_parse_channel_spec(args.channels, int(config.get_total_nfreq()))
                if (args.channels is not None) else None)

    # Before constructing the plan, not just before the GPU sweep: DedispersionPlan allocates
    # through cudaHostAlloc, so even the CPU path needs a cuda device selected.
    ksgpu.set_cuda_device(args.gpu)

    t0 = time.time()
    vmm = varmap.compute_variance_multimap(
        config, detrender=detrender, device=('cpu' if args.cpu else 'gpu'),
        L=args.coarse_grain, guard_chunk=(not args.no_guard_chunk), progress=True,
        channels=channels, scratch_dir=args.scratch_dir,
        provenance=dict(overrides=overrides, command=' '.join(sys.argv)))
    dt = time.time() - t0

    vmm.write_asdf(args.output)

    nbytes = sum(m.nbytes() for m in vmm.maps)
    atomic_print(f"varmap bf: swept {vmm.provenance['npasses']} passes in {dt:.1f} s; wrote"
                 f" {args.output} ({nbytes/2**20:.1f} MiB of float64 in"
                 f" {vmm.num_primary_trees} primary tree(s), covering {vmm.ntrees} tree(s))")


def varmap_df(args):
    """Compute the analytic detrender-free map, and write a pirate_frb.varmap file.

    NO set_cuda_device() CALL, and no -g flag: this path runs no GPU kernel. It does still
    need a CUDA CONTEXT, because config.make_channel_map() returns a ksgpu::Array allocated
    through cudaHostAlloc -- with no device visible at all it fails with 'cudaHostAlloc
    returned 100 (no CUDA-capable device is detected)'. CUDA picks device 0 by default;
    a caller who needs to steer that uses CUDA_VISIBLE_DEVICES.

    NO CONFIG OVERRIDES either, unlike 'varmap bf'. Those exist because the SWEEP requires
    them (beams_per_gpu, beams_per_batch, num_active_batches); this path runs no dedisperser
    and reads none of them. So the archived config is exactly the one the user wrote, which
    is better provenance.
    """

    # THE NO-DETRENDER HYPOTHESIS IS LOAD-BEARING here, not a missing feature: the step from
    # the base tree to the other primary trees is Proposition 2, which is FALSE with a
    # Detrender2d in front (measured against the brute-force sweep at 4.9e-7 without one and
    # 2.1 WITH one). So this is a real guard, and it points at the tool that can do the job.
    if args.detrender_file is not None:
        raise RuntimeError(f"varmap df: got a second positional argument"
                           f" '{args.detrender_file}'. This algorithm is detrender-free by"
                           " construction and takes no detrender file; use 'varmap bf' for a"
                           " map with a Detrender2d.")
    if args.no_detrender:
        raise RuntimeError("varmap df: --no-detrender is not accepted, because this algorithm"
                           " never uses a detrender -- there is nothing to switch off. It is"
                           " 'varmap bf' that requires you to say which you want.")

    config = DedispersionConfig.from_yaml(args.config_file)
    config.validate()

    t0 = time.time()
    vmm = varmap.compute_detrender_free_multi_map(
        config, L=args.coarse_grain, epsilon=args.epsilon, max_bytes=args.max_bytes,
        svd_optimization_level=args.svd_optimization_level,
        progress=True, debug=args.debug,
        provenance=dict(command=' '.join(sys.argv)))
    dt = time.time() - t0

    vmm.write_asdf(args.output)

    nbytes = sum(m.nbytes() for m in vmm.maps)
    ranks = [m.factor_rank for m in vmm.maps]
    atomic_print(f"varmap df: built {vmm.num_primary_trees} map(s) in {dt:.1f} s at rank(s)"
                 f" {ranks}; wrote {args.output} ({nbytes/2**20:.1f} MiB of float64 in"
                 f" {vmm.num_primary_trees} primary tree(s), covering {vmm.ntrees} tree(s))")


########################################   varmap mc command  #######################################


def parse_varmap_mc(subparsers):
    """Check a stored map against Monte-Carlo sims. Takes a MAP, not a config, so it shares
    none of _add_varmap_common_args() -- there is no -o and no -L."""
    help_text = ("Check a stored variance map against Monte-Carlo sims of its embedded config")
    parser = subparsers.add_parser("mc", help=help_text, description=help_text)
    parser.set_defaults(func=varmap_mc)
    parser.add_argument('map_file', help="Path to a .asdf variance-map file")
    parser.add_argument('-v', '--freq-variances', default=None, metavar='PATH',
                        help="Length-nfreq input-channel variances: a .npy file, or a text"
                             " file of whitespace-separated floats. Default all ones. NOTE the"
                             " result is a statement about THIS v; a map is admissible for all"
                             " v >= 0, and one run checks one of them.")
    parser.add_argument('-n', '--nchunks', type=int, default=None, metavar='N',
                        help="Stop after N chunks (default: run until Ctrl-C)")
    parser.add_argument('--report-every', type=int, default=1, metavar='N',
                        help="Print the summary every N chunks (default 1)")
    parser.add_argument('--cpu', action='store_true',
                        help="Use ReferenceDedisperser instead of the GPU pipeline. Orders of"
                             " magnitude slower; the path for a config the GPU pipeline"
                             " refuses (stage-2 dd_rank < 3, or a missing sbdd kernel).")
    parser.add_argument('-s', '--sophistication', type=int, default=1, metavar='N',
                        help="ReferenceDedisperser sophistication (0, 1 or 2; default 1)."
                             " --cpu only.")
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help="GPU to use (default 0). Needed on BOTH paths: DedispersionPlan"
                             " allocates through cudaHostAlloc.")


def _read_freq_variances(path, nfreq):
    """A '-v' argument as a length-nfreq float64 array. .npy, else whitespace-separated text."""

    import numpy as np

    if path.endswith('.npy'):
        v = np.load(path)
    else:
        v = np.loadtxt(path)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size != nfreq:
        raise RuntimeError(f"varmap mc: {path}: got {v.size} variances, config has"
                           f" nfreq={nfreq}")
    return v


# Config fields 'varmap mc' overrides, with why each is safe. Unlike varmap bf's list this one
# also touches dtype, which is NOT provably A-preserving -- see the comment at its _set() call.
def _varmap_mc_override_config(config):
    import numpy as np

    overrides = []

    def _set(obj, field, want, label):
        got = getattr(obj, field)
        if str(got) != str(want):
            setattr(obj, field, want)
            overrides.append(f'{label}: {got} -> {want}')

    # One beam. The beam axis is a pure spectator, so this cannot change A -- and for a fixed
    # budget of beam-chunks, one beam wastes the least on warmup: B beams over N/B chunks give
    # (N - B*S) steady samples against (N - S) for one beam, with S the settling chunk count.
    _set(config, 'beams_per_gpu', 1, 'beams_per_gpu')
    _set(config, 'beams_per_batch', 1, 'beams_per_batch')
    _set(config, 'num_active_batches', 1, 'num_active_batches')

    # A pure-GPU MegaRingbuf, which the GPU pipeline requires. This is a ring-buffer PLACEMENT
    # decision -- where segments live, not what is computed -- so it cannot change A. The map's
    # config is embedded in the asdf file, so without this a user would have to REGENERATE the
    # map to check it.
    _set(config, 'max_gpu_clag', 10000, 'max_gpu_clag')

    # NO MC PATH IMPLEMENTS float16, on either device: GpuSbDedispersionKernel is float32-only,
    # ReferenceDedispersionKernel "uses float32, regardless of what dtype is specified", and
    # both PfSquares take float32. So there is no float16 MC to offer, and this is a conversion
    # rather than a choice. What it does change: config.dtype drives the STAGE-1 kernel and the
    # MegaRingbuf layout, which do support float16 -- so a float16 config would produce a ring
    # buffer the float32-only stage-2 kernel cannot read.
    # config.dtype is a numpy dtype on the python side, so compare and assign as one.
    if np.dtype(config.dtype) != np.float32:
        overrides.append(f'dtype: {np.dtype(config.dtype)} -> float32'
                         ' (no MC path implements float16)')
        config.dtype = np.float32

    return overrides


def varmap_mc(args):
    """Monte-Carlo check of a stored variance map. See pirate_frb/varmap/mc.py."""

    import numpy as np

    if args.sophistication != 1 and not args.cpu:
        raise RuntimeError("varmap mc: -s/--sophistication applies to ReferenceDedisperser,"
                           " so it is meaningful only with --cpu.")

    vmm = varmap.VarianceMultiMap.from_asdf(args.map_file)

    # Collected rather than raised one at a time, as 'varmap bf' does.
    errs = []
    if vmm.provenance.get('partial'):
        errs.append(f"{args.map_file} is a PARTIAL map (written by 'varmap bf --channels'):"
                    " its unswept columns are zero and it carries no y_true, precisely so that"
                    " nothing downstream scores it.")
    if errs:
        raise RuntimeError('varmap mc: this file cannot be checked:\n  - ' + '\n  - '.join(errs))

    config = vmm.config
    config.validate()
    nfreq = int(config.get_total_nfreq())

    v = (np.ones(nfreq) if (args.freq_variances is None)
         else _read_freq_variances(args.freq_variances, nfreq))

    overrides = _varmap_mc_override_config(config)

    atomic_print(f"varmap mc: {args.map_file}: algorithm="
                 f"{vmm.provenance.get('algorithm', '?')}, {vmm.num_primary_trees} primary"
                 f" tree(s), {vmm.ntrees} tree(s)")
    atomic_print(f"  freq_variances: {'all ones (pass -v to override)' if args.freq_variances is None else args.freq_variances}")
    atomic_print(f"  detrender: {'none' if vmm.detrender is None else 'from the file'}")
    for o in overrides:
        atomic_print(f"  overriding {o}")
    if any(o.startswith('dtype:') for o in overrides):
        atomic_print("  NOTE: stage-1 dedispersion therefore runs in float32, where production"
                     " would use\n        float16. Stages 2 and 3 are float32 either way, so"
                     " that is the only\n        arithmetic that differs from production.")
    atomic_print("  eps = MC/map - 1;  eps > 0 means the map UNDERESTIMATES"
                 " (bad for an admissible map)")

    # Before the plan is built, and needed even for --cpu: DedispersionPlan allocates through
    # cudaHostAlloc.
    ksgpu.set_cuda_device(args.gpu)

    from .varmap.mc import run_mc
    run_mc(vmm, v, device=('cpu' if args.cpu else 'gpu'), nchunks=args.nchunks,
           report_every=args.report_every, sophistication=args.sophistication)


#########################################   time command  ##########################################


def parse_time(subparsers):
    help_text = "Run timings (use flags to select specific timings)"
    parser = subparsers.add_parser("time", help=help_text, description=help_text)
    parser.set_defaults(func=time_command)
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for timing (default 0)")
    parser.add_argument('-t', '--nthreads', type=int, default=0, help="number of CPU threads (for time_cpu_downsample and time_avx2_simulate_4bit_noise)")
    parser.add_argument('--ncu', action='store_true', help="Just run a single kernel (intended for profiling with nvidia 'ncu')")
    parser.add_argument('--gldk', action='store_true', help='Runs GpuLaggedDownsamplingKernel.time_selected()')
    parser.add_argument('--gddk', action='store_true', help='Runs GpuDedispersionKernel.time_selected()')
    parser.add_argument('--casm', action='store_true', help='Runs CasmBeamformer.run_timings()')
    parser.add_argument('--chime', action='store_true', help='Runs time_chime_frb_{beamform,upchan}()')
    parser.add_argument('--zomb', action='store_true', help='Runs "zombie" timings (code that I wrote during protoyping that may never get used)')
    parser.add_argument('--cdd2', action='store_true', help='Runs CoalescedDdKernel2.time_selected()')
    parser.add_argument('--gdqk', action='store_true', help='Runs GpuDequantizationKernel.time_selected()')
    parser.add_argument('--gtgk', action='store_true', help='Runs GpuTreeGriddingKernel.time_selected()')
    parser.add_argument('--sim', action='store_true', help='Runs avx2_simulate_4bit_noise() timing')
    parser.add_argument('--dt1d', action='store_true', help='Runs Detrender1d.time_selected() (1-d detrender kernel)')
    parser.add_argument('--dt2d', action='store_true', help='Runs Detrender2d.time_selected() (2-d spline detrender kernel)')

def time_command(args):
    timing_flags = [ 'gldk', 'gddk', 'casm', 'chime', 'zomb', 'cdd2', 'gdqk', 'gtgk', 'sim', 'dt1d', 'dt2d' ]
    run_all_timings = not any(getattr(args,x) for x in timing_flags)

    if args.ncu:
        nflags = sum((1 if getattr(args,x) else 0) for x in timing_flags)
        if nflags != 1:
            raise RuntimeError(f'If --ncu is specified, then precisely one of {timing_flags} must be specified')
        if not args.casm:
            raise RuntimeError(f'Currently, the --ncu flag is only supported with --casm (FIXME)')
        
    ksgpu.set_cuda_device(args.gpu)
    nthreads = args.nthreads if (args.nthreads > 0) else os.cpu_count()
    from . import utils   # local import (utils pulls in heavier deps)
        
    if run_all_timings or args.gldk:
        kernels.GpuLaggedDownsamplingKernel.time_selected()
    if run_all_timings or args.gddk:
        kernels.GpuDedispersionKernel.time_selected()
    if run_all_timings or args.casm:
        casm.CasmBeamformer.run_timings(args.ncu)
    if run_all_timings or args.chime:
        chime.time_chime_frb_beamform()
        chime.time_chime_frb_upchan()
    if run_all_timings or args.zomb:
        loose_ends.time_cpu_downsample(nthreads)
        loose_ends.time_gpu_downsample()
        loose_ends.time_gpu_transpose()
    if run_all_timings or args.cdd2:
        kernels.CoalescedDdKernel2.time_selected()
    if run_all_timings or args.gdqk:
        kernels.GpuDequantizationKernel.time_selected()
    if run_all_timings or args.gtgk:
        kernels.GpuTreeGriddingKernel.time_selected()
    if run_all_timings or args.dt1d:
        kernels.Detrender1d.time_selected()
    if run_all_timings or args.dt2d:
        kernels.Detrender2d.time_selected()
    if run_all_timings or args.sim:
        utils.time_avx2_simulate_4bit_noise(nthreads)


#####################################   show hardware command  #####################################


def parse_show_hardware(subparsers):
    help_text = "Show hardware information, including cpu affinity"
    parser = subparsers.add_parser("hardware", help=help_text, description=help_text)
    parser.set_defaults(func=show_hardware)
    
def show_hardware(args):
    h = Hardware()
    h.show()


######################################   show kernels command  #####################################


def parse_show_kernels(subparsers):
    help_text = "Show registered cuda kernels (use flags to select specific registries)"
    parser = subparsers.add_parser("kernels", help=help_text, description=help_text)
    parser.set_defaults(func=show_kernels)
    parser.add_argument('--pfom', action='store_true', help='Show PfOutputMicrokernel registry')
    parser.add_argument('--pfwr', action='store_true', help='Show PfWeightReaderMicrokernel registry')
    parser.add_argument('--gddk', action='store_true', help='Show GpuDedispersionKernel registry')
    parser.add_argument('--gpfk', action='store_true', help='Show GpuPeakFindingKernel registry')
    parser.add_argument('--cdd2', action='store_true', help='Show CoalescedDdKernel2 registry')
    parser.add_argument('--sbdd', action='store_true', help='Show GpuSbDedispersionKernel registry')
    
def show_kernels(args):
    show_flags = [ 'pfom', 'pfwr', 'gddk', 'gpfk', 'cdd2', 'sbdd' ]
    show_all = not any(getattr(args, x) for x in show_flags)
    first = True

    if show_all or args.cdd2:
        if not first:
            atomic_print("\n")
        first = False
        n = kernels.CoalescedDdKernel2.registry_size()
        atomic_print(f"CoalescedDdKernel2 registry ({n} entries):")
        kernels.CoalescedDdKernel2.show_registry()

    if show_all or args.sbdd:
        if not first:
            atomic_print("\n")
        first = False
        n = kernels.GpuSbDedispersionKernel.registry_size()
        atomic_print(f"GpuSbDedispersionKernel registry ({n} entries):")
        kernels.GpuSbDedispersionKernel.show_registry()

    if show_all or args.pfom:
        if not first:
            atomic_print("\n")
        first = False
        n = kernels.PfOutputMicrokernel.registry_size()
        atomic_print(f"PfOutput microkernel registry ({n} entries):")
        kernels.PfOutputMicrokernel.show_registry()

    if show_all or args.pfwr:
        if not first:
            atomic_print("\n")
        first = False
        n = kernels.PfWeightReaderMicrokernel.registry_size()
        atomic_print(f"PfWeightReader microkernel registry ({n} entries):")
        kernels.PfWeightReaderMicrokernel.show_registry()

    if show_all or args.gddk:
        if not first:
            atomic_print("\n")
        first = False
        n = kernels.GpuDedispersionKernel.registry_size()
        atomic_print(f"Dedispersion kernel registry ({n} entries):")
        kernels.GpuDedispersionKernel.show_registry()
    
    if show_all or args.gpfk:
        if not first:
            atomic_print("\n")
        first = False
        n = kernels.GpuPeakFindingKernel.registry_size()
        atomic_print(f"Peak-finding kernel registry ({n} entries):")
        kernels.GpuPeakFindingKernel.show_registry()


######################################   dev make_subbands command  #####################################


def parse_make_subbands(subparsers):
    help_text = "A utility for maintaining makefile_helper.py"
    description = textwrap.dedent("""\
        A utility for maintaining makefile_helper.py.

        The 'threshold' argument is a "target" fractional bandwidth. For example, if threshold=0.2,
        then the make_subbands command will try to make bands whose fractional bandwidth is <= 20%.
        However, some subbands may be wider than the threshold, due to technical constraints.

        Example usage::

           # Specify frequency min, max, and threshold
           python -m pirate_frb dev make_subbands 300 1500 0.2
           python -m pirate_frb dev make_subbands 400 800 0.1 -r 4""")
    parser = subparsers.add_parser(
        "make_subbands",
        help = help_text,
        description = description,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=make_subbands)

    parser.add_argument('fmin', type=float, help='Minimum frequency (MHz)')
    parser.add_argument('fmax', type=float, help='Maximum frequency (MHz)')
    parser.add_argument('threshold', type=float, help='Threshold for fmin/fmax')
    parser.add_argument('-r', '--pf-rank', type=int, default=4, help='Peak finding rank (default: 4)')


def make_subbands(args):
    atomic_print(f'Constructing FrequencySubbands(pf_rank={args.pf_rank}, fmin={args.fmin}, fmax={args.fmax}, threshold={args.threshold})')

    # These asserts detect out-of-order positional arguments.
    assert args.fmin > 99.0
    assert args.fmin < args.fmax
    assert args.threshold <= 10.0
    
    fs = core.FrequencySubbands.from_threshold(args.fmin, args.fmax, args.threshold, args.pf_rank)
    atomic_print(fs.show())


########################################   dev hwtest command  #########################################


def parse_hwtest(subparsers):
    import argparse, textwrap
    help_text = "Run hardware test from a hwtest YAML config file (use -s to send data instead of receiving)"
    description = textwrap.dedent("""\
        Run hardware test using YAML config file (use -s to send data instead of receiving).

        Runs and times parallel synthetic loads: network IO, disk IO, PCIe transfers
        between GPU and host, GPU compute kernels, CPU compute kernels, and host and
        GPU memory bandwidth.

        Example networking-only run::

          # On cf00. The test will pause after "listening for TCP connections".
          python -m pirate_frb dev hwtest configs/hwtest/cf00_net64.yml

          # On cf05. Send to all four IP addresses on cf00.
          python -m pirate_frb dev hwtest -s configs/hwtest/cf00_net64.yml

        See configs/hwtest/*.yml for more examples.""")
    parser = subparsers.add_parser(
        "hwtest", help=help_text, description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=hwtest)
    parser.add_argument('config_file', help='Path to YAML config file')
    parser.add_argument('-t', '--time', type=float, default=20, help='Number of seconds to run test (default 20)')
    parser.add_argument('-s', '--send', action='store_true', help='Send data to test server (uses ip_addrs from config file)')


def parse_hwtest_config(filename):
    """Parse and validate a hwtest YAML config file. Returns a dict."""

    import yaml

    with open(filename) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise RuntimeError(f"{filename}: expected YAML mapping at top level, got {type(config).__name__}")

    # Define all valid keys, grouped by type.
    bool_keys = ['dedisperse', 'h2g_bw', 'g2h_bw', 'gmem_bw', 'hmem_bw', 'write_asdf']
    int_keys = ['tcp_connections_per_ip_address', 'write_threads_per_ssd', 'downsampling_threads_per_cpu']
    list_of_str_keys = ['ip_addrs', 'ssd_dirs', 'ssd_devices']
    all_valid_keys = set(bool_keys + int_keys + list_of_str_keys)

    # These keys must always be present. The remaining keys (tcp_connections_per_ip_address,
    # write_threads_per_ssd, ssd_devices) are conditionally required -- see below.
    always_required = set(bool_keys + ['ip_addrs', 'ssd_dirs', 'downsampling_threads_per_cpu']) - {'write_asdf'}

    # Check for unknown keys.
    unknown = set(config.keys()) - all_valid_keys
    if unknown:
        raise RuntimeError(f"{filename}: unrecognized key(s): {', '.join(sorted(unknown))}")

    # Check required keys are present.
    missing = always_required - set(config.keys())
    if missing:
        raise RuntimeError(f"{filename}: missing required key(s): {', '.join(sorted(missing))}")

    # Type-check booleans.
    for key in bool_keys:
        if key in config and not isinstance(config[key], bool):
            raise RuntimeError(f"{filename}: '{key}' must be true or false, got {repr(config[key])}")

    # Type-check integers (note: in Python, bool is a subclass of int, so we must exclude it).
    for key in int_keys:
        if key in config:
            if isinstance(config[key], bool) or not isinstance(config[key], int):
                raise RuntimeError(f"{filename}: '{key}' must be an integer, got {repr(config[key])}")

    # Type-check lists of strings.
    for key in list_of_str_keys:
        if key in config:
            if not isinstance(config[key], list):
                raise RuntimeError(f"{filename}: '{key}' must be a list, got {repr(config[key])}")
            for i, elem in enumerate(config[key]):
                if not isinstance(elem, str):
                    raise RuntimeError(f"{filename}: {key}[{i}] must be a string, got {repr(elem)}")

    # Range-check integers.
    if config['downsampling_threads_per_cpu'] < 0:
        raise RuntimeError(f"{filename}: 'downsampling_threads_per_cpu' must be >= 0, got {config['downsampling_threads_per_cpu']}")
    if 'tcp_connections_per_ip_address' in config and config['tcp_connections_per_ip_address'] < 1:
        raise RuntimeError(f"{filename}: 'tcp_connections_per_ip_address' must be >= 1, got {config['tcp_connections_per_ip_address']}")
    if 'write_threads_per_ssd' in config and config['write_threads_per_ssd'] < 1:
        raise RuntimeError(f"{filename}: 'write_threads_per_ssd' must be >= 1, got {config['write_threads_per_ssd']}")

    # Conditionally required: tcp_connections_per_ip_address (when ip_addrs is non-empty).
    if len(config['ip_addrs']) > 0 and 'tcp_connections_per_ip_address' not in config:
        raise RuntimeError(f"{filename}: 'tcp_connections_per_ip_address' is required when 'ip_addrs' is non-empty")

    # Conditionally required: ssd_devices, write_threads_per_ssd, write_asdf (when ssd_dirs is non-empty).
    if len(config['ssd_dirs']) > 0:
        if 'ssd_devices' not in config:
            raise RuntimeError(f"{filename}: 'ssd_devices' is required when 'ssd_dirs' is non-empty")
        if 'write_threads_per_ssd' not in config:
            raise RuntimeError(f"{filename}: 'write_threads_per_ssd' is required when 'ssd_dirs' is non-empty")
        if 'write_asdf' not in config:
            raise RuntimeError(f"{filename}: 'write_asdf' is required when 'ssd_dirs' is non-empty")
        if len(config['ssd_devices']) != len(config['ssd_dirs']):
            raise RuntimeError(
                f"{filename}: 'ssd_devices' has length {len(config['ssd_devices'])}, "
                f"but 'ssd_dirs' has length {len(config['ssd_dirs'])} (must be equal)"
            )

    return config


def hwtest(args):
    config = parse_hwtest_config(args.config_file)

    if args.send:
        hwtest_send_from_config(config)
        return

    server = Hwtest('Node test')
    hw = server.hardware

    # Validate IP addresses (checks that each IP is associated with a known NIC).
    for ip in config['ip_addrs']:
        hw.vcpu_list_from_ip_addr(ip)

    # Validate SSD dirs (checks that each dir is a known mount point).
    for ssd_dir in config['ssd_dirs']:
        hw.vcpu_list_from_dirname(ssd_dir)

    # Validate ssd_devices: check that each ssd_dir is backed by the corresponding ssd_device.
    if len(config['ssd_dirs']) > 0:
        for i, (ssd_dir, ssd_dev) in enumerate(zip(config['ssd_dirs'], config['ssd_devices'])):
            actual_dev = hw.disk_from_dirname(ssd_dir)
            if os.path.basename(actual_dev) != os.path.basename(ssd_dev):
                raise RuntimeError(
                    f"ssd_dirs[{i}]={ssd_dir!r} is backed by device {actual_dev!r}, "
                    f"but ssd_devices[{i}]={ssd_dev!r} (mismatch)"
                )

    # Add workers to server.

    if config['hmem_bw']:
        for icpu in range(hw.num_cpus):
            for v in hw.vcpu_list_from_cpu(icpu):
                server.add_memcpy_thread(-1, -1, cpu=icpu)

    if config['gmem_bw']:
        for gpu in range(hw.num_gpus):
            server.add_memcpy_thread(gpu, gpu, use_copy_engine=False)

    if config['downsampling_threads_per_cpu'] > 0:
        for icpu in range(hw.num_cpus):
            for _ in range(config['downsampling_threads_per_cpu']):
                server.add_downsampling_thread(icpu)

    if len(config['ssd_dirs']) > 0:
        for issd, ssd_dir in enumerate(config['ssd_dirs']):
            for thread in range(config['write_threads_per_ssd']):
                server.add_ssd_writer(f'{ssd_dir}/thread{thread}', issd, write_asdf=config['write_asdf'])

    if config['h2g_bw']:
        for gpu in range(hw.num_gpus):
            server.add_memcpy_thread(-1, gpu)

    if config['g2h_bw']:
        for gpu in range(hw.num_gpus):
            server.add_memcpy_thread(gpu, -1)

    if config['dedisperse']:
        for gpu in range(hw.num_gpus):
            server.add_chime_dedisperser(gpu)

    if len(config['ip_addrs']) > 0:
        for ip_addr in config['ip_addrs']:
            server.add_tcp_receiver(ip_addr, config['tcp_connections_per_ip_address'])

    server.run(args.time)


def hwtest_send_from_config(config):
    """Send data using ip_addrs/tcp_connections_per_ip_address from a hwtest config."""

    ip_addrs = config['ip_addrs']
    if len(ip_addrs) == 0:
        raise RuntimeError("hwtest --send: 'ip_addrs' must be non-empty")

    tcp_connections_per_ip_address = config['tcp_connections_per_ip_address']

    with HwtestSender(send_bufsize=65536, use_zerocopy=True, use_mmap=False, use_hugepages=True) as sender:
        for ip_addr in ip_addrs:
            sender.add_endpoint(ip_addr, tcp_connections_per_ip_address, 0)

        sender.start()

        try:
            while not sender.wait(pirate_pybind11.constants.default_poll_cadence_ms):
                pass
        except KeyboardInterrupt:
            atomic_print("\nInterrupted, stopping...")


######################################   dev scratch command  #######################################


def parse_scratch(subparsers):
    help_text = "For debugging: run whatever code is currently in src_lib/scratch.cu"
    parser = subparsers.add_parser("scratch", help=help_text, description=help_text)
    parser.set_defaults(func=scratch)

def scratch(args):
    # The scratch() function is defined in src_lib/scratch.cu.
    pirate_pybind11.scratch()


####################################   dev revisit_512gb command  ####################################


def parse_revisit_512gb(subparsers):
    help_text = "Re-test the ~511 GiB cudaHostRegister cap (failure expected)."
    description = (
        "Re-test the ~511 GiB single-call cudaHostRegister() cap on the current "
        "CUDA / driver version. The cap is an undocumented driver limit that "
        "currently forces pirate's BumpAllocator to register memory in chunks "
        "(see comments in BumpAllocator.hpp and constants.hpp). If this command "
        "starts succeeding on some future CUDA / driver release, the chunked-"
        "register workaround could potentially be simplified or removed.\n\n"
        "Test: mmap 550 GiB (hugepages with -H, 4 KiB pages otherwise), "
        "prefault, then attempt a single cudaHostRegister() over the entire "
        "region. Cleans up either way.\n\n"
        "Requires ~600 GiB of free memory of the requested type."
    )
    parser = subparsers.add_parser(
        "revisit_512gb", help=help_text, description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(func=revisit_512gb)
    parser.add_argument('-H', '--hugepages', action='store_true',
                        help='Use 2 MiB hugepages (default: 4 KiB regular pages).')


def revisit_512gb(args):
    # Test parameters.
    test_gib = 550
    need_gib = 600
    test_nbytes = test_gib * (1 << 30)
    hp2m = 2 * (1 << 20)

    # Force line-buffered stdout so Python prints + the C++-helper prints
    # appear in source order when the output is piped or redirected.
    sys.stdout.reconfigure(line_buffering=True)

    # Pin process (and any child threads) to CPU 0.
    os.sched_setaffinity(0, {0})
    atomic_print('Pinned process to CPU 0.')

    h = Hardware()
    atomic_print('\nHardware:')
    for gpu in range(h.num_gpus):
        bus_id = h._pcie_bus_id_from_gpu(gpu)
        desc = h._description_from_pcie_bus_id(bus_id)
        atomic_print(f'  GPU {gpu}: pcie={bus_id}  ({desc})')

    # Check memory availability.
    atomic_print("\n")
    if args.hugepages:
        if hp2m not in h.hugepage_sizes:
            raise RuntimeError(
                "2 MiB hugepages are not configured on this system. Allocate at\n"
                f"least {need_gib} GiB ({need_gib * 1024 // 2} pages) before re-running, e.g.:\n"
                f"  sudo bash -c 'echo {need_gib * 1024 // 2} > "
                f"/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages'\n"
                "(or set per-NUMA-node nr_hugepages files.)")
        pool = h.hugepage_pool(hp2m)
        free_gib = pool['free'] * hp2m / (1 << 30)
        if free_gib < need_gib:
            raise RuntimeError(
                f"Need >= {need_gib} GiB of 2 MiB hugepages free; only {free_gib:.1f} GiB free.\n"
                "Free up hugepages or allocate more before re-running.")
        atomic_print(f'  2 MiB hugepages free: {free_gib:.1f} GiB (test needs {need_gib} GiB)')
    else:
        # MemAvailable from /proc/meminfo (the kernel's estimate of how much
        # we can allocate without swapping). Note: this is for regular RAM;
        # hugepage-reserved memory is excluded.
        with open('/proc/meminfo') as f:
            mem_avail_kb = next(int(line.split()[1]) for line in f
                                if line.startswith('MemAvailable:'))
        avail_gib = mem_avail_kb / (1 << 20)
        if avail_gib < need_gib:
            raise RuntimeError(
                f"Need >= {need_gib} GiB MemAvailable for 4 KiB-paged test; "
                f"got {avail_gib:.1f} GiB.\nFree up memory (or reduce hugepage "
                "reservations) before re-running.")
        atomic_print(f'  MemAvailable: {avail_gib:.1f} GiB (test needs {need_gib} GiB)')

    page_label = 'hugepages' if args.hugepages else '4 KiB pages'
    atomic_print(f'\nAllocating + registering {test_gib} GiB ({page_label})...')
    success = pirate_pybind11.revisit_512gb_inner(test_nbytes, args.hugepages)

    bar = '=' * 64
    atomic_print("\n")
    atomic_print(bar)
    if success:
        atomic_print(f'cudaHostRegister({test_gib} GiB) SUCCEEDED.')
        atomic_print(f'On this CUDA / driver version, the ~511 GiB single-call cap')
        atomic_print(f'appears to have been LIFTED. Pirate\'s chunked-register workaround')
        atomic_print(f'in BumpAllocator could potentially be simplified or removed --')
        atomic_print(f'verify on multiple hardware/driver combinations before doing so.')
    else:
        atomic_print(f'cudaHostRegister({test_gib} GiB) FAILED (this is the expected outcome).')
        atomic_print(f'The ~511 GiB single-call cap is still in effect on this CUDA / driver')
        atomic_print(f'version. Pirate\'s chunked-register workaround in BumpAllocator')
        atomic_print(f'remains necessary.')
    atomic_print(bar)


################################   show xengine_metadata command  ##################################


def parse_show_xengine_metadata(subparsers):
    help_text = "Parse xengine_metadata yml file and write info to stdout"
    parser = subparsers.add_parser("xengine_metadata", help=help_text, description=help_text)
    parser.set_defaults(func=show_xengine_metadata)
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Include comments explaining the meaning of each field")


def show_xengine_metadata(args):
    metadata = core.XEngineMetadata.from_yaml_file(args.config_file)
    yaml_str = metadata.to_yaml_string(args.verbose)
    atomic_print(yaml_str)


###################################   show dedisperser command  ###################################


def parse_show_dedisperser(subparsers):
    help_text = "Parse a dedisperser .yml file and write info to stdout"
    parser = subparsers.add_parser("dedisperser", help=help_text, description=help_text)
    parser.set_defaults(func=show_dedisperser)
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Include comments explaining the meaning of each field")
    parser.add_argument('-c', '--config', action='store_true', help="Also print the DedispersionConfig, with a separator, before the plan (by default only the plan is printed, matching the dedispersion_plan_yaml sent to the grouper)")
    parser.add_argument('-t', '--time', action='store_true', help="Also print how long DedispersionPlan construction and the C++ compute_detrender_free_varcoarse() took (non-deterministic lines; off by default so the output is reproducible)")
    parser.add_argument('-z', '--zones', action='store_true', help="Include the per-clag mega_ringbuf host/gpu zone breakdown (independent of -v, which controls comments)")
    parser.add_argument('-s', '--streams', type=int, help="Override config.num_active_batches with specified value")
    parser.add_argument('-b', '--beams', type=int, help="Override config.beams_per_gpu with specified value")
    parser.add_argument('-g', '--max-gpu-clag', type=int, help="Override config.max_gpu_clag with specified value")
    parser.add_argument('--channel-map', action='store_true', help="Show channel map tree->freq (warning: produces long output!)")
    parser.add_argument('-r', '--resources', action='store_true', help="Show resource tracking (all kernels must be precompiled)")
    parser.add_argument('-R', '--fine-grained-resources', action='store_true', help="Like -r, but shows fine-grained per-kernel info")
    parser.add_argument('--test', action='store_true', help="Run GpuDedisperser.test_one() with config")


def show_dedisperser(args):
    config = DedispersionConfig.from_yaml(args.config_file)
    
    # Override config members if command-line flags were specified
    if args.streams is not None:
        config.num_active_batches = args.streams
    if args.beams is not None:
        config.beams_per_gpu = args.beams
    if args.max_gpu_clag is not None:
        config.max_gpu_clag = args.max_gpu_clag
        
    config.validate()
    config.test()   # I decided to run the unit tests here, since they're very fast!

    # Header line (verbose only, like all other comments): record the exact command line,
    # so readers of a generated yaml file know how to regenerate it. Reconstructed from
    # sys.argv[1:] with a literal 'pirate_frb' prefix (argv[0] is the __main__.py path
    # under 'python -m pirate_frb'). Deterministic, so generated files (e.g.
    # configs/example_dedispersion_plan.yml) stay reproducible.
    if args.verbose:
        cmdline = ' '.join(shlex.quote(a) for a in sys.argv[1:])
        atomic_print(f'# Created with: pirate_frb {cmdline}\n\n')

    # By default print only the DedispersionPlan, with no separator, so that the
    # output matches the dedispersion_plan_yaml that the FRB search sends to the
    # grouper (see FrbServer / frb_grouper.proto). With -c, also print the
    # DedispersionConfig (the dedispersion_config_yaml wire field) first, with a
    # human-readable separator before the plan.
    if args.config:
        config_yaml = config.to_yaml_string(args.verbose)
        if args.verbose:
            config_yaml = align_inline_comments(config_yaml)
        atomic_print(config_yaml)
        print_separator('DedispersionPlan starts here')

    t0 = time.time()
    plan = DedispersionPlan(config)
    plan_dt = time.time() - t0
    if args.time:
        atomic_print(f'# DedispersionPlan construction took {plan_dt:.3f} seconds\n\n')
        # Also time the C++ compute_detrender_free_varcoarse(). This is what the real-time
        # server pays on every weight update: GpuDedisperser::_fill_analytic_weights() calls
        # it, so the number below is the one that matters operationally. (Uses unit input
        # variances -- the running time does not depend on the values.)
        import numpy as np
        freq_variances = np.ones(int(plan.nfreq), dtype=np.float64)
        t0 = time.time()
        compute_detrender_free_varcoarse(plan, freq_variances)
        avar_dt = time.time() - t0
        atomic_print(f'# C++ compute_detrender_free_varcoarse() took {avar_dt:.3f} seconds\n\n')
    plan_yaml = plan.to_yaml_string(args.verbose, args.zones)
    if args.verbose:
        plan_yaml = indent_dedispersion_plan_comments(plan_yaml)
        plan_yaml = align_inline_comments(plan_yaml)
    atomic_print(plan_yaml)

    if args.channel_map:
        print_separator('Channel map starts here')
        channel_map = config.make_channel_map()
        
        atomic_print("\n")
        atomic_print('Channel map (tree_index -> freq_index -> frequency)')
        for i in range(len(channel_map)):
            freq_index = channel_map[i]
            freq = config.index_to_frequency(freq_index)
            atomic_print(f'  tree_index={i}  freq_index={freq_index:.4f}  freq={freq:.2f}')

    if args.resources or args.fine_grained_resources:
        print_separator('Resource tracking starts here (assumes 4-bit raw data)')
        nin = plan.beams_per_batch * plan.nfreq * plan.nt_in
        nbits = plan.nbits

        # Add a dequantizer and h2g copies (raw_data + scales_offsets), to give
        # a more realistic accounting of cost. Matches GpuDedisperser::time().
        raw_bytes   = (nin * 4) // 8      # int4 input
        out_bytes   = (nin * nbits) // 8  # fp16/fp32 output
        scoff_bytes = nin // 64           # 4 bytes per (scale, offset) pair, one pair per 256 samples
        stream_pool = core.CudaStreamPool(plan.num_active_batches)
        dedisperser = GpuDedisperser(plan, stream_pool, cuda_device_id=0, num_consumers=1)
        rt = dedisperser.resource_tracker.clone()
        rt.add_kernel('dequantizer',        raw_bytes + scoff_bytes + out_bytes)
        rt.add_memcpy_h2g('raw_data',       raw_bytes)
        rt.add_memcpy_h2g('scales_offsets', scoff_bytes)

        multiplier = (config.beams_per_gpu / config.beams_per_batch) / (1.0e-3 * config.time_samples_per_chunk * config.time_sample_ms)
        fine_grained = args.fine_grained_resources
        atomic_print(rt.to_yaml_string(multiplier, fine_grained))

    if args.test:
        print_separator('Testing GpuDedisperser')
        nchunks = (2**(config.toplevel_tree_rank + config.num_primary_trees - 1)) // config.time_samples_per_chunk + 10
        atomic_print(f'Running GpuDedisperser.test_one(config, nchunks={nchunks})')
        GpuDedisperser.test_one(config, nchunks)
        atomic_print('Test passed!')


###################################   show random_config command  ###################################


def parse_show_random_config(subparsers):
    help_text = "For debugging: generate random DedispersionConfig(s) and print as YAML"
    parser = subparsers.add_parser("random_config", help=help_text, description=help_text)
    parser.set_defaults(func=show_random_config)
    parser.add_argument('-n', type=int, default=1, metavar='NCONFIG', help='generate multiple random configs')
    parser.add_argument('-a', action='store_true', help='generate arbitrary random config, without restricting to precompiled kernels')
    parser.add_argument('-v', action='store_true', help='verbose')


def show_random_config(args):
    gpu_valid = not args.a
    
    for i in range(args.n):
        if args.n > 1:
            print_separator(f'iteration {i+1}/{args.n}', filler='#')
        
        config = DedispersionConfig.make_random(gpu_valid=gpu_valid)
        yaml_str = config.to_yaml_string(verbose=args.v)
        atomic_print(yaml_str)


######################################   dev coverage command  ######################################


def parse_coverage(subparsers):
    help_text = "Coverage analysis of randomization utils in unit tests"
    parser = subparsers.add_parser("coverage", help=help_text, description=help_text)
    parser.set_defaults(func=coverage)
    parser.add_argument('--config', action='store_true',
                        help='DedispersionConfig::make_random(), at each setting its callers use')
    parser.add_argument('--reg', action='store_true',
                        help='Kernel registry marginals (what this build compiled, not a draw)')
    parser.add_argument('--varmap', action='store_true',
                        help='varmap draws: _random_config(), the LP cell, the sweep loops')
    parser.add_argument('--dt', action='store_true',
                        help='Detrending draws: random_knots(), random_nfreq(), the 2-d masks')
    parser.add_argument('-s', '--scale', type=float, default=1.0, metavar='X',
                        help='Multiply every draw count by X (default 1). Scale up when a rate'
                             ' is near a band edge and you want to know whether it moved.')


def coverage(args):
    flags = [f for f in ('config', 'reg', 'varmap', 'dt') if getattr(args, f)]
    tests.report_coverage(select=flags, scale=args.scale)


###################################   time_dedisperser command  ###################################


def parse_time_dedisperser(subparsers):
    help_text = "Run timing benchmarks from a dedisperser .yml file"
    parser = subparsers.add_parser("time_dedisperser", help=help_text, description=help_text)
    parser.set_defaults(func=time_dedisperser)
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-n', '--niter', type=int, default=1000, help="Number of iterations for timing (default 1000)")
    parser.add_argument('-b', '--beams', type=int, help="Override config.beams_per_gpu with specified value")
    parser.add_argument('-g', '--max-gpu-clag', type=int, help="Override config.max_gpu_clag with specified value")
    parser.add_argument('-H', '--no-hugepages', action='store_true', help="Disable hugepages")
    parser.add_argument('--python', action='store_true', help="Use Python/cupy timing instead of C++ (for testing pybind11 interface)")


def time_dedisperser(args):
    from . import utils
    from .run_server import compute_async_bump_nthreads

    # Pin thread to first CPU (for consistent timing on dual-CPU systems)
    hw = Hardware()
    vcpu_list = hw.vcpu_list_from_cpu(0)
    core.set_thread_affinity(vcpu_list)
    atomic_print(f'Pinned thread to CPU 0 (vcpus {vcpu_list})')
    
    config = DedispersionConfig.from_yaml(args.config_file)

    # Override config members if command-line flags were specified
    if args.beams is not None:
        config.beams_per_gpu = args.beams
    if args.max_gpu_clag is not None:
        config.max_gpu_clag = args.max_gpu_clag

    plan = DedispersionPlan(config)
    
    niterations = args.niter
    use_hugepages = not args.no_hugepages
    use_python = args.python
    
    # Set up allocator flags
    gpu_aflags = 'af_gpu | af_zero'
    cpu_aflags = 'af_rhost | af_zero'
    if use_hugepages:
        cpu_aflags += ' | af_mmap_huge'
    
    # Create GpuDedisperser (unallocated, to get resource tracking)
    atomic_print(f'Creating GpuDedisperser...')
    stream_pool = core.CudaStreamPool(plan.num_active_batches)
    dedisperser = GpuDedisperser(plan, stream_pool, cuda_device_id=0,
                                 num_consumers=1)
    
    # Calculate total memory needed. Dedisperser footprints come from
    # resource tracking and already include BumpAllocator's 128-byte
    # alignment. The timing loop additionally allocates four user-side
    # arrays (matching the layouts in GpuDedisperser::time() and
    # pirate_frb.utils.time_cupy_dedisperser()):
    #   multi_raw_{cpu,gpu}:   (S, B, F, T) int4   ->  S*B*F*T/2 bytes each
    #   multi_scoff_{cpu,gpu}: (S, B, F, T//256, 2) fp16 -> S*B*F*T/64 bytes each
    # Both raw and scoff are needed on each side (cpu_allocator and
    # gpu_allocator) -- the timing loop copies them h2g.
    S = plan.num_active_batches
    B = plan.beams_per_batch
    F = plan.nfreq
    T = plan.nt_in
    raw_nbytes   = S * B * F * (T // 2)
    scoff_nbytes = S * B * F * (T // 256) * 2 * 2     # 2 fp16 entries per minichunk
    alignment_margin = 256                            # 128 bytes per user-side allocation (raw + scoff)

    gpu_nbytes = dedisperser.resource_tracker.get_gmem_footprint() + raw_nbytes + scoff_nbytes + alignment_margin
    cpu_nbytes = dedisperser.resource_tracker.get_hmem_footprint() + raw_nbytes + scoff_nbytes + alignment_margin
    
    # Create allocators with pre-computed capacities and allocate. The
    # cpu allocator runs in async mode so its (slow) cudaHostRegister +
    # zeroing overlaps with the gpu allocator's cudaMalloc + cudaMemset;
    # nthreads uses the same formula as run_server. The gpu allocator
    # stays sync (gpu init is fast enough that the async machinery
    # isn't worth it here).
    nthreads = compute_async_bump_nthreads(vcpu_list, cpu_nbytes)
    atomic_print(f'Allocating (gpu={gpu_nbytes/1e9:.3f} GB sync, '
                 f'cpu={cpu_nbytes/1e9:.3f} GB async, nthreads={nthreads})...')
    gpu_allocator = core.BumpAllocator(gpu_aflags, gpu_nbytes, cuda_device=0)
    cpu_allocator = core.BumpAllocator(cpu_aflags, cpu_nbytes,
                                       is_async=True, nthreads=nthreads,
                                       cuda_device=0)
    cpu_allocator.wait_until_initialized()
    dedisperser.allocate(gpu_allocator, cpu_allocator)
    
    # Run timing
    atomic_print(f'Running timing (niterations={niterations}, use_hugepages={use_hugepages}, python={use_python})...')
    if use_python:
        # Python version of timing code: pirate_frb.utils.time_cupy_dedisperser().
        utils.time_cupy_dedisperser(dedisperser, gpu_allocator, cpu_allocator, niterations)
    else:
        # C++ version of timing code: GpuDedisperser::time().
        dedisperser.time(gpu_allocator, cpu_allocator, niterations)
    
    atomic_print('Timing complete!')


###################################   show asdf command  ###################################


def parse_show_asdf(subparsers):
    help_text = "Print the YAML header of an ASDF file. (Note: 'asdftool --info' is also useful)"
    parser = subparsers.add_parser("asdf", help=help_text, description=help_text)
    parser.set_defaults(func=show_asdf)
    parser.add_argument('asdf_file', help="Path to ASDF file")


def show_asdf(args):
    from .utils import show_asdf as _show_asdf
    _show_asdf(args.asdf_file)


######################################   show file_format command  ##################################


def parse_show_file_format(subparsers):
    help_text = "Make an asdf file from an xengine_metadata YAML file, and write the header to stdout."
    parser = subparsers.add_parser("file_format", help=help_text, description=help_text)
    parser.set_defaults(func=show_file_format)
    parser.add_argument('metadata_yaml', help="Path to xengine_metadata YAML file")
    parser.add_argument('-n', '--non-verbose', action='store_true',
                        help="Emit the YAML header without the documentation comments (verbose=False).")


def show_file_format(args):
    # NOTE: the 'configs/example_asdf_header.yml' Makefile rule depends on
    # this command defaulting to verbose=True (no -n flag).
    # Do not flip that default without updating the Makefile rule.
    import tempfile
    from .utils import show_asdf as _show_asdf

    xmd = core.XEngineMetadata.from_yaml_file(args.metadata_yaml)
    if not xmd.beam_ids:
        raise RuntimeError(f"{args.metadata_yaml}: xengine_metadata has no beam_ids; "
                           f"cannot construct an AssembledFrame")

    # ntime=256 (one minichunk) is the smallest valid value -- keeps the binary
    # blob small since we don't actually look at it.
    #
    # We only read back the YAML header, so the data contents are irrelevant:
    # make_uninitialized() (no fill) is enough -- no need to randomize.

    frame = core.AssembledFrame.make_uninitialized(
        xmd, ntime=256, beam_id=xmd.beam_ids[0], time_chunk_index=0)

    # Random filename + try/finally so concurrent invocations don't race on
    # the same path and so we don't leave the binary blob behind on /dev/shm.
    fd, filename = tempfile.mkstemp(
        dir='/dev/shm', prefix='pirate_show_file_format_', suffix='.asdf')
    os.close(fd)
    try:
        frame.write_asdf(filename, verbose=not args.non_verbose)
        _show_asdf(filename)
    finally:
        os.remove(filename)


####################################   subcommand groups   ##########################################
#
# Five groups -- rpc, varmap, run, show, dev -- each a NESTED subparser level, so a command is
# 'pirate_frb run server' rather than a flat 'run_server'. _add_group() is the one place that
# knows how to make one; each parse_<group>() below just names its leaves.
#
# EVERY LEAF CARRIES ITS OWN HANDLER, via parser.set_defaults(func=...), and main() ends in a
# single args.func(args). That replaces what used to be a top-level if/elif chain over
# args.command plus one hand-written dispatch function per group -- which the old code already
# flagged as not scaling past three groups. The win is not brevity: a leaf's parser and its
# handler are now named in the SAME place, so adding a subcommand cannot half-land.


def _add_group(subparsers, name, help_text):
    """Add a group parser and return the subparsers object its leaves attach to.

    parser_class=_PirateParser propagates the terse invalid-choice errors down to the leaves;
    without it the leaves revert to argparse's default '(choose from ...)' wording. dest is
    per-group ('<name>_command') rather than shared, so a future group that wants to read its
    own subcommand name still can, and two groups can never collide.
    """
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    return parser.add_subparsers(dest=f"{name}_command", required=True, metavar="subcommand",
                                 parser_class=_PirateParser)


def parse_run(subparsers):
    """The 'run' group: long-running processes -- the real server, and the toy/offline rigs."""
    help_text = "Subcommand for running server/grouper/fake-xengine/etc (see run --help)"
    sub = _add_group(subparsers, "run", help_text)
    parse_run_server(sub)
    parse_run_toy_grouper(sub)
    parse_run_offline_dedisperser(sub)
    parse_run_toy_sifter(sub)
    parse_run_fake_xengine(sub)


def parse_show(subparsers):
    """The 'show' group: print something and exit. No side effects, no GPU work."""
    help_text = "Subcommand for printing information about config files or kernel registry (see show --help)"
    sub = _add_group(subparsers, "show", help_text)
    parse_show_asdf(sub)
    parse_show_file_format(sub)
    parse_show_dedisperser(sub)
    parse_show_hardware(sub)
    parse_show_kernels(sub)
    parse_show_random_config(sub)
    parse_show_xengine_metadata(sub)


def parse_dev(subparsers):
    """The 'dev' group: tools for working ON pirate, rather than for running it.

    The membership rule is "would an operator ever type this?" -- if not, it belongs here.
    That covers the makefile_helper.py maintenance utilities (make_subbands,
    random_kernels), the scratch/hardware-probe entry points (scratch, revisit_512gb,
    hwtest), the unit-test coverage report, and test_simpulse, which is a plotting rig
    rather than part of 'pirate_frb test'.

    'pirate_frb test' is deliberately NOT here: it is the one thing in this list that a
    non-developer is told to run, and it is the entry point every notes/*.md points at.
    """
    help_text = "Subcommand for developer utils: coverage, hardware probes, makefile helpers, scratch (see dev --help)"
    sub = _add_group(subparsers, "dev", help_text)
    parse_coverage(sub)
    parse_hwtest(sub)
    parse_scratch(sub)
    parse_make_subbands(sub)
    parse_random_kernels(sub)
    parse_test_simpulse(sub)
    parse_revisit_512gb(sub)


#########################################   rpc subcommands  ########################################


def parse_rpc(subparsers):
    """The 'rpc' group: clients that talk to a running FrbServer over gRPC."""
    help_text = "Subcommand for sending RPCs (e.g. status, file-writes) to a running FrbServer (see rpc --help)"
    sub = _add_group(subparsers, "rpc", help_text)
    parse_rpc_status(sub)
    parse_rpc_rand_write(sub)
    parse_rpc_start_stream(sub)
    parse_rpc_cancel_stream(sub)
    parse_rpc_show_streams(sub)


########################################   rpc status command  ######################################


def parse_rpc_status(subparsers):
    help_text = "Connect to FrbServer(s) and stream status + filenames"
    parser = subparsers.add_parser("status", help=help_text, description=help_text)
    parser.set_defaults(func=rpc_status)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS', help='Server address(es) (e.g. 127.0.0.1:6000)')


def rpc_status(args):
    from .run_rpc_status import run_rpc_status
    run_rpc_status(args.server_addresses)


######################################   rpc rand_write command  ####################################


def parse_rpc_rand_write(subparsers):
    help_text = "Send write_files RPC to FrbServer(s) with random beams/time range"
    parser = subparsers.add_parser("rand_write", help=help_text, description=help_text)
    parser.set_defaults(func=rpc_rand_write)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS', help='Server address(es) (e.g. 127.0.0.1:6000)')


def _rpc_rand_write_one(addr):
    """Send a write_files RPC to a single FrbServer."""

    import datetime
    from .rpc import FrbSearchClient

    client = FrbSearchClient(addr)
    atomic_print(f"[{addr}] Connected")

    try:
        # Get XEngine metadata to obtain beam IDs. client.beam_ids / xengine_metadata_yaml
        # raise RuntimeError until the server has received metadata.
        try:
            beam_ids = list(client.beam_ids)
        except RuntimeError:
            atomic_print(f"[{addr}] Error: metadata not yet available")
            return

        nbeams = len(beam_ids)
        atomic_print(f"[{addr}] Got metadata: {nbeams} beams, beam_ids={beam_ids}")

        # seq_per_chunk (fpga seqs per time chunk) = time_samples_per_chunk *
        # seq_per_frb_time_sample. The former comes from GetConfig, the latter
        # from the X-engine metadata. Used to convert our chunk range (derived
        # from the ring-buffer frame-id counters below) into the fpga-seq range
        # that write_files expects.
        seq_per_chunk = client.config.time_samples_per_chunk * client.xengine_metadata_yaml['seq_per_frb_time_sample']

        # Select random subset of beam IDs (1 to min(nbeams, 3)).
        n = random.randint(1, min(nbeams, 3))
        selected_beams = random.sample(beam_ids, n)
        atomic_print(f"[{addr}] Selected {n} beams: {selected_beams}")

        # Loop until we have frames available.
        while True:
            status = client.get_status()
            rb_reaped    = status.rb_reaped
            rb_processed = status.rb_processed

            # Convert frame IDs to time_chunk_index range.
            # frame_id = time_chunk_index * nbeams + beam_index
            # So time_chunk_index = frame_id // nbeams
            # rb_t0: first fully available time chunk (round up)
            # rb_t1: last available time chunk + 1 (round down)
            # Upper bound is rb_processed (not rb_end): frames in
            # [rb_processed, rb_end) are not rpc-writeable.
            rb_t0 = (rb_reaped    + nbeams - 1) // nbeams  # round up
            rb_t1 =  rb_processed // nbeams                # round down

            atomic_print(f"[{addr}] Status: rb_reaped={rb_reaped}, rb_processed={rb_processed} -> time_chunk_index range [{rb_t0}, {rb_t1})")

            if rb_t0 >= rb_t1:
                dt = pirate_pybind11.constants.default_print_cadence_sec
                atomic_print(f"[{addr}] No frames available yet, sleeping {dt}s...")
                time.sleep(dt)
                continue

            break

        # Choose random time range: rb_t0 <= t0 < t1 <= rb_t1, with 1 <= (t1-t0) <= 3.
        max_range = min(3, rb_t1 - rb_t0)
        range_size = random.randint(1, max_range)
        t0 = random.randint(rb_t0, rb_t1 - range_size)
        t1 = t0 + range_size

        atomic_print(f"[{addr}] Requesting time_chunk_index range [{t0}, {t1})")

        # Send write_files RPC. Convert the chunk range [t0, t1) to the
        # half-open fpga-seq range [t0*seq_per_chunk, t1*seq_per_chunk) that
        # write_files expects. Files land in {nfs_root}/rand_write_{date}_{time}/.
        acqdir = 'rand_write_' + datetime.datetime.now().strftime('%y_%m_%d_%H%M%S')
        filenames = client.write_files(
            beams=selected_beams,
            fpga_seq_start=t0 * seq_per_chunk,
            fpga_seq_end=t1 * seq_per_chunk,
            acqdir=acqdir
        )

        atomic_print(f"[{addr}] write_files returned {len(filenames)} filenames:")
        for fn in filenames:
            atomic_print(f"[{addr}]   {fn}")

    finally:
        client.close()


def rpc_rand_write(args):
    for addr in args.server_addresses:
        _rpc_rand_write_one(addr)


##############################   multi-server stream commands  ######################################
#
# rpc_start_stream / rpc_show_streams / rpc_cancel_stream each take one or more server
# addresses and treat the collection as a single "super-server": each FrbServer processes
# a DISJOINT set of beams, and the CLI routes (-b) / fans out (-B, -A) / loops so the user
# sees one logical stream namespace across all servers.


def _stream_clients(addresses):
    """Open an FrbSearchClient per address; returns a list of (addr, client).

    The caller is responsible for closing them (see the finally blocks below)."""
    from .rpc import FrbSearchClient
    return [(addr, FrbSearchClient(addr)) for addr in addresses]


def _rpc_error_str(e):
    """Human-readable message from a grpc.RpcError.

    Unary-call errors are grpc.Call and carry the server's message in .details();
    fall back to str()."""
    details = getattr(e, "details", None)
    return details() if callable(details) else str(e)


def parse_rpc_start_stream(subparsers):
    help_text = ("Send StartStream RPC to one or more FrbServers (write data to disk as it is "
                 "received). Multiple addresses act as one 'super-server'.")
    parser = subparsers.add_parser("start_stream", help=help_text, description=help_text)
    parser.set_defaults(func=rpc_start_stream)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS',
                        help='Server address(es) (e.g. 127.0.0.1:6000); multiple = one super-server')
    parser.add_argument('-s', '--stem', default='stream',
                        help='Filename stem; the CLI sets stream_name == acqdir == '
                             '"{stem}_{date}_{time}", shared across all servers '
                             '(default stem "stream", e.g. stream_26_07_07_143052)')
    parser.add_argument('-b', '--beam-id', type=int, action='append', metavar='BEAM_ID',
                        help='Beam id to stream (repeatable), routed to the server that owns it; '
                             'either -b or -B must be specified')
    parser.add_argument('-B', '--all-beams', action='store_true',
                        help='Stream all beams; starts a stream on every server with its full beam list')
    parser.add_argument('-d', type=float, default=None, metavar='DURATION_SECONDS', dest='duration',
                        help='Stream duration in seconds; either -d or -D must be specified')
    parser.add_argument('-D', '--no-duration', action='store_true',
                        help='Run indefinitely (fpga_seq_end = 2^63 - 1)')


def rpc_start_stream(args):
    import datetime
    import grpc

    if bool(args.beam_id) == bool(args.all_beams):
        raise RuntimeError("rpc_start_stream: specify exactly one of -b/--beam-id or -B/--all-beams")
    if (args.duration is not None) == bool(args.no_duration):
        raise RuntimeError("rpc_start_stream: specify exactly one of -d or -D/--no-duration")
    if (args.duration is not None) and (args.duration <= 0):
        raise RuntimeError(f"rpc_start_stream: duration must be positive (got {args.duration})")

    clients = _stream_clients(args.server_addresses)

    try:
        # Phase 1: query every server up front. show_streams() fails cleanly if a
        # server hasn't locked onto the X-engine stream yet, and returns its
        # current fpga position (for -d) and its beam list (for -B / routing). We
        # need ALL beam lists before routing -b, so any failure here is fatal --
        # nothing has been started, so no partial super-server stream is left behind.
        infos = []   # list of (addr, client, show_streams_response)
        for addr, client in clients:
            try:
                infos.append((addr, client, client.show_streams()))
            except grpc.RpcError as e:
                raise RuntimeError(
                    f"rpc_start_stream: server {addr} is not ready ({_rpc_error_str(e)}); "
                    "no streams were started") from e

        # Phase 2: decide each target server's beam subset -> list of (addr, client, ss, beams).
        if args.all_beams:
            # -B: every server streams its own full beam list.
            targets = [(addr, client, ss, list(ss.beam_ids)) for (addr, client, ss) in infos]
        else:
            # -b: route each requested beam to the server that owns it (servers
            # process disjoint beam sets), then group by owning server preserving
            # the order beams were given on the command line.
            beam_to_server = {b: (addr, client, ss)
                              for (addr, client, ss) in infos for b in ss.beam_ids}
            missing = [b for b in args.beam_id if b not in beam_to_server]
            if missing:
                raise RuntimeError(
                    f"rpc_start_stream: beam id(s) {missing} are not processed by any of the given "
                    f"servers (available beams: {sorted(beam_to_server)})")
            grouped = {}   # addr -> (addr, client, ss, [beams])
            for b in args.beam_id:
                addr, client, ss = beam_to_server[b]
                grouped.setdefault(addr, (addr, client, ss, []))[3].append(b)
            targets = list(grouped.values())

        # Generate ONE stream_name/acqdir, shared across all target servers, so a
        # multi-server event lands in a single acqdir. (If each server defaulted
        # stream_name=None, each would generate a different timestamp.) The CLI
        # keeps stream_name == acqdir == "{stem}_{date}_{time}"; the date format
        # mirrors FrbSearchClient.start_stream's default (with a caller-chosen stem).
        stream_name = args.stem + '_' + datetime.datetime.now().strftime('%y_%m_%d_%H%M%S')

        # Phase 3: start one stream per target server (fpga_seq_end is per-server,
        # since each server has its own current_fpga_seq).
        had_error = False
        for (addr, client, ss, beams) in targets:
            if args.no_duration:
                fpga_seq_end, end_str = None, "indefinite"     # "run indefinitely"
            else:
                dt_ns_per_seq = client.xengine_metadata_yaml['dt_ns_per_seq']
                fpga_seq_end = ss.current_fpga_seq + round(args.duration * 1.0e9 / dt_ns_per_seq)
                end_str = str(fpga_seq_end)
            try:
                sn, acqdir = client.start_stream(
                    beams, stream_name=stream_name, acqdir=stream_name,
                    fpga_seq_end=fpga_seq_end,   # fpga_seq_start defaults to 0 ("start asap")
                )
            except grpc.RpcError as e:
                had_error = True
                atomic_print(f"[{addr}] ERROR: {_rpc_error_str(e)}", fd=2)
                continue
            atomic_print(f"[{addr}] started stream stream_name={sn!r}")
            atomic_print(f"[{addr}]   acqdir = {acqdir!r}")
            atomic_print(f"[{addr}]   beam_ids = {beams}")
            atomic_print(f"[{addr}]   fpga_seq range = [0, {end_str})")

        if had_error:
            sys.exit(1)
    finally:
        for _, client in clients:
            client.close()


##################################   rpc cancel_stream command  #####################################


def parse_rpc_cancel_stream(subparsers):
    help_text = ("Send CancelStream RPC to one or more FrbServers. Multiple addresses act as one "
                 "'super-server' (cancels loop over all servers).")
    parser = subparsers.add_parser("cancel_stream", help=help_text, description=help_text)
    parser.set_defaults(func=rpc_cancel_stream)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS',
                        help='Server address(es) (e.g. 127.0.0.1:6000); multiple = one super-server')
    parser.add_argument('-a', '--stream-name', default=None, metavar='STREAM_NAME',
                        help='Cancel the stream with this stream_name, on every server that has it')
    parser.add_argument('-A', '--all', action='store_true', dest='cancel_all',
                        help='Cancel all active streams on every server')


def rpc_cancel_stream(args):
    import grpc
    from .rpc.grpc import frb_search_pb2

    if bool(args.stream_name) == bool(args.cancel_all):
        raise RuntimeError("rpc_cancel_stream: specify exactly one of -a/--stream-name or -A/--all")

    clients = _stream_clients(args.server_addresses)
    had_error = False

    try:
        if args.cancel_all:
            # -A: cancel every active stream on every server.
            for addr, client in clients:
                try:
                    n = client.cancel_stream(cancel_all=True)
                    atomic_print(f"[{addr}] cancelled {n} stream(s)")
                except grpc.RpcError as e:
                    had_error = True
                    atomic_print(f"[{addr}] ERROR: {_rpc_error_str(e)}", fd=2)
        else:
            # -a NAME: cancel the named stream wherever it is ACTIVE. We check
            # show_streams() first (rather than catching a per-server "not found"
            # error) so a name that exists on no server is a single clear error
            # rather than a pile of per-server failures.
            name = args.stream_name
            found = False
            for addr, client in clients:
                try:
                    ss = client.show_streams()
                except grpc.RpcError as e:
                    had_error = True
                    atomic_print(f"[{addr}] ERROR: {_rpc_error_str(e)}", fd=2)
                    continue
                if not any(i.args.stream_name == name and
                           i.status == frb_search_pb2.STREAM_STATUS_ACTIVE
                           for i in ss.streams):
                    continue
                found = True
                try:
                    n = client.cancel_stream(stream_name=name)
                    atomic_print(f"[{addr}] cancelled {n} stream(s) named {name!r}")
                except grpc.RpcError as e:
                    had_error = True
                    atomic_print(f"[{addr}] ERROR: {_rpc_error_str(e)}", fd=2)
            if not found and not had_error:
                raise RuntimeError(
                    f"rpc_cancel_stream: no server has an active stream named {name!r} "
                    f"(servers: {', '.join(args.server_addresses)})")
    finally:
        for _, client in clients:
            client.close()

    if had_error:
        sys.exit(1)


###################################   rpc show_streams command  #####################################


def parse_rpc_show_streams(subparsers):
    help_text = ("Send ShowStreams RPC to one or more FrbServers and print the responses. "
                 "Multiple addresses act as one 'super-server' (loops over all servers).")
    parser = subparsers.add_parser("show_streams", help=help_text, description=help_text)
    parser.set_defaults(func=rpc_show_streams)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS',
                        help='Server address(es) (e.g. 127.0.0.1:6000); multiple = one super-server')


def rpc_show_streams(args):
    import datetime
    import grpc
    from .rpc.grpc import frb_search_pb2

    ACTIVE = frb_search_pb2.STREAM_STATUS_ACTIVE
    INDEF = 2**63 - 1   # fpga_seq_end sentinel for "run indefinitely"

    def fmt_time(unix_ns):
        if unix_ns == 0:
            return "-"
        return datetime.datetime.fromtimestamp(unix_ns * 1.0e-9).strftime('%Y-%m-%d %H:%M:%S')

    def fmt_duration(sec):
        sec = max(0, int(round(sec)))
        if sec < 60:
            return f"{sec}s"
        m, s = divmod(sec, 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    def print_server(addr, ss, dt_ns_per_seq):
        n_listed_inactive = sum(1 for i in ss.streams if i.status != ACTIVE)
        atomic_print(f"[{addr}] current_fpga_seq = {ss.current_fpga_seq}")
        atomic_print(f"[{addr}] beam_ids = {list(ss.beam_ids)}")
        atomic_print(f"[{addr}] num_deactivated_streams = {ss.num_deactivated_streams}"
                     f" ({n_listed_inactive} retained in history)")

        if not ss.streams:
            atomic_print(f"[{addr}] no active or recently-deactivated streams")

        for info in ss.streams:
            a = info.args
            # "STREAM_STATUS_ACTIVE" -> "active", etc.
            status = frb_search_pb2.StreamStatus.Name(info.status)
            status = status.removeprefix('STREAM_STATUS_').lower()
            if (info.status != ACTIVE) and info.cancelled:
                status += " (cancelled)"
            end_str = "indefinite" if (a.fpga_seq_end == INDEF) else str(a.fpga_seq_end)
            # For an active, finite-duration stream, show the estimated wall-clock
            # time until its end fpga_seq is processed. Data flows in real time, so
            # (remaining fpga-seqs) * dt_ns_per_seq ~= seconds left. This is
            # independent of fpga_seq_start, so it makes no assumption about how
            # the caller set the range's start.
            remaining = ""
            if (info.status == ACTIVE) and (a.fpga_seq_end != INDEF):
                remaining_sec = (a.fpga_seq_end - ss.current_fpga_seq) * dt_ns_per_seq * 1.0e-9
                remaining = f" (~{fmt_duration(remaining_sec)} remaining)"
            atomic_print(f"[{addr}] stream stream_name={a.stream_name!r}:")
            atomic_print(f"[{addr}]   status = {status}")
            atomic_print(f"[{addr}]   acqdir = {a.acqdir!r}")
            atomic_print(f"[{addr}]   beam_ids = {list(a.beam_ids)}")
            atomic_print(f"[{addr}]   fpga_seq range = [{a.fpga_seq_start}, {end_str}){remaining}")
            atomic_print(f"[{addr}]   started = {fmt_time(info.started_at_unix_ns)}, "
                         f"deactivated = {fmt_time(info.deactivated_at_unix_ns)}")
            atomic_print(f"[{addr}]   files: queued = {info.num_files_queued}, "
                         f"written = {info.num_files_written}, errored = {info.num_files_errored}")

    clients = _stream_clients(args.server_addresses)
    had_error = False

    try:
        for i, (addr, client) in enumerate(clients):
            if i > 0:
                atomic_print("\n")   # blank line between per-server blocks
            try:
                ss = client.show_streams()
                # dt_ns_per_seq (for "time remaining") is only needed when some
                # active stream has a finite end; fetch it lazily to skip an extra
                # RPC otherwise. client.xengine_metadata_yaml is cached and is
                # available here -- show_streams() and the metadata both require
                # the server to have received X-engine metadata.
                need_dt = any((s.status == ACTIVE) and (s.args.fpga_seq_end != INDEF)
                              for s in ss.streams)
                dt = client.xengine_metadata_yaml['dt_ns_per_seq'] if need_dt else None
                print_server(addr, ss, dt)
            except grpc.RpcError as e:
                had_error = True
                atomic_print(f"[{addr}] ERROR: {_rpc_error_str(e)}", fd=2)
    finally:
        for _, client in clients:
            client.close()

    if had_error:
        sys.exit(1)


#####################################   dev random_kernels command  #####################################


def parse_random_kernels(subparsers):
    help_text = "A utility for maintaining makefile_helper.py"
    parser = subparsers.add_parser("random_kernels", help=help_text, description=help_text)
    parser.set_defaults(func=random_kernels)
    parser.add_argument('-n', type=int, default=20, help='Number of random kernels to print (default 20)')
    parser.add_argument('--pf', action='store_true', help='Print random PeakFinder kernel params')
    parser.add_argument('--cdd2', action='store_true', help='Print random CoalescedDdKernel2 kernel params')
    parser.add_argument('--pfwr', action='store_true', help='Print random PfWeightReader kernel params')


def random_kernels(args):
    import numpy
    
    flags = [ 'pf', 'cdd2', 'pfwr' ]
    nflags = sum(1 if getattr(args, x) else 0 for x in flags)
    
    if nflags != 1:
        atomic_print("Error: precisely one of --pf, --cdd2, --pfwr must be specified", fd=2)
        atomic_print("  --pf     Print random PeakFinder kernel params", fd=2)
        atomic_print("  --cdd2   Print random CoalescedDdKernel2 kernel params", fd=2)
        atomic_print("  --pfwr   Print random PfWeightReader kernel params", fd=2)
        sys.exit(2)

    randi = lambda *a: int(numpy.random.randint(*a))

    if args.pf:
        atomic_print('# (dtype, subband_counts, K, Wmax, Dcore, Dout, Tinner)')

        for _ in range(args.n):
            nbits = 32 // randi(1,3)
            Dcore = 32 // nbits
            Dout = Dcore
            Tinner = 1

            for _ in range(5):
                n = randi(4)
                if n == 0:
                    Tinner *= 2
                if n == 1:
                    Dcore *= 2
                if 1 <= n <= 2:
                    Dout *= 2
            
            Wmax = 2**randi(5)
            subband_counts = core.FrequencySubbands.make_random_subband_counts()
            K = randi(5)   # extra-DM bits; the standalone kernel accepts any K >= 0
            atomic_print(f"('fp{nbits}', {list(subband_counts)}, {K}, {Wmax}, {Dcore}, {Dout}, {Tinner})")

    if args.cdd2:
        # NOTE no Dcore/Dout: a cdd2 kernel's Dout is pinned to 2^dd_rank1 and its Dcore to
        # min(Dout,8). See cuda_generator.cdd2_dout(). (The --pf branch below still prints
        # both, since standalone PeakFinder kernels have an independent Dout.)
        #
        # EVERY ROW PRINTED HERE IS BUILDABLE. A cdd2 row has to satisfy constraints that
        # live in three different places -- the Dedisperser/PeakFinder constructors,
        # FrequencySubbands.restrict_subband_counts(), and the Dout invariant -- so this
        # draws within cuda_generator's bounds and then checks the result with
        # cuda_generator.check_cdd2_row(), the same function makefile_helper.py calls on the
        # rows it consumes. Do not re-derive a bound here; ask cuda_generator for it.
        from .cuda_generator import max_cdd2_tinner, check_cdd2_row

        atomic_print("# (dtype, dd_rank, Wmax, Tinner, subband_counts, num_early_triggers)")

        for _ in range(args.n):
            nbits = 32 // randi(1,3)
            dtype = f'fp{nbits}'
            Wmax = 2**randi(5)

            # Currently, cdd2 assumes dd_rank >= 3
            dd_rank_max = randi(3,9)
            dd_rank_min = max(dd_rank_max-1, 3)

            # PeakFinder requires Dout*Tinner <= 32*SW, and Dout follows dd_rank, so the
            # bound is tightest at the LARGEST dd_rank in the row (fp32 dd_rank 8 admits
            # only Tinner <= 2).
            # max_cdd2_tinner() returns a power of two, so bit_length() is log2 + 1.
            Tinner = 2**randi(max_cdd2_tinner(dtype, dd_rank_max).bit_length())

            # subband_counts is a CONFIG's frequency_subband_counts, so it must satisfy
            # pf_rank <= dd_rank1 at the SMALLEST dd_rank of the row (every larger one is
            # implied), and must survive restriction by every et_level emitted.
            #
            # DRAW pf_rank, rather than always taking the maximum. A kernel with
            # pf_rank < dd_rank1 emits (dd_rank1 - pf_rank) "extra DM" bits per warp, and
            # taking the maximum here means that code path is only ever reached via early
            # triggers -- so xdm_rank 3 and 4 were unreachable, and half the kernels drawn
            # had xdm_rank 0. (Short subband_counts became legal after this function was
            # written, and it was never updated.)
            #
            # Safe at every early-trigger level: restrict_subband_counts() drops pf_rank by
            # one per level while dd_rank1 = ceil(dd_rank/2) drops by one every OTHER level,
            # so pf_rank <= dd_rank1 at et_level 0 implies it everywhere above.
            dd_rank1_min = (dd_rank_min + 1) // 2
            subband_counts = core.FrequencySubbands.make_random_subband_counts(randi(0, dd_rank1_min + 1))

            for dd_rank in range(dd_rank_min, dd_rank_max+1):
                # The early-trigger levels of a row are 0..num_early_triggers, CONTIGUOUS,
                # exactly as for a DedispersionConfig primary tree. So find the largest N
                # for which every level up to N is usable, by walking up and stopping at the
                # first failure -- the same thing DedispersionConfig::make_random() does.
                # (restrict_subband_counts() is not monotone in et_level, so this is not the
                # same as the largest individually-usable level.)
                net_max = 0
                while (net_max + 1 <= dd_rank - 3) and \
                      core.FrequencySubbands.can_early_trigger(list(subband_counts), net_max + 1):
                    net_max += 1

                num_early_triggers = randi(0, net_max + 1)

                # Belt and braces: the draws above are meant to guarantee this, so a failure
                # here is a bug in them rather than a bad roll.
                err = check_cdd2_row(dtype, dd_rank, Wmax, Tinner, list(subband_counts),
                                     num_early_triggers)
                assert err is None, f'random_kernels --cdd2 generated an unbuildable row: {err}'

                s = '     # continuation' if (dd_rank > dd_rank_min) else ''
                atomic_print(f"('{dtype}', {dd_rank}, {Wmax}, {Tinner}, {list(subband_counts)},"
                             f" {num_early_triggers}),{s}")

    if args.pfwr:
        atomic_print('# (dtype, subband_counts, K, Dcore, P, Tinner)')
        
        for _ in range(args.n):
            nbits = 32 // randi(1,3)
            rank = randi(2,5)
            Tinner_log = randi(6)
            Dcore_log = randi(6-Tinner_log) + (32//nbits) - 1
            P = randi(1,15)
            subband_counts = core.FrequencySubbands.make_random_subband_counts()
            K = randi(5)   # extra-DM bits
            atomic_print(f"('fp{nbits}', {tuple(subband_counts)}, {K}, {2**Dcore_log}, {P}, {2**Tinner_log})")


######################################  run server command  #####################################


def parse_run_server(subparsers):
    help_text = "Start FRB server(s) from an frb_server .yml file and a dedispersion .yml file"
    parser = subparsers.add_parser("server", help=help_text, description=help_text)
    parser.set_defaults(func=run_server_command)
    parser.add_argument('server_config', help='Path to FrbServer YAML config file')
    parser.add_argument('dedispersion_config', help='Path to DedispersionConfig YAML file')
    parser.add_argument('-d', '--delay', type=float, default=0.0, metavar='SECONDS',
                        help='Artificial per-frame delay in the processing thread '
                             '(seconds; default 0). Used to simulate slow GPU work '
                             'for testing FakeXEngine pacing.')
    parser.add_argument('-G', '--no-grouper', action='store_true',
                        help='Disable FrbGrouper RPC even if grouper_ip_addrs '
                             'is set in the config (GpuDedisperser runs with '
                             'num_consumers=0).')
    parser.add_argument('-D', '--no-dedispersion', action='store_true',
                        help='Skip ALL GPU work in the processing thread: data '
                             'is not even copied host->GPU, and no dequantization '
                             'or dedispersion kernels run. The '
                             'receive/assemble/ringbuf path still runs in full '
                             '(the dedisperser is still built, just never fed). '
                             'Implies --no-grouper. Infrequently used corner case.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress the per-chunk "FrbServer: beamset=..." line '
                             '(printed once per fully-processed time chunk).')


def run_server_command(args):
    from .run_server import run_server
    run_server(args.server_config, args.dedispersion_config,
               processing_delay_sec=args.delay,
               no_grouper=args.no_grouper,
               no_dedispersion=args.no_dedispersion,
               quiet=args.quiet)


######################################  run toy_grouper command  #####################################


def parse_run_toy_grouper(subparsers):
    help_text = "Toy FrbGrouper consumer(s): per-chunk peak SNR + argmax, optionally reported to a sifter"
    parser = subparsers.add_parser("toy_grouper", help=help_text, description=help_text)
    parser.set_defaults(func=run_toy_grouper_command)
    parser.add_argument('grouper_addrs', nargs='+', metavar='grouper_addr',
                        help="FrbGrouper listen address(es) 'ip:port' (e.g. 127.0.0.1:7000). "
                             "With more than one, each grouper runs in its own child "
                             "subprocess; if any child exits, the parent and all siblings exit.")
    parser.add_argument('-d', '--delay', type=float, default=0.0, metavar='SECONDS',
                        help="Artificial per-chunk delay (seconds) inserted into the grouper "
                             "loop, e.g. -d 0.001 for a 1 ms delay (default: 0, no delay).")
    parser.add_argument('-t', '--snr-threshold', type=float, default=10.0, metavar='SNR_THRESHOLD',
                        help="Emit one event per chunk per beam whose peak SNR exceeds this "
                             "threshold (default: 10).")
    parser.add_argument('--histogram', metavar='STEM',
                        help="Write histograms of steady-state SNR values (all values, plus "
                             "per-(beam, chunk) maxes; warmup values are excluded) to 'STEM.pkl', "
                             "plus a plot 'STEM.pdf', upon termination. With multiple groupers, "
                             "the i-th grouper writes 'STEM<i>.pkl' / 'STEM<i>.pdf' (e.g. "
                             "hist1.pkl, hist2.pkl, ...) so the filenames don't collide.")
    # Exactly one of -s/-S is required.
    sifter_group = parser.add_mutually_exclusive_group(required=True)
    sifter_group.add_argument('-s', '--sifter', metavar='SIFTER_ADDR',
                              help="Report to the FrbSifter at this 'ip:port' (e.g. 127.0.0.1:7100).")
    sifter_group.add_argument('-S', '--no-sifter', action='store_true',
                              help="Run without a sifter (don't send any sifter RPCs).")


def run_toy_grouper_command(args):
    from .run_toy_grouper import run_toy_grouper

    # Fail fast (before launching anything): a '.' in the stem means the caller
    # almost certainly passed a filename. (Checked here, not just in
    # run_toy_grouper(), so a multi-grouper run errors once in the parent with
    # the original stem, rather than once per child with an index-mangled one.)
    if args.histogram and ('.' in args.histogram):
        raise ValueError(f"run_toy_grouper: --histogram takes a filename STEM, got {args.histogram!r} "
                         f"(contains a '.', looks like a full filename). The '.pkl' suffix is appended "
                         f"automatically, with a per-grouper index if there are multiple groupers.")
    # A single grouper runs in this process (no subprocess indirection). With more
    # than one, launch each in its own child process (re-invoking this CLI with a
    # single address), and fail-fast: if any child exits, run_processes() stops the
    # rest. A fresh process (not fork) avoids CUDA-after-fork hazards.
    if len(args.grouper_addrs) == 1:
        run_toy_grouper(args.grouper_addrs[0], sifter_addr=args.sifter, delay=args.delay,
                        snr_threshold=args.snr_threshold, histogram_stem=args.histogram)
        return
    from .utils import run_processes
    # Re-pass exactly one of the (mutually-exclusive, required) sifter flags.
    sifter_flag = ['--sifter', args.sifter] if (args.sifter is not None) else ['--no-sifter']
    base = [sys.executable, '-m', 'pirate_frb', 'run', 'toy_grouper', *sifter_flag,
            '--delay', str(args.delay), '--snr-threshold', str(args.snr_threshold)]
    # Each child gets a distinct histogram stem (STEM1, STEM2, ...), so the
    # '<stem>.pkl' output filenames don't collide.
    cmds = []
    for i, addr in enumerate(args.grouper_addrs, start=1):
        hist_flag = ['--histogram', f'{args.histogram}{i}'] if args.histogram else []
        cmds.append(base + hist_flag + [addr])
    rc = run_processes(cmds)
    if rc:
        sys.exit(rc)


######################################  run offline_dedisperser command  #####################################


def parse_run_offline_dedisperser(subparsers):
    help_text = "Toy offline dedispersion: per-chunk peak SNR over an acqdir of .asdf frames"
    parser = subparsers.add_parser("offline_dedisperser", help=help_text, description=help_text)
    parser.set_defaults(func=run_offline_dedisperser_command)
    parser.add_argument("acqdir",
                        help="acqdir of frame_b(BEAM)_t(CHUNK).asdf files")
    parser.add_argument("config",
                        help="dedispersion config yaml")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="only process the first N chunks of each beam (default: all)")


def run_offline_dedisperser_command(args):
    from .run_offline_dedisperser import run_offline_dedisperser
    run_offline_dedisperser(args.acqdir, args.config, max_chunks=args.max_chunks)


######################################  run toy_sifter command  #####################################


def parse_run_toy_sifter(subparsers):
    help_text = "Toy FrbSifter gRPC server: print a one-line summary of each received message"
    parser = subparsers.add_parser("toy_sifter", help=help_text, description=help_text)
    parser.set_defaults(func=run_toy_sifter_command)
    parser.add_argument('addr', metavar='ADDR',
                        help="Listen address 'ip:port' for the sifter gRPC server "
                             "(e.g. 127.0.0.1:7100; use [::]:7100 or 0.0.0.0:7100 for all interfaces).")


def run_toy_sifter_command(args):
    from .run_toy_sifter import run_toy_sifter
    run_toy_sifter(args.addr)


######################################  run fake_xengine command  #####################################


def parse_run_fake_xengine(subparsers):
    help_text = "Send fake X-engine data to one or more running FrbServers"
    parser = subparsers.add_parser("fake_xengine", help=help_text, description=help_text)
    parser.set_defaults(func=run_fake_xengine_command)
    parser.add_argument('rpc_addrs', nargs='+', metavar='RPC_ADDR',
                        help='One or more "ip:port" strings (one per server, matching the config\'s rpc_ip_addrs)')
    parser.add_argument('-w', '--workers', type=int, default=128,
                        help='Number of worker threads per FakeXEngine (default 128)')
    parser.add_argument('-P', '--unpaced', action='store_true',
                        help='Disable pacing -- send chunks as fast as possible '
                             '(default: pace to stay <=4 chunks ahead of server)')
    parser.add_argument('-N', '--unnormalized', action='store_true',
                        help='Send unnormalized data -- leave scales/offsets '
                             'arbitrary (default: calibrate them to the per-zone '
                             'noise variance)')
    parser.add_argument('-G', '--non-gaussian', action='store_true',
                        help='Fill int4 data with uniform noise over [-8,+7] '
                             '(default: simulated Gaussian noise clamped to '
                             '[-7,+7])')
    parser.add_argument('-j', '--send-junk', action='store_true',
                        help='Randomize+send only the first chunk; send all-zero '
                             'junk for every subsequent chunk (skips per-chunk '
                             'randomization)')
    parser.add_argument('-f', '--frbs', action='store_true',
                        help='Inject simulated FRBs (parameters derived from the '
                             'server GetConfig: max DM, base-tree width, and the '
                             'frequency subbands). Prints one line per injected '
                             'FRB. Incompatible with -N, -G, and -j.')
    parser.add_argument('-g', '--gap', metavar='GAP_SEC', type=float, default=0.0,
                        help='Extra padding (seconds) between consecutive simulated '
                             'FRBs on a beam (default 0). Requires -f.')
    parser.add_argument('-s', '--sifter', metavar='SIFTER_ADDR', default=None,
                        help='Send the simulated FRB events (from_simulator=True) to '
                             'an FrbSifter at this "ip:port". Requires -f.')
    parser.add_argument('--frb-snr', metavar='SNR', type=float, default=30.0, dest='frb_snr',
                        help='Matched-filter SNR of injected simulated FRBs (default 30). '
                             'Requires -f.')


def run_fake_xengine_command(args):
    # FRB injection requires normalized + gaussian data (SimulatedFrameFactory
    # enforces this), and randomizes every chunk -- so it is incompatible with
    # -N/--unnormalized, -G/--non-gaussian, and -j/--send-junk.
    if args.frbs:
        bad = []
        if args.unnormalized: bad.append('-N/--unnormalized')
        if args.non_gaussian: bad.append('-G/--non-gaussian')
        if args.send_junk:    bad.append('-j/--send-junk')
        if bad:
            atomic_print(f"Error: -f/--frbs is incompatible with {', '.join(bad)} "
                         f"(FRB injection requires normalized + gaussian data and "
                         f"randomizes every chunk).", fd=2)
            sys.exit(2)

    # Sending events to a sifter only makes sense when FRBs are being simulated.
    if args.sifter is not None and not args.frbs:
        atomic_print("Error: -s/--sifter requires -f/--frbs (there are no events to send "
                     "without FRB simulation).", fd=2)
        sys.exit(2)

    # An inter-FRB gap only has meaning when FRBs are being simulated.
    if args.gap != 0.0 and not args.frbs:
        atomic_print("Error: -g/--gap requires -f/--frbs (there are no FRBs to space "
                     "without FRB simulation).", fd=2)
        sys.exit(2)
    if args.gap < 0.0:
        atomic_print("Error: -g/--gap must be >= 0 seconds.", fd=2)
        sys.exit(2)

    # An FRB SNR only has meaning when FRBs are being simulated.
    if args.frb_snr != 30.0 and not args.frbs:
        atomic_print("Error: --frb-snr requires -f/--frbs (there are no FRBs to inject "
                     "without FRB simulation).", fd=2)
        sys.exit(2)

    from .run_fake_xengine import run_fake_xengine
    run_fake_xengine(args.rpc_addrs, nworkers=args.workers,
                     paced=not args.unpaced, normalized=not args.unnormalized,
                     gaussian=not args.non_gaussian,
                     send_junk=args.send_junk, simulate_frbs=args.frbs,
                     sifter_addr=args.sifter, frb_gap_sec=args.gap, frb_snr=args.frb_snr)

####################################################################################################


class _PirateParser(argparse.ArgumentParser):
    """ArgumentParser variant with terser invalid-choice errors.

    Swallows argparse's auto-appended '(choose from {...})' in invalid-choice
    errors and points the user at --help instead. Pairs with metavar='command'
    on add_subparsers() so the run-on choices listing also disappears from
    --help / usage."""
    def error(self, message):
        # Strip the "(choose from ...)" suffix argparse appends on
        # invalid-subcommand errors. Wording is fragile across Python
        # versions; falls through harmlessly if argparse changes it.
        message = re.sub(r" \(choose from .*\)$", "", message)
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        sys.stderr.write(f"For a list of all commands, see '{self.prog} --help'.\n")
        sys.exit(2)


def get_parser():
    """
    Create and return the argument parser for pirate_frb.

    This function is separate from main() so that sphinx-argparse can
    introspect the parser without actually parsing command-line arguments.
    """
    parser = _PirateParser(description="pirate_frb command-line driver (use --help for more info)")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")

    # Declaration order here is what the reader of 'pirate_frb --help' sees, and what the
    # docs build walks (docs/source/conf.py) to lay out the CLI reference. Groups first,
    # then the handful of commands that are not in one.
    parse_run(subparsers)
    parse_rpc(subparsers)
    parse_show(subparsers)
    parse_varmap(subparsers)
    parse_dev(subparsers)

    parse_test(subparsers)
    parse_time(subparsers)
    parse_time_dedisperser(subparsers)

    return parser


def _install_atomic_hooks():
    """Route uncaught-exception output through atomic_print().

    The shutdown cascade is a designed feature (Ctrl-C on the sifter is
    supposed to end with an exception in every upstream process), so
    tracebacks are our most common multi-line output. The interpreter's
    default hooks emit them in many small writes, which interleave both with
    other threads and -- since run_toy_grouper's children share the parent's
    stderr -- with other PROCESSES. Formatting each traceback and emitting it
    in a single write fixes both.

    Child subprocesses run 'python -m pirate_frb ...', so they install these
    hooks themselves.
    """
    def _format(exc_type, exc_value, exc_tb, prefix=""):
        return prefix + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    def _excepthook(exc_type, exc_value, exc_tb):
        atomic_print(_format(exc_type, exc_value, exc_tb), fd=2)

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return   # parity with the default threading.excepthook
        name = args.thread.name if args.thread is not None else "<unknown>"
        atomic_print(_format(args.exc_type, args.exc_value, args.exc_traceback,
                             prefix=f"Exception in thread {name}:\n"), fd=2)

    def _unraisable_hook(unraisable):
        # Errors in __del__ / GC callbacks. The object can be mid-teardown, so
        # guard its repr().
        err = unraisable.err_msg or "Exception ignored in"
        try:
            obj = repr(unraisable.object)
        except Exception:
            obj = "<object repr() failed>"
        atomic_print(_format(unraisable.exc_type, unraisable.exc_value,
                             unraisable.exc_traceback, prefix=f"{err}: {obj}\n"), fd=2)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook
    sys.unraisablehook = _unraisable_hook


def main():
    _install_atomic_hooks()
    seed_rngs(DEFAULT_SEED)   # 'pirate_frb test' re-seeds from its own --seed / -r flags

    parser = get_parser()
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # Every leaf parser set its own handler (parser.set_defaults(func=...) in each parse_*),
    # so there is no dispatch table to keep in step with the parsers. A group parser sets no
    # 'func' of its own, but its subparsers are required=True, so argparse has already errored
    # out before we get here if the user typed a bare group name.
    #
    # The getattr guard is for the one way this can go wrong: a new parse_*() that adds a
    # subcommand but forgets its set_defaults(func=...). Without it that is a bare
    # AttributeError on 'args.func' with no hint of the cause. It cannot be triggered by
    # anything a USER types.
    func = getattr(args, "func", None)
    if func is None:
        atomic_print(f"Internal error: subcommand '{parser.prog} {' '.join(sys.argv[1:])}'"
                     " has no handler. Its parse_*() function is missing a"
                     " parser.set_defaults(func=...) call.", fd=2)
        sys.exit(2)
    func(args)


if __name__ == '__main__':
    main()
