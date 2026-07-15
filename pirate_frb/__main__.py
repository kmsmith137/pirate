import os
import re
import sys
import time
import shlex
import random
import textwrap
import argparse

import argcomplete
import ksgpu

from . import pirate_pybind11
from . import casm
from . import chime
from . import kernels
from . import loose_ends
from . import core
from . import tests
from . import slow_avar
from .fast_avar import PfAvarApproximation, test_fast_avar

from .slow_avar import SparseTile, SparseTileTriple, SparseTilePerM, PfVarianceConvolver, PfVariance

from . import (
    DedispersionConfig,
    DedispersionPlan,
    GpuDedisperser,
)

from .Hardware import Hardware
from .Hwtest import Hwtest
from .HwtestSender import HwtestSender
from .yaml_utils import indent_dedispersion_plan_comments, align_inline_comments


#########################################   test command  ##########################################


def parse_test(subparsers):
    help_text = "Run unit tests (use flags to select specific tests)"
    parser = subparsers.add_parser("test", help=help_text, description=help_text)
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for tests (default 0)")
    parser.add_argument('-n', '--niter', type=int, default=100, help="Number of unit test iterations (default 100)")
    parser.add_argument('--rt', action='store_true', help='Runs ReferenceTree and ReferenceLagbuf tests')
    parser.add_argument('--pfwr', action='store_true', help='Runs PfWeightReaderMicrokernel.test_random()')
    parser.add_argument('--pfom', action='store_true', help='Runs PfOutputMicrokernel.test_random()')
    parser.add_argument('--gldk', action='store_true', help='Runs GpuLaggedDownsamplingKernel.test_random()')
    parser.add_argument('--gddk', action='store_true', help='Runs GpuDedispersionKernel.test_random()')
    parser.add_argument('--gpfk', action='store_true', help='Runs GpuPeakFindingKernel.test_random()')
    parser.add_argument('--grck', action='store_true', help='Runs GpuRingbufCopyKernel.test_random()')
    parser.add_argument('--gtgk', action='store_true', help='Runs GpuTreeGriddingKernel.test_random()')
    parser.add_argument('--gdqk', action='store_true', help='Runs GpuDequantizationKernel.test_random()')
    parser.add_argument('--cdd2', action='store_true', help='Runs CoalescedDdKernel2.test_random()')
    parser.add_argument('--casm', action='store_true', help='Runs some casm tests')
    parser.add_argument('--zomb', action='store_true', help='Runs "zombie" tests (code that I wrote during protoyping that may never get used)')
    parser.add_argument('--dd', action='store_true', help='Runs GpuDedisperser.test_random()')
    parser.add_argument('--avar', action='store_true', help='Runs tests related to analytic variance')
    parser.add_argument('--chime', action='store_true', help='Runs test_chime_frb_{beamform,upchan}()')
    parser.add_argument('--net', action='store_true', help='Runs network/allocator tests (AssembledFrameAllocator, etc.)')
    parser.add_argument('--serv', action='store_true', help='Runs end-to-end FakeXEngine -> FrbServer -> GpuDedisperser -> FrbGrouper test')
    parser.add_argument('--sim', action='store_true', help='Runs avx2_simulate_4bit_noise() distribution test + AssembledFrame pulse-injection and pulse-invariants tests')
    parser.add_argument('--amax', action='store_true', help='Runs DedispersionPlan.decode_argmax() tests (black-box probe arrays)')


def rrange(registry_class):
    """
    This function is used to iterate over a kernel registry, for an appropriate number
    of iterations, so that every kernel is tested a few times. See usage in test() below.
    """

    n = registry_class.registry_size()

    if n == 0:
        print(f'{registry_class.__name__}: no kernels were registered, associated unit test will be skipped.')
        return
    
    for i in range((n+9)//10):
        yield i


def test(args):
    test_flags = [ 'rt', 'pfwr', 'pfom', 'gldk', 'gddk', 'gpfk', 'grck', 'gtgk', 'gdqk', 'cdd2', 'casm', 'chime', 'zomb', 'dd', 'avar', 'net', 'serv', 'sim', 'amax' ]
    run_all_tests = not any(getattr(args,x) for x in test_flags)
    
    ksgpu.set_cuda_device(args.gpu)
    from . import utils   # local import (utils pulls in heavier deps)

    for i in range(args.niter):
        print(f'\nIteration {i+1}/{args.niter}\n')
        
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
        
        if run_all_tests or args.casm:
            print()
            if i == 0:
                # This test is slower than the others, but I don't think we need it more than once.
                casm.CasmReferenceBeamformer.test_interpolative_beamforming()
            
            casm.CasmBeamformer.test_microkernels()
            casm.CasmReferenceBeamformer.test_cuda_python_equivalence(linkage='pybind11')
            
        if run_all_tests or args.chime:
            chime.test_chime_frb_beamform()
            chime.test_chime_frb_upchan()

        if run_all_tests or args.zomb:
            # print()
            loose_ends.test_avx2_m64_outbuf()
            loose_ends.test_cpu_downsampler()
            loose_ends.test_gpu_downsample()
            loose_ends.test_gpu_transpose()
            loose_ends.test_gpu_reduce2()
            
        if run_all_tests or args.dd:
            if i == 0:
                # Catches errors in DedispersionConfig::make_random() or validate().
                for _ in range(500):
                    c = DedispersionConfig.make_random(max_toplevel_rank=8, max_early_triggers=4, gpu_valid=False)
                    c.test()
            for _ in rrange(kernels.CoalescedDdKernel2):
                GpuDedisperser.test_random()
        
        if run_all_tests or args.avar:
            SparseTileTriple.test_random_tree_gridding()
            SparseTile.test_random_iterate_aligned()
            SparseTile.test_random_iterate_singletons()
            SparseTile.test_random_specialize_dbits()
            SparseTile.test_random_remap_d()
            SparseTile.test_random_scale()
            SparseTilePerM.test_random_subbanded_dedispersion()
            PfVarianceConvolver.test_reduces_to_norms()
            PfVarianceConvolver.test_random_variance()
            PfVariance.test_add_truncate_upper_half()
            if i == 0:  # deterministic (no randomness); run once
                PfVarianceConvolver.test_kernels_match_reference()

            # fast_avar: C++ ports compared against the slow_avar python reference.
            test_fast_avar.test_cpp_convolver()
            test_fast_avar.test_cpp_sparse_tile_triple()
            test_fast_avar.test_cpp_pf_variance()
            if i == 0:  # end-to-end (builds a plan + runs the full python reference); run once
                test_fast_avar.test_cpp_pf_avar_approximation()

        if run_all_tests or args.amax:
            tests.test_decode_argmax()

        if run_all_tests or args.net:
            # Network/allocator tests only need to run once (not niter times)
            if i == 0:
                tests.test_assembled_frame_allocator()
                tests.test_slow_subscriber()
            tests.test_assembled_frame_asdf()
            tests.test_network()

        if run_all_tests or args.serv:
            tests.test_server()


######################################   test_simpulse command  #####################################


def parse_test_simpulse(subparsers):
    help_text = "Run simpulse tests (pulse-upsampling self-consistency) and write example plots to cwd"
    parser = subparsers.add_parser("test_simpulse", help=help_text, description=help_text)
    parser.add_argument('-n', '--niter', type=int, default=100, help="Number of upsampling-test iterations (default 100)")


def test_simpulse(args):
    # Import lazily so matplotlib (needed by plot_pulses) is only required for this command.
    from .simpulse import test_pulse_upsampling, plot_pulses
    test_pulse_upsampling.run_tests(args.niter)
    plot_pulses.make_plots()


#################################   check_avar_approximation command  ###############################


def parse_check_avar_approximation(subparsers):
    help_text = "Compare exact vs approximate analytic peak-finding variance for a config"
    parser = subparsers.add_parser("check_avar_approximation", help=help_text, description=help_text)
    parser.add_argument('config_file', help="Path to dedispersion YAML config file")
    parser.add_argument('-r', '--random-variances', action='store_true',
                        help="Use random per-channel variances (config.make_random_freq_variances) instead of all-ones")


def check_avar_approximation(args):
    config = DedispersionConfig.from_yaml(args.config_file)
    config.validate()
    plan = DedispersionPlan(config)
    freq_variances = config.make_random_freq_variances(noisy=True) if args.random_variances else None
    slow_avar.check_approximation(plan, freq_variances)


#################################   check_avar_mc command  ###############################


def parse_check_avar_mc(subparsers):
    help_text = "Monte-Carlo check of analytic peak-finding variance vs a ReferenceDedisperser"
    parser = subparsers.add_parser("check_avar_mc", help=help_text, description=help_text)
    parser.add_argument('config_file', help="Path to dedispersion YAML config file")
    parser.add_argument('-r', '--random-variances', action='store_true',
                        help="Use random per-channel input variances (config.make_random_freq_variances) instead of all-ones")
    parser.add_argument('-s', '--sophistication', type=int, default=1,
                        help="ReferenceDedisperser sophistication (0, 1, or 2; default 1)")


def check_avar_mc(args):
    config = DedispersionConfig.from_yaml(args.config_file)
    print(f"check_avar_mc: forcing nbeams=1 (config had beams_per_gpu={config.beams_per_gpu}, "
          f"beams_per_batch={config.beams_per_batch})", flush=True)
    config.beams_per_gpu = 1
    config.beams_per_batch = 1
    config.num_active_batches = 1
    config.validate()
    plan = DedispersionPlan(config)
    freq_variances = config.make_random_freq_variances(noisy=True) if args.random_variances else None
    slow_avar.check_avar_mc(plan, sophistication=args.sophistication, freq_variances=freq_variances)


#########################################   time command  ##########################################


def parse_time(subparsers):
    help_text = "Run timings (use flags to select specific timings)"
    parser = subparsers.add_parser("time", help=help_text, description=help_text)
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for timing (default 0)")
    parser.add_argument('-t', '--nthreads', type=int, default=0, help="number of CPU threads (for time_cpu_downsample and time_avx2_simulate_4bit_noise)")
    parser.add_argument('--ncu', action='store_true', help="Just run a single kernel (intended for profiling with nvidia 'ncu')")
    parser.add_argument('--gldk', action='store_true', help='Runs time_lagged_downsampling_kernels()')
    parser.add_argument('--gddk', action='store_true', help='Runs time_gpu_dedispersion_kernels()')
    parser.add_argument('--casm', action='store_true', help='Runs CasmBeamformer.run_timings()')
    parser.add_argument('--chime', action='store_true', help='Runs time_chime_frb_{beamform,upchan}()')
    parser.add_argument('--zomb', action='store_true', help='Runs "zombie" timings (code that I wrote during protoyping that may never get used)')
    parser.add_argument('--cdd2', action='store_true', help='Runs CoalescedDdKernel2.time_selected()')
    parser.add_argument('--gdqk', action='store_true', help='Runs GpuDequantizationKernel.time_selected()')
    parser.add_argument('--gtgk', action='store_true', help='Runs GpuTreeGriddingKernel.time_selected()')
    parser.add_argument('--sim', action='store_true', help='Runs avx2_simulate_4bit_noise() timing')

def time_command(args):
    timing_flags = [ 'gldk', 'gddk', 'casm', 'chime', 'zomb', 'cdd2', 'gdqk', 'gtgk', 'sim' ]
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
    if run_all_timings or args.sim:
        utils.time_avx2_simulate_4bit_noise(nthreads)


#####################################   show_hardware command  #####################################


def parse_show_hardware(subparsers):
    help_text = "Show hardware information, including cpu affinity"
    subparsers.add_parser("show_hardware", help=help_text, description=help_text)
    
def show_hardware(args):
    h = Hardware()
    h.show()


######################################   show_kernels command  #####################################


def parse_show_kernels(subparsers):
    help_text = "Show registered cuda kernels (use flags to select specific registries)"
    parser = subparsers.add_parser("show_kernels", help=help_text, description=help_text)
    parser.add_argument('--pfom', action='store_true', help='Show PfOutputMicrokernel registry')
    parser.add_argument('--pfwr', action='store_true', help='Show PfWeightReaderMicrokernel registry')
    parser.add_argument('--gddk', action='store_true', help='Show GpuDedispersionKernel registry')
    parser.add_argument('--gpfk', action='store_true', help='Show GpuPeakFindingKernel registry')
    parser.add_argument('--cdd2', action='store_true', help='Show CoalescedDdKernel2 registry')
    
def show_kernels(args):
    show_flags = [ 'pfom', 'pfwr', 'gddk', 'gpfk', 'cdd2' ]
    show_all = not any(getattr(args, x) for x in show_flags)
    first = True

    if show_all or args.cdd2:
        if not first:
            print()
        first = False
        n = kernels.CoalescedDdKernel2.registry_size()
        print(f"CoalescedDdKernel2 registry ({n} entries):", flush=True)
        kernels.CoalescedDdKernel2.show_registry()

    if show_all or args.pfom:
        if not first:
            print()
        first = False
        n = kernels.PfOutputMicrokernel.registry_size()
        print(f"PfOutput microkernel registry ({n} entries):", flush=True)
        kernels.PfOutputMicrokernel.show_registry()

    if show_all or args.pfwr:
        if not first:
            print()
        first = False
        n = kernels.PfWeightReaderMicrokernel.registry_size()
        print(f"PfWeightReader microkernel registry ({n} entries):", flush=True)
        kernels.PfWeightReaderMicrokernel.show_registry()

    if show_all or args.gddk:
        if not first:
            print()
        first = False
        n = kernels.GpuDedispersionKernel.registry_size()
        print(f"Dedispersion kernel registry ({n} entries):", flush=True)
        kernels.GpuDedispersionKernel.show_registry()
    
    if show_all or args.gpfk:
        if not first:
            print()
        first = False
        n = kernels.GpuPeakFindingKernel.registry_size()
        print(f"Peak-finding kernel registry ({n} entries):", flush=True)
        kernels.GpuPeakFindingKernel.show_registry()


######################################   make_subbands command  #####################################


def parse_make_subbands(subparsers):
    help_text = "A utility for maintaining makefile_helper.py"
    description = textwrap.dedent("""\
        A utility for maintaining makefile_helper.py.

        The 'threshold' argument is a "target" fractional bandwidth. For example, if threshold=0.2,
        then the make_subbands command will try to make bands whose fractional bandwidth is <= 20%.
        However, some subbands may be wider than the threshold, due to technical constraints.

        Example usage::

           # Specify frequency min, max, and threshold
           python -m pirate_frb make_subbands 300 1500 0.2
           python -m pirate_frb make_subbands 400 800 0.1 -r 4""")
    parser = subparsers.add_parser(
        "make_subbands",
        help = help_text,
        description = description,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('fmin', type=float, help='Minimum frequency (MHz)')
    parser.add_argument('fmax', type=float, help='Maximum frequency (MHz)')
    parser.add_argument('threshold', type=float, help='Threshold for fmin/fmax')
    parser.add_argument('-r', '--pf-rank', type=int, default=4, help='Peak finding rank (default: 4)')


def make_subbands(args):
    print(f'Constructing FrequencySubbands(pf_rank={args.pf_rank}, fmin={args.fmin}, fmax={args.fmax}, threshold={args.threshold})')

    # These asserts detect out-of-order positional arguments.
    assert args.fmin > 99.0
    assert args.fmin < args.fmax
    assert args.threshold <= 10.0
    
    fs = core.FrequencySubbands.from_threshold(args.fmin, args.fmax, args.threshold, args.pf_rank)
    print(fs.show())


########################################   hwtest command  #########################################


def parse_hwtest(subparsers):
    import argparse, textwrap
    help_text = "Run hardware test from hwtest.yml (use -s to send data instead of receiving)"
    description = textwrap.dedent("""\
        Run hardware test using YAML config file (use -s to send data instead of receiving).

        Runs and times parallel synthetic loads: network IO, disk IO, PCIe transfers
        between GPU and host, GPU compute kernels, CPU compute kernels, host memory
        bandwidth.

        Example networking-only run::

          # On cf05. The test will pause after "listening for TCP connections".
          python -m pirate_frb hwtest configs/hwtest/cf00_net64.yml

          # On cf00. Send to all four IP addresses on cf05.
          python -m pirate_frb hwtest -s configs/hwtest/cf00_net64.yml

        See configs/hwtest/*.yml for more examples.""")
    parser = subparsers.add_parser(
        "hwtest", help=help_text, description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
            print("\nInterrupted, stopping...")


######################################   scratch command  #######################################


def parse_scratch(subparsers):
    help_text = "For debugging: run whatever code is currently in src_lib/scratch.cu"
    subparsers.add_parser("scratch", help=help_text, description=help_text)

def scratch(args):
    # The scratch() function is defined in src_lib/scratch.cu.
    pirate_pybind11.scratch()


####################################   revisit_512gb command  ####################################


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
    print('Pinned process to CPU 0.')

    h = Hardware()
    print('\nHardware:')
    for gpu in range(h.num_gpus):
        bus_id = h._pcie_bus_id_from_gpu(gpu)
        desc = h._description_from_pcie_bus_id(bus_id)
        print(f'  GPU {gpu}: pcie={bus_id}  ({desc})')

    # Check memory availability.
    print()
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
        print(f'  2 MiB hugepages free: {free_gib:.1f} GiB (test needs {need_gib} GiB)')
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
        print(f'  MemAvailable: {avail_gib:.1f} GiB (test needs {need_gib} GiB)')

    page_label = 'hugepages' if args.hugepages else '4 KiB pages'
    print(f'\nAllocating + registering {test_gib} GiB ({page_label})...')
    success = pirate_pybind11.revisit_512gb_inner(test_nbytes, args.hugepages)

    bar = '=' * 64
    print()
    print(bar)
    if success:
        print(f'cudaHostRegister({test_gib} GiB) SUCCEEDED.')
        print(f'On this CUDA / driver version, the ~511 GiB single-call cap')
        print(f'appears to have been LIFTED. Pirate\'s chunked-register workaround')
        print(f'in BumpAllocator could potentially be simplified or removed --')
        print(f'verify on multiple hardware/driver combinations before doing so.')
    else:
        print(f'cudaHostRegister({test_gib} GiB) FAILED (this is the expected outcome).')
        print(f'The ~511 GiB single-call cap is still in effect on this CUDA / driver')
        print(f'version. Pirate\'s chunked-register workaround in BumpAllocator')
        print(f'remains necessary.')
    print(bar)


################################   show_xengine_metadata command  ##################################


def parse_show_xengine_metadata(subparsers):
    help_text = "Parse xengine_metadata yml file and write info to stdout"
    parser = subparsers.add_parser("show_xengine_metadata", help=help_text, description=help_text)
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Include comments explaining the meaning of each field")


def show_xengine_metadata(args):
    metadata = core.XEngineMetadata.from_yaml_file(args.config_file)
    yaml_str = metadata.to_yaml_string(args.verbose)
    print(yaml_str)


###################################   show_dedisperser command  ###################################


def print_separator(label, filler='-'):
    t = filler * (50 - len(label)//2)
    print(f'\n{t}  {label}  {t}\n')
    sys.stdout.flush()


def parse_show_dedisperser(subparsers):
    help_text = "Parse a dedisperser .yml file and write info to stdout"
    parser = subparsers.add_parser("show_dedisperser", help=help_text, description=help_text)
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Include comments explaining the meaning of each field")
    parser.add_argument('-c', '--config', action='store_true', help="Also print the DedispersionConfig, with a separator, before the plan (by default only the plan is printed, matching the dedispersion_plan_yaml sent to the grouper)")
    parser.add_argument('-t', '--time', action='store_true', help="Also print how long DedispersionPlan and C++ PfAvarApproximation construction took (non-deterministic lines; off by default so the output is reproducible)")
    parser.add_argument('-z', '--zones', action='store_true', help="Include the per-clag mega_ringbuf host/gpu zone breakdown (independent of -v, which controls comments)")
    parser.add_argument('-s', '--streams', type=int, help="Override config.num_active_batches with specified value")
    parser.add_argument('-b', '--beams', type=int, help="Override config.beams_per_gpu with specified value")
    parser.add_argument('-g', '--max-gpu-clag', type=int, help="Override config.max_gpu_clag with specified value")
    parser.add_argument('--channel-map', action='store_true', help="Show channel map tree->freq (warning: produces long output!)")
    parser.add_argument('-a', '--authoritative', action='store_true', help="guarantees consistency with GPU kernels (kernels must be precompiled)")
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
        print(f'# Created with: pirate_frb {cmdline}\n')

    # By default print only the DedispersionPlan, with no separator, so that the
    # output matches the dedispersion_plan_yaml that the FRB search sends to the
    # grouper (see FrbServer / frb_grouper.proto). With -c, also print the
    # DedispersionConfig (the dedispersion_config_yaml wire field) first, with a
    # human-readable separator before the plan.
    if args.config:
        config_yaml = config.to_yaml_string(args.verbose)
        if args.verbose:
            config_yaml = align_inline_comments(config_yaml)
        print(config_yaml)
        print_separator('DedispersionPlan starts here')

    # gpu_runnable iff some flag needs consistency with the compiled GPU kernels:
    # -a requests it explicitly; -r/-R construct a GpuDedisperser from this plan.
    # Otherwise the plan is displayable even in a build without the config's cdd2
    # kernels, at the cost of showing default (non-registry) Dcore values.
    gpu_runnable = args.authoritative or args.resources or args.fine_grained_resources

    t0 = time.time()
    plan = DedispersionPlan(config, gpu_runnable=gpu_runnable)
    plan_dt = time.time() - t0
    if args.time:
        print(f'# DedispersionPlan construction took {plan_dt:.3f} seconds\n')
        # Also time the C++ PfAvarApproximation build from the plan (uses unit input variances;
        # the construction time is independent of the variance values).
        import numpy as np
        freq_variances = np.ones(int(plan.nfreq), dtype=np.float64)
        t0 = time.time()
        PfAvarApproximation(plan, freq_variances)
        avar_dt = time.time() - t0
        print(f'# C++ PfAvarApproximation construction took {avar_dt:.3f} seconds\n')
    plan_yaml = plan.to_yaml_string(args.verbose, args.zones)
    if args.verbose:
        plan_yaml = indent_dedispersion_plan_comments(plan_yaml)
        plan_yaml = align_inline_comments(plan_yaml)
    print(plan_yaml)

    if args.channel_map:
        print_separator('Channel map starts here')
        channel_map = config.make_channel_map()
        
        print()
        print('Channel map (tree_index -> freq_index -> frequency)')
        for i in range(len(channel_map)):
            freq_index = channel_map[i]
            freq = config.index_to_frequency(freq_index)
            print(f'  tree_index={i}  freq_index={freq_index:.4f}  freq={freq:.2f}')

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
        print(rt.to_yaml_string(multiplier, fine_grained))

    if args.test:
        print_separator('Testing GpuDedisperser')
        nchunks = (2**(config.toplevel_tree_rank + config.num_primary_trees - 1)) // config.time_samples_per_chunk + 10
        print(f'Running GpuDedisperser.test_one(config, nchunks={nchunks})')
        GpuDedisperser.test_one(config, nchunks)
        print('Test passed!')


###################################   show_random_config command  ###################################


def parse_show_random_config(subparsers):
    help_text = "For debugging: generate random DedispersionConfig(s) and print as YAML"
    parser = subparsers.add_parser("show_random_config", help=help_text, description=help_text)
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
        print(yaml_str)


###################################   time_dedisperser command  ###################################


def parse_time_dedisperser(subparsers):
    help_text = "Run timing benchmarks from a dedisperser .yml file"
    parser = subparsers.add_parser("time_dedisperser", help=help_text, description=help_text)
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
    print(f'Pinned thread to CPU 0 (vcpus {vcpu_list})')
    
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
    print(f'Creating GpuDedisperser...')
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
    print(f'Allocating (gpu={gpu_nbytes/1e9:.3f} GB sync, '
          f'cpu={cpu_nbytes/1e9:.3f} GB async, nthreads={nthreads})...')
    gpu_allocator = core.BumpAllocator(gpu_aflags, gpu_nbytes, cuda_device=0)
    cpu_allocator = core.BumpAllocator(cpu_aflags, cpu_nbytes,
                                       is_async=True, nthreads=nthreads,
                                       cuda_device=0)
    cpu_allocator.wait_until_initialized()
    dedisperser.allocate(gpu_allocator, cpu_allocator)
    
    # Run timing
    print(f'Running timing (niterations={niterations}, use_hugepages={use_hugepages}, python={use_python})...')
    if use_python:
        # Python version of timing code: pirate_frb.utils.time_cupy_dedisperser().
        utils.time_cupy_dedisperser(dedisperser, gpu_allocator, cpu_allocator, niterations)
    else:
        # C++ version of timing code: GpuDedisperser::time().
        dedisperser.time(gpu_allocator, cpu_allocator, niterations)
    
    print('Timing complete!')


###################################   show_asdf command  ###################################


def parse_show_asdf(subparsers):
    help_text = "Print the YAML header of an ASDF file. (Note: 'asdftool --info' is also useful)"
    parser = subparsers.add_parser("show_asdf", help=help_text, description=help_text)
    parser.add_argument('asdf_file', help="Path to ASDF file")


def show_asdf(args):
    from .utils import show_asdf as _show_asdf
    _show_asdf(args.asdf_file)


######################################   show_file_format command  ##################################


def parse_show_file_format(subparsers):
    help_text = "Make an asdf file from an xengine_metadata YAML file, and write the header to stdout."
    parser = subparsers.add_parser("show_file_format", help=help_text, description=help_text)
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


########################################   rpc_status command  ######################################


def parse_rpc_status(subparsers):
    help_text = "Connect to FrbServer(s) and stream status + filenames"
    parser = subparsers.add_parser("rpc_status", help=help_text, description=help_text)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS', help='Server address(es) (e.g. 127.0.0.1:6000)')


def rpc_status(args):
    from .run_rpc_status import run_rpc_status
    run_rpc_status(args.server_addresses)


######################################   rpc_rand_write command  ####################################


def parse_rpc_rand_write(subparsers):
    help_text = "Send write_files RPC to FrbServer(s) with random beams/time range"
    parser = subparsers.add_parser("rpc_rand_write", help=help_text, description=help_text)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS', help='Server address(es) (e.g. 127.0.0.1:6000)')


def _rpc_rand_write_one(addr):
    """Send a write_files RPC to a single FrbServer."""

    import datetime
    from .rpc import FrbSearchClient

    client = FrbSearchClient(addr)
    print(f"[{addr}] Connected")

    try:
        # Get XEngine metadata to obtain beam IDs. client.beam_ids / xengine_metadata_yaml
        # raise RuntimeError until the server has received metadata.
        try:
            beam_ids = list(client.beam_ids)
        except RuntimeError:
            print(f"[{addr}] Error: metadata not yet available")
            return

        nbeams = len(beam_ids)
        print(f"[{addr}] Got metadata: {nbeams} beams, beam_ids={beam_ids}")

        # seq_per_chunk (fpga seqs per time chunk) = time_samples_per_chunk *
        # seq_per_frb_time_sample. The former comes from GetConfig, the latter
        # from the X-engine metadata. Used to convert our chunk range (derived
        # from the ring-buffer frame-id counters below) into the fpga-seq range
        # that write_files expects.
        seq_per_chunk = client.config.time_samples_per_chunk * client.xengine_metadata_yaml['seq_per_frb_time_sample']

        # Select random subset of beam IDs (1 to min(nbeams, 3)).
        n = random.randint(1, min(nbeams, 3))
        selected_beams = random.sample(beam_ids, n)
        print(f"[{addr}] Selected {n} beams: {selected_beams}")

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

            print(f"[{addr}] Status: rb_reaped={rb_reaped}, rb_processed={rb_processed} -> time_chunk_index range [{rb_t0}, {rb_t1})")

            if rb_t0 >= rb_t1:
                dt = pirate_pybind11.constants.default_print_cadence_sec
                print(f"[{addr}] No frames available yet, sleeping {dt}s...")
                time.sleep(dt)
                continue

            break

        # Choose random time range: rb_t0 <= t0 < t1 <= rb_t1, with 1 <= (t1-t0) <= 3.
        max_range = min(3, rb_t1 - rb_t0)
        range_size = random.randint(1, max_range)
        t0 = random.randint(rb_t0, rb_t1 - range_size)
        t1 = t0 + range_size

        print(f"[{addr}] Requesting time_chunk_index range [{t0}, {t1})")

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

        print(f"[{addr}] write_files returned {len(filenames)} filenames:")
        for fn in filenames:
            print(f"[{addr}]   {fn}")

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
    """Human-readable message from a grpc.RpcError: unary-call errors are
    grpc.Call and carry the server's message in .details(); fall back to str()."""
    details = getattr(e, "details", None)
    return details() if callable(details) else str(e)


def parse_rpc_start_stream(subparsers):
    help_text = ("Send StartStream RPC to one or more FrbServers (write data to disk as it is "
                 "received). Multiple addresses act as one 'super-server'.")
    parser = subparsers.add_parser("rpc_start_stream", help=help_text, description=help_text)
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
                print(f"[{addr}] ERROR: {_rpc_error_str(e)}", file=sys.stderr)
                continue
            print(f"[{addr}] started stream stream_name={sn!r}")
            print(f"[{addr}]   acqdir = {acqdir!r}")
            print(f"[{addr}]   beam_ids = {beams}")
            print(f"[{addr}]   fpga_seq range = [0, {end_str})")

        if had_error:
            sys.exit(1)
    finally:
        for _, client in clients:
            client.close()


##################################   rpc_cancel_stream command  #####################################


def parse_rpc_cancel_stream(subparsers):
    help_text = ("Send CancelStream RPC to one or more FrbServers. Multiple addresses act as one "
                 "'super-server' (cancels loop over all servers).")
    parser = subparsers.add_parser("rpc_cancel_stream", help=help_text, description=help_text)
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
                    print(f"[{addr}] cancelled {n} stream(s)")
                except grpc.RpcError as e:
                    had_error = True
                    print(f"[{addr}] ERROR: {_rpc_error_str(e)}", file=sys.stderr)
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
                    print(f"[{addr}] ERROR: {_rpc_error_str(e)}", file=sys.stderr)
                    continue
                if not any(i.args.stream_name == name and
                           i.status == frb_search_pb2.STREAM_STATUS_ACTIVE
                           for i in ss.streams):
                    continue
                found = True
                try:
                    n = client.cancel_stream(stream_name=name)
                    print(f"[{addr}] cancelled {n} stream(s) named {name!r}")
                except grpc.RpcError as e:
                    had_error = True
                    print(f"[{addr}] ERROR: {_rpc_error_str(e)}", file=sys.stderr)
            if not found and not had_error:
                raise RuntimeError(
                    f"rpc_cancel_stream: no server has an active stream named {name!r} "
                    f"(servers: {', '.join(args.server_addresses)})")
    finally:
        for _, client in clients:
            client.close()

    if had_error:
        sys.exit(1)


###################################   rpc_show_streams command  #####################################


def parse_rpc_show_streams(subparsers):
    help_text = ("Send ShowStreams RPC to one or more FrbServers and print the responses. "
                 "Multiple addresses act as one 'super-server' (loops over all servers).")
    parser = subparsers.add_parser("rpc_show_streams", help=help_text, description=help_text)
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
        print(f"[{addr}] current_fpga_seq = {ss.current_fpga_seq}")
        print(f"[{addr}] beam_ids = {list(ss.beam_ids)}")
        print(f"[{addr}] num_deactivated_streams = {ss.num_deactivated_streams}"
              f" ({n_listed_inactive} retained in history)")

        if not ss.streams:
            print(f"[{addr}] no active or recently-deactivated streams")

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
            print(f"[{addr}] stream stream_name={a.stream_name!r}:")
            print(f"[{addr}]   status = {status}")
            print(f"[{addr}]   acqdir = {a.acqdir!r}")
            print(f"[{addr}]   beam_ids = {list(a.beam_ids)}")
            print(f"[{addr}]   fpga_seq range = [{a.fpga_seq_start}, {end_str}){remaining}")
            print(f"[{addr}]   started = {fmt_time(info.started_at_unix_ns)}, "
                  f"deactivated = {fmt_time(info.deactivated_at_unix_ns)}")
            print(f"[{addr}]   files: queued = {info.num_files_queued}, "
                  f"written = {info.num_files_written}, errored = {info.num_files_errored}")

    clients = _stream_clients(args.server_addresses)
    had_error = False

    try:
        for i, (addr, client) in enumerate(clients):
            if i > 0:
                print()   # blank line between per-server blocks
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
                print(f"[{addr}] ERROR: {_rpc_error_str(e)}", file=sys.stderr)
    finally:
        for _, client in clients:
            client.close()

    if had_error:
        sys.exit(1)


#####################################   random_kernels command  #####################################


def parse_random_kernels(subparsers):
    help_text = "A utility for maintaining makefile_helper.py"
    parser = subparsers.add_parser("random_kernels", help=help_text, description=help_text)
    parser.add_argument('-n', type=int, default=20, help='Number of random kernels to print (default 20)')
    parser.add_argument('--pf', action='store_true', help='Print random PeakFinder kernel params')
    parser.add_argument('--cdd2', action='store_true', help='Print random CoalescedDdKernel2 kernel params')
    parser.add_argument('--pfwr', action='store_true', help='Print random PfWeightReader kernel params')


def random_kernels(args):
    import numpy
    
    flags = [ 'pf', 'cdd2', 'pfwr' ]
    nflags = sum(1 if getattr(args, x) else 0 for x in flags)
    
    if nflags != 1:
        print("Error: precisely one of --pf, --cdd2, --pfwr must be specified", file=sys.stderr)
        print("  --pf     Print random PeakFinder kernel params", file=sys.stderr)
        print("  --cdd2   Print random CoalescedDdKernel2 kernel params", file=sys.stderr)
        print("  --pfwr   Print random PfWeightReader kernel params", file=sys.stderr)
        sys.exit(2)

    randi = lambda *a: int(numpy.random.randint(*a))

    if args.pf:
        print('# (dtype, subband_counts, Wmax, Dcore, Dout, Tinner)')

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
            print(f"('fp{nbits}', {list(subband_counts)}, {Wmax}, {Dcore}, {Dout}, {Tinner})")

    if args.cdd2:
        print("# (dtype, dd_rank, Wmax, Dcore, Dout, Tinner, subband_counts, et_levels)")

        for _ in range(args.n):
            nbits = 32 // randi(1,3)
            subband_counts = core.FrequencySubbands.make_random_subband_counts()
            Wmax = 2**randi(5)
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

            # Currently, cdd2 assumes dd_rank >= 3
            dd_rank_max = randi(3,9)
            dd_rank_min = max(dd_rank_max-1, 3)

            for dd_rank in range(dd_rank_min, dd_rank_max+1):
                et_level_min = 1
                et_level_max = dd_rank-3
                et_candidates = list(range(et_level_min, et_level_max+1))

                ncand = randi(0, len(et_candidates)+1)
                et_levels = [0] + random.sample(et_candidates, ncand)

                s = '     # continuation' if (dd_rank > dd_rank_min) else ''
                print(f"('fp{nbits}', {dd_rank}, {Wmax}, {Dcore}, {Dout}, {Tinner}, {list(subband_counts)}, {et_levels}),{s}")

    if args.pfwr:
        print('# (dtype, subband_counts, Dcore, P, Tinner)')
        
        for _ in range(args.n):
            nbits = 32 // randi(1,3)
            rank = randi(2,5)
            Tinner_log = randi(6)
            Dcore_log = randi(6-Tinner_log) + (32//nbits) - 1
            P = randi(1,15)
            subband_counts = core.FrequencySubbands.make_random_subband_counts()
            print(f"('fp{nbits}', {tuple(subband_counts)}, {2**Dcore_log}, {P}, {2**Tinner_log})")


######################################  run_server command  #####################################


def parse_run_server(subparsers):
    help_text = "Start FRB server(s) from an frb_server .yml file and a dedispersion .yml file"
    parser = subparsers.add_parser("run_server", help=help_text, description=help_text)
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
                             '(printed once per assembled time chunk).')


def run_server_command(args):
    from .run_server import run_server
    run_server(args.server_config, args.dedispersion_config,
               processing_delay_sec=args.delay,
               no_grouper=args.no_grouper,
               no_dedispersion=args.no_dedispersion,
               quiet=args.quiet)


######################################  run_toy_grouper command  #####################################


def parse_run_toy_grouper(subparsers):
    help_text = "Toy FrbGrouper consumer(s): per-chunk peak SNR + argmax, optionally reported to a sifter"
    parser = subparsers.add_parser("run_toy_grouper", help=help_text, description=help_text)
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
                             "per-(beam, chunk) maxes; warmup values are excluded) to 'STEM.pkl' "
                             "upon termination. With multiple groupers, the i-th grouper writes "
                             "'STEM<i>.pkl' (e.g. hist1.pkl, hist2.pkl, ...) so the filenames "
                             "don't collide.")
    # Exactly one of -s/-S is required.
    sifter_group = parser.add_mutually_exclusive_group(required=True)
    sifter_group.add_argument('-s', '--sifter', metavar='SIFTER_ADDR',
                              help="Report to the FrbSifter at this 'ip:port' (e.g. 127.0.0.1:7100):")
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
    base = [sys.executable, '-m', 'pirate_frb', 'run_toy_grouper', *sifter_flag,
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


######################################  run_offline_dedisperser command  #####################################


def parse_run_offline_dedisperser(subparsers):
    help_text = "Toy offline dedispersion: per-chunk peak SNR over an acqdir of .asdf frames"
    parser = subparsers.add_parser("run_offline_dedisperser", help=help_text, description=help_text)
    parser.add_argument("acqdir",
                        help="acqdir of frame_b(BEAM)_t(CHUNK).asdf files")
    parser.add_argument("config",
                        help="dedispersion config yaml")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="only process the first N chunks of each beam (default: all)")


def run_offline_dedisperser_command(args):
    from .run_offline_dedisperser import run_offline_dedisperser
    run_offline_dedisperser(args.acqdir, args.config, max_chunks=args.max_chunks)


######################################  run_toy_sifter command  #####################################


def parse_run_toy_sifter(subparsers):
    help_text = "Toy FrbSifter gRPC server: print a one-line summary of each received message"
    parser = subparsers.add_parser("run_toy_sifter", help=help_text, description=help_text)
    parser.add_argument('addr', metavar='ADDR',
                        help="Listen address 'ip:port' for the sifter gRPC server "
                             "(e.g. 127.0.0.1:7100; use [::]:7100 or 0.0.0.0:7100 for all interfaces).")


def run_toy_sifter_command(args):
    from .run_toy_sifter import run_toy_sifter
    run_toy_sifter(args.addr)


######################################  run_fake_xengine command  #####################################


def parse_run_fake_xengine(subparsers):
    help_text = "Send fake X-engine data to one or more running FrbServers"
    parser = subparsers.add_parser("run_fake_xengine", help=help_text, description=help_text)
    parser.add_argument('rpc_addrs', nargs='+', metavar='RPC_ADDR',
                        help='One or more "ip:port" strings (one per receiver)')
    parser.add_argument('-w', '--workers', type=int, default=128,
                        help='Number of worker threads per FakeXEngine (default 128)')
    parser.add_argument('-P', '--unpaced', action='store_true',
                        help='Disable pacing -- send chunks as fast as possible '
                             '(default: pace to stay <=5 chunks ahead of server)')
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
            print(f"Error: -f/--frbs is incompatible with {', '.join(bad)} "
                  f"(FRB injection requires normalized + gaussian data and "
                  f"randomizes every chunk).", file=sys.stderr)
            sys.exit(2)

    # Sending events to a sifter only makes sense when FRBs are being simulated.
    if args.sifter is not None and not args.frbs:
        print("Error: -s/--sifter requires -f/--frbs (there are no events to send "
              "without FRB simulation).", file=sys.stderr)
        sys.exit(2)

    # An inter-FRB gap only has meaning when FRBs are being simulated.
    if args.gap != 0.0 and not args.frbs:
        print("Error: -g/--gap requires -f/--frbs (there are no FRBs to space "
              "without FRB simulation).", file=sys.stderr)
        sys.exit(2)
    if args.gap < 0.0:
        print("Error: -g/--gap must be >= 0 seconds.", file=sys.stderr)
        sys.exit(2)

    # An FRB SNR only has meaning when FRBs are being simulated.
    if args.frb_snr != 30.0 and not args.frbs:
        print("Error: --frb-snr requires -f/--frbs (there are no FRBs to inject "
              "without FRB simulation).", file=sys.stderr)
        sys.exit(2)

    from .run_fake_xengine import run_fake_xengine
    run_fake_xengine(args.rpc_addrs, nworkers=args.workers,
                     paced=not args.unpaced, normalized=not args.unnormalized,
                     gaussian=not args.non_gaussian,
                     send_junk=args.send_junk, simulate_frbs=args.frbs,
                     sifter_addr=args.sifter, frb_gap_sec=args.gap, frb_snr=args.frb_snr)

####################################################################################################


class _PirateParser(argparse.ArgumentParser):
    """ArgumentParser variant that swallows argparse's auto-appended
    '(choose from {...})' in invalid-choice errors and points the user at
    --help instead. Pairs with metavar='command' on add_subparsers() so
    the run-on choices listing also disappears from --help / usage."""
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

    parse_run_server(subparsers)
    parse_run_toy_grouper(subparsers)
    parse_run_offline_dedisperser(subparsers)
    parse_run_toy_sifter(subparsers)
    parse_run_fake_xengine(subparsers)
    parse_rpc_status(subparsers)
    parse_rpc_rand_write(subparsers)
    parse_rpc_start_stream(subparsers)
    parse_rpc_cancel_stream(subparsers)
    parse_rpc_show_streams(subparsers)
    
    parse_test(subparsers)
    parse_test_simpulse(subparsers)
    parse_check_avar_approximation(subparsers)
    parse_check_avar_mc(subparsers)
    parse_time(subparsers)
    parse_time_dedisperser(subparsers)
    
    parse_show_asdf(subparsers)
    parse_show_file_format(subparsers)
    parse_show_dedisperser(subparsers)
    parse_show_hardware(subparsers)
    parse_show_kernels(subparsers)
    parse_show_random_config(subparsers)
    parse_show_xengine_metadata(subparsers)
    
    parse_hwtest(subparsers)
    parse_make_subbands(subparsers)
    parse_random_kernels(subparsers)
    parse_scratch(subparsers)
    parse_revisit_512gb(subparsers)

    return parser


def main():
    ksgpu.seed_default_rng(137)   # reproducible run; remove for full randomness

    parser = get_parser()
    argcomplete.autocomplete(parser)

    args = parser.parse_args()

    if args.command == "test":
        test(args)
    elif args.command == "test_simpulse":
        test_simpulse(args)
    elif args.command == "check_avar_approximation":
        check_avar_approximation(args)
    elif args.command == "check_avar_mc":
        check_avar_mc(args)
    elif args.command == "time":
        time_command(args)
    elif args.command == "show_hardware":
        show_hardware(args)
    elif args.command == "show_kernels":
        show_kernels(args)
    elif args.command == "make_subbands":
        make_subbands(args)
    elif args.command == "show_xengine_metadata":
        show_xengine_metadata(args)
    elif args.command == "show_dedisperser":
        show_dedisperser(args)
    elif args.command == "time_dedisperser":
        time_dedisperser(args)
    elif args.command == "show_random_config":
        show_random_config(args)
    elif args.command == "hwtest":
        hwtest(args)
    elif args.command == "scratch":
        scratch(args)
    elif args.command == "revisit_512gb":
        revisit_512gb(args)
    elif args.command == "random_kernels":
        random_kernels(args)
    elif args.command == "show_asdf":
        show_asdf(args)
    elif args.command == "show_file_format":
        show_file_format(args)
    elif args.command == "rpc_status":
        rpc_status(args)
    elif args.command == "rpc_rand_write":
        rpc_rand_write(args)
    elif args.command == "rpc_start_stream":
        rpc_start_stream(args)
    elif args.command == "rpc_cancel_stream":
        rpc_cancel_stream(args)
    elif args.command == "rpc_show_streams":
        rpc_show_streams(args)
    elif args.command == "run_server":
        run_server_command(args)
    elif args.command == "run_toy_grouper":
        run_toy_grouper_command(args)
    elif args.command == "run_offline_dedisperser":
        run_offline_dedisperser_command(args)
    elif args.command == "run_toy_sifter":
        run_toy_sifter_command(args)
    elif args.command == "run_fake_xengine":
        run_fake_xengine_command(args)
    else:
        print(f"Command '{args.command}' not recognized", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
