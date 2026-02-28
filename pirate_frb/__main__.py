import os
import sys
import time
import random
import textwrap
import argparse

import argcomplete
import ksgpu

from . import pirate_pybind11
from . import casm
from . import kernels
from . import loose_ends
from . import core
from . import tests

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
    help_text = "Run unit tests (by default, all tests are run)"
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
    parser.add_argument('--ana', action='store_true', help='Runs AnalyticDedisperser.test_random()')
    parser.add_argument('--net', action='store_true', help='Runs network/allocator tests (AssembledFrameAllocator, etc.)')


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
    test_flags = [ 'rt', 'pfwr', 'pfom', 'gldk', 'gddk', 'gpfk', 'grck', 'gtgk', 'gdqk', 'cdd2', 'casm', 'zomb', 'dd', 'ana', 'net' ]
    run_all_tests = not any(getattr(args,x) for x in test_flags)
    
    ksgpu.set_cuda_device(args.gpu)

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
            
        if run_all_tests or args.zomb:
            # print()
            loose_ends.test_avx2_m64_outbuf()
            loose_ends.test_cpu_downsampler()
            loose_ends.test_gpu_downsample()
            loose_ends.test_gpu_transpose()
            loose_ends.test_gpu_reduce2()
            
        if run_all_tests or args.dd:
            for _ in rrange(kernels.CoalescedDdKernel2):
                GpuDedisperser.test_random()
        
        if run_all_tests or args.ana:
            core.AnalyticDedisperser.test_random()
        
        if run_all_tests or args.net:
            # Network/allocator tests only need to run once (not niter times)
            if i == 0:
                tests.test_assembled_frame_allocator()
            tests.test_assembled_frame_asdf()


#########################################   time command  ##########################################


def parse_time(subparsers):
    help_text = "Run unit times (by default, all timings are run)"
    parser = subparsers.add_parser("time", help=help_text, description=help_text)
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for timing (default 0)")
    parser.add_argument('-t', '--nthreads', type=int, default=0, help="number of CPU threads (only for time_cpu_downsample)")
    parser.add_argument('--ncu', action='store_true', help="Just run a single kernel (intended for profiling with nvidia 'ncu')")
    parser.add_argument('--gldk', action='store_true', help='Runs time_lagged_downsampling_kernels()')
    parser.add_argument('--gddk', action='store_true', help='Runs time_gpu_dedispersion_kernels()')
    parser.add_argument('--casm', action='store_true', help='Runs CasmBeamformer.run_timings()')
    parser.add_argument('--zomb', action='store_true', help='Runs "zombie" timings (code that I wrote during protoyping that may never get used)')
    parser.add_argument('--cdd2', action='store_true', help='Runs CoalescedDdKernel2.time_selected()')
    parser.add_argument('--gdqk', action='store_true', help='Runs GpuDequantizationKernel.time_selected()')
    parser.add_argument('--gtgk', action='store_true', help='Runs GpuTreeGriddingKernel.time_selected()')
    
def time_command(args):
    timing_flags = [ 'gldk', 'gddk', 'casm', 'zomb', 'cdd2', 'gdqk', 'gtgk' ]
    run_all_timings = not any(getattr(args,x) for x in timing_flags)

    if args.ncu:
        nflags = sum((1 if getattr(args,x) else 0) for x in timing_flags)
        if nflags != 1:
            raise RuntimeError(f'If --ncu is specified, then precisely one of {timing_flags} must be specified')
        if not args.casm:
            raise RuntimeError(f'Currently, the --ncu flag is only supported with --casm (FIXME)')
        
    ksgpu.set_cuda_device(args.gpu)
    nthreads = args.nthreads if (args.nthreads > 0) else os.cpu_count()
        
    if run_all_timings or args.gldk:
        kernels.GpuLaggedDownsamplingKernel.time_selected()
    if run_all_timings or args.gddk:
        kernels.GpuDedispersionKernel.time_selected()
    if run_all_timings or args.casm:
        casm.CasmBeamformer.run_timings(args.ncu)
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


#####################################   show_hardware command  #####################################


def parse_show_hardware(subparsers):
    help_text = "Show hardware information"
    subparsers.add_parser("show_hardware", help=help_text, description=help_text)
    
def show_hardware(args):
    h = Hardware()
    h.show()


######################################   show_kernels command  #####################################


def parse_show_kernels(subparsers):
    help_text = "Show registered cuda kernels (by default, all registries are shown)"
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
    help_text = "Create subband_counts with specified freq range and width"
    parser = subparsers.add_parser(
        "make_subbands",
        help = help_text,
        description = help_text,
        formatter_class = argparse.RawDescriptionHelpFormatter,   # multi-line epilog
        epilog = textwrap.dedent("""
        Example usage:

           # Specify frequency min, max, and threshold
           python -m make_subbands 300 1500 0.2
           python -m make_subbands 400 800 0.1 -r 4

        The 'threshold' argument is a "target" fractional bandwidth. For example, if threshold=0.2,
        then the make_subbands command will try to make bands whose fractional bandwidth is <= 20%.
        However, some subbands may be wider than the threshold, due to technical constraints.
        """)
    )

    parser.add_argument('fmin', type=float, help='Minimum frequency (MHz)')
    parser.add_argument('fmax', type=float, help='Maximum frequency (MHz)')
    parser.add_argument('threshold', type=float, help='Threshold for flo/fhi')
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
    help_text = "Run hardware test using YAML config file (use -s to send data instead of receiving)"
    description = textwrap.dedent("""\
        Run hardware test using YAML config file (use -s to send data instead of receiving).

        Runs and times parallel synthetic loads: network IO, disk IO, PCIe transfers
        between GPU and host, GPU compute kernels, CPU compute kernels, host memory
        bandwidth.

        Example networking-only run:

          # On cf05. The test will pause after "listening for TCP connections".
          python -m pirate_frb hwtest configs/hwtest/cf05_net64.yml

          # On cf00. Send to all four IP addresses on cf05.
          python -m pirate_frb hwtest -s configs/hwtest/cf05_net64.yml

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
            while not sender.wait(500):
                pass
        except KeyboardInterrupt:
            print("\nInterrupted, stopping...")


######################################   scratch command  #######################################


def parse_scratch(subparsers):
    # The scratch() function is defined in src_lib/scratch.cu.
    help_text = "Run scratch code (defined in src_lib/scratch.cu)"
    subparsers.add_parser("scratch", help=help_text, description=help_text)

def scratch(args):
    # The scratch() function is defined in src_lib/scratch.cu.
    pirate_pybind11.scratch()


################################   show_xengine_metadata command  ##################################


def parse_show_xengine_metadata(subparsers):
    help_text = "Parse X-engine metadata file and write YAML to stdout"
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
    help_text = "Parse a dedisperser config file and write YAML to stdout"
    parser = subparsers.add_parser("show_dedisperser", help=help_text, description=help_text)
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Include comments explaining the meaning of each field")
    parser.add_argument('-c', '--config-only', action='store_true', help="Print config only (skip plan)")
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
    
    config_yaml = config.to_yaml_string(args.verbose)
    if args.verbose:
        config_yaml = align_inline_comments(config_yaml)
    print(config_yaml)
    
    if not args.config_only:
        print_separator('DedispersionPlan starts here')
        plan = DedispersionPlan(config)
        plan_yaml = plan.to_yaml_string(args.verbose)
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
        if args.config_only:
            plan = DedispersionPlan(config)

        nin = plan.beams_per_batch * plan.nfreq * plan.nt_in
        nbits = plan.nbits

        # Add a dequantizer and raw-data h2g copy, to give a more realistic accounting of cost.
        stream_pool = core.CudaStreamPool.create(plan.num_active_batches)
        dedisperser = GpuDedisperser(plan, stream_pool)
        rt = dedisperser.resource_tracker.clone()
        rt.add_kernel('dequantizer', (nin * (nbits+4)) // 8)
        rt.add_memcpy_h2g('raw_data', (nin*4) // 8)

        multiplier = (config.beams_per_gpu / config.beams_per_batch) / (1.0e-3 * config.time_samples_per_chunk * config.time_sample_ms)
        fine_grained = args.fine_grained_resources
        print(rt.to_yaml_string(multiplier, fine_grained))

    if args.test:
        print_separator('Testing GpuDedisperser')
        nchunks = (2**(config.tree_rank + config.num_downsampling_levels - 1)) // config.time_samples_per_chunk + 10
        print(f'Running GpuDedisperser.test_one(config, nchunks={nchunks})')
        GpuDedisperser.test_one(config, nchunks)
        print('Test passed!')


###################################   show_random_config command  ###################################


def parse_show_random_config(subparsers):
    help_text = "Generate random DedispersionConfig(s) and print as YAML"
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
    help_text = "Run GpuDedisperser timing benchmarks"
    parser = subparsers.add_parser("time_dedisperser", help=help_text, description=help_text)
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-n', '--niter', type=int, default=1000, help="Number of iterations for timing (default 1000)")
    parser.add_argument('-b', '--beams', type=int, help="Override config.beams_per_gpu with specified value")
    parser.add_argument('-g', '--max-gpu-clag', type=int, help="Override config.max_gpu_clag with specified value")
    parser.add_argument('-H', '--no-hugepages', action='store_true', help="Disable hugepages")
    parser.add_argument('--python', action='store_true', help="Use Python/cupy timing instead of C++ (for testing pybind11 interface)")


def time_dedisperser(args):
    from . import utils
    
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
    dedisperser = GpuDedisperser(plan, stream_pool, detect_deadlocks=True)
    
    # Calculate total memory needed.
    # Dedisperser memory footprints come from resource tracking.
    # Raw data arrays have shape (S, B, F, T // 2) with dtype uint8.
    # BumpAllocator aligns all allocations to 128 bytes, so we add margin for alignment overhead.
    S = plan.num_active_batches
    B = plan.beams_per_batch
    F = plan.nfreq
    T = plan.nt_in
    raw_nbytes = S * B * F * (T // 2)  # multi_raw_gpu and multi_raw_cpu
    alignment_margin = 128             # 1 MB margin for alignment overhead
    
    gpu_nbytes = dedisperser.resource_tracker.get_gmem_footprint() + raw_nbytes + alignment_margin
    cpu_nbytes = dedisperser.resource_tracker.get_hmem_footprint() + raw_nbytes + alignment_margin
    
    # Create allocators with pre-computed capacities and allocate
    print(f'Allocating (gpu={gpu_nbytes/1e9:.3f} GB, cpu={cpu_nbytes/1e9:.3f} GB)...')
    gpu_allocator = core.BumpAllocator(gpu_aflags, gpu_nbytes)
    cpu_allocator = core.BumpAllocator(cpu_aflags, cpu_nbytes)
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


###################################   random_kernels command  ###################################


def parse_show_asdf(subparsers):
    help_text = "Print the YAML header of an ASDF file. (Note: 'asdftool --info' is also useful)"
    parser = subparsers.add_parser("show_asdf", help=help_text, description=help_text)
    parser.add_argument('asdf_file', help="Path to ASDF file")


def show_asdf(args):
    from .utils import show_asdf as _show_asdf
    _show_asdf(args.asdf_file)


########################################   rpc_status command  ######################################


def parse_rpc_status(subparsers):
    help_text = "Connect to FrbServer(s) and stream status + filenames"
    parser = subparsers.add_parser("rpc_status", help=help_text, description=help_text)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS', help='Server address(es) (e.g. 127.0.0.1:6000)')


def rpc_status(args):
    import threading
    from .rpc import FrbClient

    def status_thread(addr, client, stop_event):
        """Poll get_status once per second and print summary."""
        try:
            prev_time = None
            prev_bytes = None

            while not stop_event.is_set():
                status = client.get_status()
                now = time.monotonic()

                bw_str = ""
                if prev_time is not None and (now - prev_time) > 0:
                    delta_bytes = status.num_bytes - prev_bytes
                    delta_time = now - prev_time
                    gbps = (delta_bytes * 8) / (delta_time * 1e9)
                    bw_str = f", bw={gbps:.2f} Gbps"

                prev_time = now
                prev_bytes = status.num_bytes

                print(f"[{addr}] connections={status.num_connections}, bytes={status.num_bytes}{bw_str}, "
                      f"rb=[{status.rb_start},{status.rb_reaped},{status.rb_finalized},{status.rb_end}], "
                      f"free={status.num_free_frames}")

                for _ in range(10):
                    if stop_event.is_set():
                        return
                    time.sleep(0.1)
        except Exception as e:
            print(f"[{addr}] ERROR: {e}", file=sys.stderr)
            stop_event.set()

    def subscribe_thread(addr, client, stop_event):
        """Subscribe to filenames and print as they arrive."""
        try:
            # subscribe_files() yields (filename, error_message) pairs.
            # Empty error_message indicates success; non-empty indicates error.
            for filename, error_message in client.subscribe_files():
                if stop_event.is_set():
                    return
                if error_message:
                    print(f"[{addr}] {filename} failed: {error_message}")
                else:
                    print(f"[{addr}] {filename} received")
        except Exception as e:
            print(f"[{addr}] subscribe_files ERROR: {e}", file=sys.stderr)
            stop_event.set()

    clients = []
    for addr in args.server_addresses:
        clients.append((addr, FrbClient(addr)))

    print(f"RPC client(s) connected to {', '.join(args.server_addresses)}")
    print("Running get_status (1/sec) and subscribe_files. Press Ctrl-C to stop.")
    print()

    stop_event = threading.Event()
    threads = []

    for addr, client in clients:
        t = threading.Thread(target=status_thread, args=(addr, client, stop_event), daemon=True)
        t.start()
        threads.append(t)
        t = threading.Thread(target=subscribe_thread, args=(addr, client, stop_event), daemon=True)
        t.start()
        threads.append(t)

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    for t in threads:
        t.join(timeout=1.0)
    for _, client in clients:
        client.close()
    print("RPC client(s) stopped.")

    if stop_event.is_set():
        sys.exit(1)


#########################################   rpc_write command  ######################################


def parse_rpc_write(subparsers):
    help_text = "Send write_files RPC to FrbServer(s) with random beams/time range"
    parser = subparsers.add_parser("rpc_write", help=help_text, description=help_text)
    parser.add_argument('server_addresses', nargs='+', metavar='ADDRESS', help='Server address(es) (e.g. 127.0.0.1:6000)')


def _rpc_write_one(addr):
    """Send a write_files RPC to a single FrbServer."""

    import yaml
    from .rpc import FrbClient

    client = FrbClient(addr)
    print(f"[{addr}] Connected")

    try:
        # Get metadata to obtain beam IDs.
        metadata_yaml = client.get_metadata(verbose=False)
        if not metadata_yaml:
            print(f"[{addr}] Error: metadata not yet available")
            return

        metadata = yaml.safe_load(metadata_yaml)
        beam_ids = metadata['beam_ids']
        nbeams = len(beam_ids)
        print(f"[{addr}] Got metadata: {nbeams} beams, beam_ids={beam_ids}")

        # Select random subset of beam IDs (1 to min(nbeams, 3)).
        n = random.randint(1, min(nbeams, 3))
        selected_beams = random.sample(beam_ids, n)
        print(f"[{addr}] Selected {n} beams: {selected_beams}")

        # Loop until we have frames available.
        while True:
            status = client.get_status()
            rb_reaped = status.rb_reaped
            rb_end = status.rb_end

            # Convert frame IDs to time_chunk_index range.
            # frame_id = time_chunk_index * nbeams + beam_index
            # So time_chunk_index = frame_id // nbeams
            # rb_t0: first fully available time chunk (round up)
            # rb_t1: last available time chunk + 1 (round down)
            rb_t0 = (rb_reaped + nbeams - 1) // nbeams  # round up
            rb_t1 = rb_end // nbeams  # round down

            print(f"[{addr}] Status: rb_reaped={rb_reaped}, rb_end={rb_end} -> time_chunk_index range [{rb_t0}, {rb_t1})")

            if rb_t0 >= rb_t1:
                print(f"[{addr}] No frames available yet, sleeping 1 second...")
                time.sleep(1)
                continue

            break

        # Choose random time range: rb_t0 <= t0 < t1 <= rb_t1, with 1 <= (t1-t0) <= 3.
        max_range = min(3, rb_t1 - rb_t0)
        range_size = random.randint(1, max_range)
        t0 = random.randint(rb_t0, rb_t1 - range_size)
        t1 = t0 + range_size

        print(f"[{addr}] Requesting time_chunk_index range [{t0}, {t1})")

        # Send write_files RPC.
        # Note: write_files takes (min_time_chunk_index, max_time_chunk_index) as inclusive range.
        filename_pattern = "test_(BEAM)_(CHUNK).asdf"
        filenames = client.write_files(
            beams=selected_beams,
            min_time_chunk_index=t0,
            max_time_chunk_index=t1 - 1,  # inclusive
            filename_pattern=filename_pattern
        )

        print(f"[{addr}] write_files returned {len(filenames)} filenames:")
        for fn in filenames:
            print(f"[{addr}]   {fn}")

    finally:
        client.close()


def rpc_write(args):
    for addr in args.server_addresses:
        _rpc_write_one(addr)


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
        print("# (dtype, dd_rank, Wmax, Dcore, Dout, Tinner, subband_counts, et_delta_ranks)")

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
                et_delta_rank_min = 1
                et_delta_rank_max = dd_rank-3
                et_candidates = list(range(et_delta_rank_min, et_delta_rank_max+1))

                ncand = randi(0, len(et_candidates)+1)
                et_delta_ranks = [0] + random.sample(et_candidates, ncand)

                s = '     # continuation' if (dd_rank > dd_rank_min) else ''
                print(f"('fp{nbits}', {dd_rank}, {Wmax}, {Dcore}, {Dout}, {Tinner}, {list(subband_counts)}, {et_delta_ranks}),{s}")

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
    help_text = "Start FRB server(s) from a YAML config file"
    parser = subparsers.add_parser("run_server", help=help_text, description=help_text)
    parser.add_argument('config', help='Path to YAML config file')
    # -s flag reserved for future use (fake X-engine sender mode).
    parser.add_argument('-s', '--send', action='store_true', help='(not yet implemented) Send fake X-engine data')


####################################################################################################


def get_parser():
    """
    Create and return the argument parser for pirate_frb.
    
    This function is separate from main() so that sphinx-argparse can
    introspect the parser without actually parsing command-line arguments.
    """
    parser = argparse.ArgumentParser(description="pirate_frb command-line driver (use --help for more info)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_test(subparsers)
    parse_time(subparsers)
    parse_time_dedisperser(subparsers)
    
    parse_show_asdf(subparsers)
    parse_show_dedisperser(subparsers)
    parse_show_hardware(subparsers)
    parse_show_kernels(subparsers)
    parse_show_random_config(subparsers)
    parse_show_xengine_metadata(subparsers)
    
    parse_rpc_status(subparsers)
    parse_rpc_write(subparsers)
    
    parse_make_subbands(subparsers)
    parse_random_kernels(subparsers)
    parse_hwtest(subparsers)
    parse_run_server(subparsers)
    parse_scratch(subparsers)

    return parser


def main():
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.command == "test":
        test(args)
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
    elif args.command == "random_kernels":
        random_kernels(args)
    elif args.command == "show_asdf":
        show_asdf(args)
    elif args.command == "rpc_status":
        rpc_status(args)
    elif args.command == "rpc_write":
        rpc_write(args)
    elif args.command == "run_server":
        if args.send:
            from .run_server import run_fake_xengine
            run_fake_xengine(args.config)
        else:
            from .run_server import run_server
            run_server(args.config)
    else:
        print(f"Command '{args.command}' not recognized", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
