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
from .FakeServer import FakeServer
from .FakeCorrelator import FakeCorrelator
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


#######################################   test_node command  #######################################


def parse_test_node(subparsers):
    help_text = "Run test server (if no flags are specified, then -dcsdn --h2g --g2h is the default)"
    parser = subparsers.add_parser("test_node", help=help_text, description=help_text)
    parser.add_argument('-d', '--dedisperse', dest='d', action='store_true', help='Run GPU dedispersion')
    parser.add_argument('-c', '--cpu', dest='c', action='store_true', help='Run AVX2 downsampling kernels on CPU')
    parser.add_argument('-s', '--ssd', dest='s', action='store_true', help='Write files to SSDs')
    parser.add_argument('-n', '--net', dest='n', action='store_true', help='Receive data over the network')
    parser.add_argument('-H', '--hmem', dest='H', action='store_true', help='Host memory bandwidth test')
    parser.add_argument('-G', '--gmem', dest='G', action='store_true', help='GPU memory bandwidth test')
    parser.add_argument('--h2g', dest='h2g', action='store_true', help='Copy host->GPU')
    parser.add_argument('--g2h', dest='g2h', action='store_true', help='Copy GPU->host')
    parser.add_argument('-x', '--cross-numa', dest='cross_numa', action='store_true', help='Split SSD disk writer threads between numa domains')
    parser.add_argument('-t', '--time', type=float, default=20, help='Number of seconds to run test (default 20)')
    parser.add_argument('--ip', type=str, help='Comma-separated list of IP addresses')
    parser.add_argument('--nic', type=str, help='Comma-separated list of NICs')
    parser.add_argument('--ssd-dirs', type=str, help='Comma-separated list of directory names (one per SSD)')
    parser.add_argument('--ssd-devs', type=str, help='Comma-separated list of SSD device names (e.g. /dev/nvme0n1p2 or just nvme0n1p2)')
    parser.add_argument('--toronto', action='store_true', help='Equivalent to --nic=enp55s0f0np0,enp55s0f1np1,enp181s0f0np0,enp181s0f1np1 --ssd-dirs=/scratch')
    parser.add_argument('--chord', action='store_true', help='Equivalent to --nic=enp13s0f0np0,enp13s0f1np1,enp160s0f0np0,enp160s0f1np1 --ssd-dirs=/scratch,/disk2/scratch')
    

def test_node(args):
    # FIXME currently hardcoded
    ssd_dirs = [ '/scratch' ]
    
    tcp_connections_per_ip_address = 1
    downsampling_threads_per_cpu = 8
    write_threads_per_ssd = 4

    no_flags = not (args.d or args.c or args.s or args.n or args.H or args.G or args.h2g or args.g2h)
    server = FakeServer('Node test')
    hw = server.hardware

    # IP address parsing starts here.
    
    ip_flag = 0
    ip_addrs = [ ]
    ip_needed = (no_flags or args.n) 
   
    if args.toronto:
        ip_addrs = [ hw.ip_addr_from_nic(nic) for nic in [ 'enp55s0f0np0', 'enp55s0f1np1', 'enp181s0f0np0', 'enp181s0f1np1' ] ]
        ip_flag += 1   
    elif args.chord:
        ip_addrs = [ hw.ip_addr_from_nic(nic) for nic in [ 'enp13s0f0np0', 'enp13s0f1np1', 'enp160s0f0np0', 'enp160s0f1np1' ] ]
        ip_flag += 1
    elif args.ip is not None:
        ip_addrs = args.ip.split(',')
        ip_flag += 1
    elif args.nic is not None:
        ip_addrs = [ hw.ip_addr_from_nic(nic) for nic in args.nic.split(',') ]
        ip_flag += 1

    # An indirect way of checking that all IP addresses are valid.
    for ip in ip_addrs:
        hw.vcpu_list_from_ip_addr(ip)

    if (ip_flag >= 2) or (ip_needed and (ip_flag == 0)):
        s = 'precisely' if ip_needed else 'at most'
        print(f"pirate 'test_node' command: {s} one of the following must be specified on the command line:", file=sys.stderr)
        print(f"  --ip=[IPADDRS]     for example --ip=10.1.1.2,10.1.2.2,10.1.3.2,10.1.4.2", file=sys.stderr)
        print(f'  --nic=[NICS]       for example --nic=enp55s0f0np0,enp55s0f1np1,enp181s0f0np0,enp181s0f1np1', file=sys.stderr)
        print(f'  --toronto          equivalent to --nic=enp55s0f0np0,enp55s0f1np1,enp181s0f0np0,enp181s0f1np1 --ssd-dirs=/scratch', file=sys.stderr)
        print(f'  --chord            equivalent to --nic=enp13s0f0np0,enp13s0f1np1,enp160s0f0np0,enp160s0f1np1 --ssd-dirs=/scratch,/disk2/scratch', file=sys.stderr)
        sys.exit(2)

    # SSD parsing starts here.

    ssd_flag = 0
    ssd_dirs = [ ]
    ssd_needed = (no_flags or args.s)

    if args.toronto:
        ssd_dirs = [ '/scratch' ]
        ssd_flag += 1
    elif args.chord:
        ssd_dirs = [ '/scratch', '/disk2/scratch' ]
        ssd_flag += 1
    elif args.ssd_dirs is not None:
        ssd_dirs = args.ssd_dirs.split(',')
        ssd_flag += 1
    elif args.ssd_devs is not None:
        ssd_dirs = [ hw.mount_point_from_device(dev) for dev in args.ssd_devs.split(',') ]
        ssd_flag += 1

    # An indirect way of checking that all SSD dirnames are valid.
    for ssd_dir in ssd_dirs:
        hw.vcpu_list_from_dirname(ssd_dir)

    if (ssd_flag >= 2) or (ssd_needed and (ssd_flag == 0)):
        s = 'precisely' if ssd_needed else 'at most'
        print(f"pirate 'test_node' command: {s} one of the following must be specified on the command line:", file=sys.stderr)
        print(f"  --ssd-dirs=[DIRS]     for example --ssd-dirs=/scratch1,/scratch2", file=sys.stderr)
        print(f'  --ssd-devs=[DEVS]     for example --ssd-devs=nvme0n1p1,nvme0n2p1', file=sys.stderr)
        print(f'  --toronto             equivalent to --ssd-dirs=/scratch --nic=enp55s0f0np0,enp55s0f1np1,enp181s0f0np0,enp181s0f1np1', file=sys.stderr)
        print(f'  --chord               equivalent to --ssd-dirs=/scratch,/disk2/scratch --nic=enp13s0f0np0,enp13s0f1np1,enp160s0f0np0,enp160s0f1np1', file=sys.stderr)
        sys.exit(2)
        
    # Add threads to server.
    
    if no_flags:
        print("No flags passed to test_node.run() -- by default, all tasks except hmem will be run")

    if args.H:
        # FIXME -- currently submit one thread per vcpu (should do something better)
        for icpu in range(hw.num_cpus):
            for v in hw.vcpu_list_from_cpu(icpu):
                server.add_memcpy_thread(-1, -1, cpu=icpu)
    if args.G:
        for gpu in range(hw.num_gpus):
            server.add_memcpy_thread(gpu, gpu, use_copy_engine=False)
                
    if no_flags or args.c:
        for icpu in range(hw.num_cpus):
            for _ in range(downsampling_threads_per_cpu):
                server.add_downsampling_thread(icpu)

    if no_flags or args.s:
        for issd,ssd_dir in enumerate(ssd_dirs):
            for thread in range(write_threads_per_ssd):
                cpu = (thread % hw.num_cpus) if args.cross_numa else None
                server.add_ssd_writer(f'{ssd_dir}/thread{thread}', issd, cpu=cpu)

    if no_flags or args.h2g:
        for gpu in range(hw.num_gpus):
            server.add_memcpy_thread(-1, gpu)  # h2g
    
    if no_flags or args.g2h:
        for gpu in range(hw.num_gpus):
            server.add_memcpy_thread(gpu, -1)  # g2h
    
    if no_flags or args.d:
        for gpu in range(hw.num_gpus):
            server.add_chime_dedisperser(gpu)

    if no_flags or args.n:
        for ip_addr in ip_addrs:
            server.add_tcp_receiver(ip_addr, tcp_connections_per_ip_address)

    server.run(args.time)


#########################################   send command  ##########################################


def parse_send(subparsers):
    help_text = 'Send data to test server (this is the "other half" of "python -m pirate_frb test_node")'
    parser = subparsers.add_parser("send", help=help_text, description=help_text)
    parser.add_argument('-r', '--rate', type=float, default=0, help='rate limit per ip address (default 0, meaning no limit)')
    parser.add_argument('-b', '--bufsize', type=int, default=65536, help="Send bufsize (default 65536)")
    parser.add_argument('ip_addrs', nargs='*', help="list of ip addresses, for example: 10.1.1.2 10.1.2.2 10.1.3.2 10.1.4.2")
    parser.add_argument('--toronto', type=int, help="'--toronto X' is equivalent to arguments: 10.1.1.X 10.1.2.X 10.1.3.X 10.1.4.X")
    parser.add_argument('--chord', type=int, help="'--chord X' is also equivalent to arguments: 10.1.1.X 10.1.2.X 10.1.3.X 10.1.4.X")


def send(args):
    # FIXME currently hardcoded
    tcp_connections_per_ip_address = 1

    # Init 'ip_addrs' (list of strings)
    flag = 0
    if len(args.ip_addrs) > 0:
        ip_addrs = args.ip_addrs
        flag += 1
    elif args.toronto is not None:
        n = args.toronto
        ip_addrs = [ f'10.1.1.{n}', f'10.1.2.{n}', f'10.1.3.{n}', f'10.1.4.{n}' ]
        flag += 1
    elif args.chord is not None:
        n = args.chord
        ip_addrs = [ f'10.1.1.{n}', f'10.1.2.{n}', f'10.1.3.{n}', f'10.1.4.{n}' ]
        flag += 1

    if flag != 1:
        print(f"pirate 'send' command: precisely one of the following must be specified on the command line:", file=sys.stderr)
        print(f"  - A list of ip addresses, for example: 10.1.1.2 10.1.2.2 10.1.3.2 10.1.4.2", file=sys.stderr)
        print(f"  - The flag '--toronto X', which is equivalent to: 10.1.1.X 10.1.2.X 10.1.3.X 10.1.4.X", file=sys.stderr)
        print(f"  - The flag '--chord X', which is also equivalent to: 10.1.1.X 10.1.2.X 10.1.3.X 10.1.4.X", file=sys.stderr)
        sys.exit(2)

    with FakeCorrelator(send_bufsize=args.bufsize, use_zerocopy=True, use_mmap=False, use_hugepages=True) as correlator:
        for ip_addr in ip_addrs:
            correlator.add_endpoint(ip_addr, tcp_connections_per_ip_address, args.rate)

        correlator.start()

        # Block until Ctrl-C or worker threads exit (e.g. receiver closes connections).
        # The context manager calls stop()+join() on exit.
        try:
            correlator.join()
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
    parser.add_argument('--channel-map', action='store_true', help="Show channel map tree->freq (warning: produces long output!)")
    parser.add_argument('-r', '--resources', action='store_true', help="Show resource tracking (all kernels must be precompiled)")
    parser.add_argument('-R', '--fine-grained-resources', action='store_true', help="Like -r, but shows fine-grained per-kernel info")
    parser.add_argument('--test', action='store_true', help="Run GpuDedisperser.test_one() with config")


def show_dedisperser(args):
    config = DedispersionConfig.from_yaml(args.config_file)
    
    # Override num_active_batches if --streams was specified
    if args.streams is not None:
        config.num_active_batches = args.streams
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
    help_text = "Connect to FrbServer and stream status + filenames"
    parser = subparsers.add_parser("rpc_status", help=help_text, description=help_text)
    parser.add_argument('server_address', help='Server address (e.g. 127.0.0.1:6000)')


def rpc_status(args):
    import threading
    from .rpc import FrbClient

    def status_thread(client, stop_event):
        """Poll get_status once per second and print summary."""
        try:
            while not stop_event.is_set():
                status = client.get_status()
                print(f"[status] connections={status.num_connections}, bytes={status.num_bytes}, "
                      f"rb=[{status.rb_start},{status.rb_reaped},{status.rb_finalized},{status.rb_end}], "
                      f"free={status.num_free_frames}")

                for _ in range(10):
                    if stop_event.is_set():
                        return
                    time.sleep(0.1)
        except Exception as e:
            print(f"[status] ERROR: {e}", file=sys.stderr)
            stop_event.set()

    def subscribe_thread(client, stop_event):
        """Subscribe to filenames and print as they arrive."""
        try:
            # subscribe_files() yields (filename, error_message) pairs.
            # Empty error_message indicates success; non-empty indicates error.
            for filename, error_message in client.subscribe_files():
                if stop_event.is_set():
                    return
                if error_message:
                    print(f"[subscribe_files] {filename} failed: {error_message}")
                else:
                    print(f"[subscribe_files] {filename} received")
        except Exception as e:
            print(f"[subscribe_files] ERROR: {e}", file=sys.stderr)
            stop_event.set()

    client = FrbClient(args.server_address)

    print(f"RPC client connected to {args.server_address}")
    print("Running get_status (1/sec) and subscribe_files. Press Ctrl-C to stop.")
    print()

    stop_event = threading.Event()

    t_status = threading.Thread(target=status_thread, args=(client, stop_event), daemon=True)
    t_subscribe = threading.Thread(target=subscribe_thread, args=(client, stop_event), daemon=True)

    t_status.start()
    t_subscribe.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    t_status.join(timeout=1.0)
    t_subscribe.join(timeout=1.0)

    client.close()
    print("RPC client stopped.")

    if stop_event.is_set():
        sys.exit(1)


#########################################   rpc_write command  ######################################


def parse_rpc_write(subparsers):
    help_text = "Send write_files RPC to FrbServer with random beams/time range"
    parser = subparsers.add_parser("rpc_write", help=help_text, description=help_text)
    parser.add_argument('server_address', help='Server address (e.g. 127.0.0.1:6000)')


def rpc_write(args):
    import yaml
    from .rpc import FrbClient

    client = FrbClient(args.server_address)
    print(f"Connected to {args.server_address}")

    # Get metadata to obtain beam IDs.
    metadata_yaml = client.get_metadata(verbose=False)
    if not metadata_yaml:
        print("Error: metadata not yet available")
        client.close()
        return

    metadata = yaml.safe_load(metadata_yaml)
    beam_ids = metadata['beam_ids']
    nbeams = len(beam_ids)
    print(f"Got metadata: {nbeams} beams, beam_ids={beam_ids}")

    # Select random subset of beam IDs (1 to min(nbeams, 3)).
    n = random.randint(1, min(nbeams, 3))
    selected_beams = random.sample(beam_ids, n)
    print(f"Selected {n} beams: {selected_beams}")

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

        print(f"Status: rb_reaped={rb_reaped}, rb_end={rb_end} -> time_chunk_index range [{rb_t0}, {rb_t1})")

        if rb_t0 >= rb_t1:
            print("No frames available yet, sleeping 1 second...")
            time.sleep(1)
            continue

        break

    # Choose random time range: rb_t0 <= t0 < t1 <= rb_t1, with 1 <= (t1-t0) <= 3.
    max_range = min(3, rb_t1 - rb_t0)
    range_size = random.randint(1, max_range)
    t0 = random.randint(rb_t0, rb_t1 - range_size)
    t1 = t0 + range_size

    print(f"Requesting time_chunk_index range [{t0}, {t1})")

    # Send write_files RPC.
    # Note: write_files takes (min_time_chunk_index, max_time_chunk_index) as inclusive range.
    filename_pattern = "test_(BEAM)_(CHUNK).asdf"
    filenames = client.write_files(
        beams=selected_beams,
        min_time_chunk_index=t0,
        max_time_chunk_index=t1 - 1,  # inclusive
        filename_pattern=filename_pattern
    )

    print(f"\nwrite_files returned {len(filenames)} filenames:")
    for fn in filenames:
        print(f"  {fn}")

    client.close()


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
    parse_show_hardware(subparsers)
    parse_show_kernels(subparsers)
    parse_make_subbands(subparsers)
    parse_show_xengine_metadata(subparsers)
    parse_show_dedisperser(subparsers)
    parse_time_dedisperser(subparsers)
    parse_show_random_config(subparsers)
    parse_test_node(subparsers)
    parse_send(subparsers)
    parse_scratch(subparsers)
    parse_random_kernels(subparsers)
    parse_show_asdf(subparsers)
    parse_rpc_status(subparsers)
    parse_rpc_write(subparsers)

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
    elif args.command == "test_node":
        test_node(args)
    elif args.command == "send":
        send(args)
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
    else:
        print(f"Command '{args.command}' not recognized", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
