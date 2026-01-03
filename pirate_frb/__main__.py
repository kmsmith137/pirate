import os
import sys
import random
import textwrap
import argparse

import argcomplete
import ksgpu

from . import pirate_pybind11
from . import casm
from . import kernels

from .Hardware import Hardware
from .FakeServer import FakeServer
from .FakeCorrelator import FakeCorrelator
from .yaml_utils import indent_dedispersion_plan_comments, align_inline_comments


#########################################   test command  ##########################################


def parse_test(subparsers):
    parser = subparsers.add_parser("test", help="Run unit tests (by default, all tests are run)")
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
    test_flags = [ 'rt', 'pfwr', 'pfom', 'gldk', 'gddk', 'gpfk', 'grck', 'gtgk', 'gdqk', 'cdd2', 'casm', 'zomb', 'dd' ]
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
            pirate_pybind11.test_avx2_m64_outbuf()
            pirate_pybind11.test_cpu_downsampler()
            pirate_pybind11.test_gpu_downsample()
            pirate_pybind11.test_gpu_transpose()
            pirate_pybind11.test_gpu_reduce2()
            
        if run_all_tests or args.dd:
            for _ in rrange(kernels.CoalescedDdKernel2):
                pirate_pybind11.GpuDedisperser.test_random()
            

#########################################   time command  ##########################################


def parse_time(subparsers):
    parser = subparsers.add_parser("time", help="Run unit times (by default, all timings are run)")
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
    
def time(args):
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
        pirate_pybind11.time_cpu_downsample(nthreads)
        pirate_pybind11.time_gpu_downsample()
        pirate_pybind11.time_gpu_transpose()
    if run_all_timings or args.cdd2:
        kernels.CoalescedDdKernel2.time_selected()
    if run_all_timings or args.gdqk:
        kernels.GpuDequantizationKernel.time_selected()
    if run_all_timings or args.gtgk:
        kernels.GpuTreeGriddingKernel.time_selected()


#####################################   show_hardware command  #####################################


def parse_show_hardware(subparsers):
    subparsers.add_parser("show_hardware", help="Show hardware information")
    
def show_hardware(args):
    h = Hardware()
    h.show()


######################################   show_kernels command  #####################################


def parse_show_kernels(subparsers):
    parser = subparsers.add_parser("show_kernels", help="Show registered cuda kernels (by default, all registries are shown)")
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
    parser = subparsers.add_parser(
        "make_subbands",
        help = "Create subband_counts with specified freq range and width (see 'python -m make_subbands --help')",
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
    
    fs = pirate_pybind11.FrequencySubbands.from_threshold(args.fmin, args.fmax, args.threshold, args.pf_rank)
    print(fs.show())


#######################################   test_node command  #######################################


def parse_test_node(subparsers):
    parser = subparsers.add_parser("test_node", help="Run test server (if no flags are specified, then -dcsdn --h2g --g2h is the default)")
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
    parser = subparsers.add_parser("send", help='Send data to test server (this is the "other half" of "python -m pirate_frb test_node")')
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

    correlator = FakeCorrelator(send_bufsize=args.bufsize, use_zerocopy=True, use_mmap=False, use_hugepages=True)

    for ip_addr in ip_addrs:
        correlator.add_endpoint(ip_addr, tcp_connections_per_ip_address, args.rate)

    correlator.run()


######################################   scratch command  #######################################


def parse_scratch(subparsers):
    # The scratch() function is defined in src_lib/utils.cu.
    subparsers.add_parser("scratch", help="Run scratch code (defined in src_lib/utils.cu)")

def scratch(args):
    # The scratch() function is defined in src_lib/utils.cu.
    pirate_pybind11.scratch()


###################################   show_dedisperser command  ###################################


def print_separator(label, filler='-'):
    t = filler * (50 - len(label)//2)
    print(f'\n{t}  {label}  {t}\n')
    sys.stdout.flush()


def parse_show_dedisperser(subparsers):
    parser = subparsers.add_parser("show_dedisperser", help="Parse a dedisperser config file and write YAML to stdout")
    parser.add_argument('config_file', help="Path to YAML config file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Include comments explaining the meaning of each field")
    parser.add_argument('-c', '--config-only', action='store_true', help="Print config only (skip plan)")
    parser.add_argument('-s', '--streams', type=int, help="Override config.num_active_batches with specified value")
    parser.add_argument('--channel-map', action='store_true', help="Show channel map tree->freq (warning: produces long output!)")
    parser.add_argument('-r', '--resources', action='store_true', help="Show resource tracking (all kernels must be precompiled)")
    parser.add_argument('-R', '--fine-grained-resources', action='store_true', help="Like -r, but shows fine-grained per-kernel info")
    parser.add_argument('--test', action='store_true', help="Run GpuDedisperser.test_one() with config")
    parser.add_argument('--time', action='store_true', help="Run GpuDedisperser.time_one() with niterations=1000")
    parser.add_argument('-H', '--no-hugepages', action='store_true', help="Disable hugepages (only meaningful with --time)")


def show_dedisperser(args):
    config = pirate_pybind11.DedispersionConfig.from_yaml(args.config_file)
    
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
        plan = pirate_pybind11.DedispersionPlan(config)
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
            plan = pirate_pybind11.DedispersionPlan(config)

        nin = plan.beams_per_batch * plan.nfreq * plan.nt_in
        nbits = plan.nbits

        # Add a dequantizer and raw-data h2g copy, to give a more realistic accounting of cost.
        params = pirate_pybind11.GpuDedisperserParams()
        params.plan = plan
        params.stream_pool = pirate_pybind11.CudaStreamPool.create(plan.num_active_batches)
        dedisperser = pirate_pybind11.GpuDedisperser(params)
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
        pirate_pybind11.GpuDedisperser.test_one(config, nchunks)
        print('Test passed!')

    if args.time:
        print_separator('Timing GpuDedisperser')
        niterations = 1000
        use_hugepages = not args.no_hugepages
        print(f'Running GpuDedisperser.time_one(config, niterations={niterations}, use_hugepages={use_hugepages})')
        pirate_pybind11.GpuDedisperser.time_one(config, niterations, use_hugepages)
        print('Timing complete!')


###################################   show_random_config command  ###################################


def parse_show_random_config(subparsers):
    parser = subparsers.add_parser("show_random_config", help="Generate random DedispersionConfig(s) and print as YAML")
    parser.add_argument('-n', type=int, default=1, metavar='NCONFIG', help='generate multiple random configs')
    parser.add_argument('-a', action='store_true', help='generate arbitrary random config, without restricting to precompiled kernels')
    parser.add_argument('-v', action='store_true', help='verbose')


def show_random_config(args):
    gpu_valid = not args.a
    
    for i in range(args.n):
        if args.n > 1:
            print_separator(f'iteration {i+1}/{args.n}', filler='#')
        
        config = pirate_pybind11.DedispersionConfig.make_random(gpu_valid=gpu_valid)
        yaml_str = config.to_yaml_string(verbose=args.v)
        print(yaml_str)


###################################   random_kernels command  ###################################


def parse_random_kernels(subparsers):
    parser = subparsers.add_parser("random_kernels", help="A utility for maintaining makefile_helper.py")
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
            subband_counts = pirate_pybind11.FrequencySubbands.make_random_subband_counts()
            print(f"('fp{nbits}', {list(subband_counts)}, {Wmax}, {Dcore}, {Dout}, {Tinner})")

    if args.cdd2:
        print("# (dtype, dd_rank, Wmax, Dcore, Dout, Tinner, subband_counts, et_delta_ranks)")

        for _ in range(args.n):
            nbits = 32 // randi(1,3)
            subband_counts = pirate_pybind11.FrequencySubbands.make_random_subband_counts()
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
            subband_counts = pirate_pybind11.FrequencySubbands.make_random_subband_counts()
            print(f"('fp{nbits}', {tuple(subband_counts)}, {2**Dcore_log}, {P}, {2**Tinner_log})")


####################################################################################################


def main():
    parser = argparse.ArgumentParser(description="pirate_frb command-line driver (use --help for more info)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_test(subparsers)
    parse_time(subparsers)
    parse_show_hardware(subparsers)
    parse_show_kernels(subparsers)
    parse_make_subbands(subparsers)
    parse_show_dedisperser(subparsers)
    parse_show_random_config(subparsers)
    parse_test_node(subparsers)
    parse_send(subparsers)
    parse_scratch(subparsers)
    parse_random_kernels(subparsers)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.command == "test":
        test(args)
    elif args.command == "time":
        time(args)
    elif args.command == "show_hardware":
        show_hardware(args)
    elif args.command == "show_kernels":
        show_kernels(args)
    elif args.command == "make_subbands":
        make_subbands(args)
    elif args.command == "show_dedisperser":
        show_dedisperser(args)
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
    else:
        print(f"Command '{args.command}' not recognized", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
