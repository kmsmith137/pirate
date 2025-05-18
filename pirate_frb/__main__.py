import sys
import argparse

import ksgpu

from . import pirate_pybind11

from .Hardware import Hardware
from .FakeServer import FakeServer
from .FakeCorrelator import FakeCorrelator


#########################################   test command  ##########################################


def parse_test(subparsers):
    parser = subparsers.add_parser("test", help="Run unit tests (by default, all tests are run)")
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for tests (default 0)")
    parser.add_argument('-n', '--niter', type=int, default=100, help="Number of unit test iterations (default 100)")
    parser.add_argument('--nid', action='store_true', help='Runs test_non_incremental_dedispersion()')
    parser.add_argument('--rl', action='store_true', help='Runs test_reference_lagbuf()')
    parser.add_argument('--rt', action='store_true', help='Runs test_reference_tree()')
    parser.add_argument('--tr', action='store_true', help='Runs test_tree_recursion()')
    parser.add_argument('--gldk', action='store_true', help='Runs test_gpu_lagged_downsampling_kernel)')
    parser.add_argument('--gddk', action='store_true', help='Runs test_gpu_dedispersion_lernels()')
    parser.add_argument('--dd', action='store_true', help='Runs test_dedisperser()')

    
def test(args):
    test_flags = [ 'nid', 'rl', 'rt', 'tr', 'gldk', 'gddk', 'dd' ]
    run_all_tests = not any(getattr(args,x) for x in test_flags)
    
    ksgpu.set_cuda_device(args.gpu)

    for i in range(args.niter):
        print(f'Iteration {i+1}/{args.niter}')
        
        if run_all_tests or args.nid:
            pirate_pybind11.test_non_incremental_dedispersion()
        if run_all_tests or args.rl:
            pirate_pybind11.test_reference_lagbuf()
        if run_all_tests or args.rt:
            pirate_pybind11.test_reference_tree()
        if run_all_tests or args.tr:
            pirate_pybind11.test_tree_recursion()
        if run_all_tests or args.gldk:
            pirate_pybind11.test_gpu_lagged_downsampling_kernel()
        if run_all_tests or args.gddk:
            pirate_pybind11.test_gpu_dedispersion_kernels()
        if run_all_tests or args.dd:
            pirate_pybind11.test_dedisperser()
            

#####################################   show_hardware command  #####################################


def parse_show_hardware(subparsers):
    subparsers.add_parser("show_hardware", help="Show hardware information")
    
def show_hardware(args):
    h = Hardware()
    h.show()


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
                server.add_ssd_writer(f'{ssd_dir}/thread{thread}', issd)

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
    parser = subparsers.add_parser("send", help='Send data to test server (his is the "other half" of "python -m pirate_frb test_node")')
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


####################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pirate_frb command-line driver (use --help for more info)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_test(subparsers)
    parse_show_hardware(subparsers)
    parse_test_node(subparsers)
    parse_send(subparsers)

    args = parser.parse_args()

    if args.command == "test":
        test(args)
    elif args.command == "show_hardware":
        show_hardware(args)
    elif args.command == "test_node":
        test_node(args)
    elif args.command == "send":
        send(args)
    else:
        print(f"Command '{args.command}' not recognized", file=sys.stderr)
        sys.exit(2)
