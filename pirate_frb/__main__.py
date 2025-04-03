import sys
import argparse

import ksgpu

from . import pirate_pybind11

from .Hardware import Hardware
from .FakeServer import FakeServer
from .FakeCorrelator import FakeCorrelator


#########################################   test command  ##########################################


def parse_test(subparsers):
    parser = subparsers.add_parser("test", help="Run unit tests")
    parser.add_argument('-g', '--gpu', type=int, default=0, help="GPU to use for tests (default 0)")
    parser.add_argument('-n', '--niter', type=int, default=100, help="Number of unit test iterations (default 100)")

    
def test(args):
    ksgpu.set_cuda_device(args.gpu)
    
    for i in range(args.niter):
        print(f'Iteration {i+1}/{args.niter}')
        pirate_pybind11.test_non_incremental_dedispersion()
        pirate_pybind11.test_reference_lagbuf()
        pirate_pybind11.test_reference_tree()
        pirate_pybind11.test_tree_recursion()
        pirate_pybind11.test_gpu_lagged_downsampling_kernel()
        pirate_pybind11.test_gpu_dedispersion_kernels()
        pirate_pybind11.test_dedisperser()


#####################################   show_hardware command  #####################################


def parse_show_hardware(subparsers):
    subparsers.add_parser("show_hardware", help="Show hardware information")
    
def show_hardware(args):
    h = Hardware()
    Hardware.show()


#######################################   test_node command  #######################################


def parse_test_node(subparsers):
    parser = subparsers.add_parser("test_node", help="Run test server (if no flags are specified, then all tasks execept hmem will be run by default)")
    parser.add_argument('-d', '--dedisperse', dest='d', action='store_true', help='Run GPU dedispersion')
    parser.add_argument('-c', '--cpu', dest='c', action='store_true', help='Run AVX2 downsampling kernels on CPU')
    parser.add_argument('-s', '--ssd', dest='s', action='store_true', help='Write files to SSDs')
    parser.add_argument('-n', '--net', dest='n', action='store_true', help='Receive data over the network')
    parser.add_argument('-H', '--hmem', dest='H', action='store_true', help='Host memory bandwidth test')
    parser.add_argument('--h2g', dest='h2g', action='store_true', help='Copy host->GPU')
    parser.add_argument('--g2h', dest='g2h', action='store_true', help='Copy GPU->host')
    parser.add_argument('-t', '--time', type=float, default=20, help='Number of seconds to run test (default 20)')
    

def test_node(args):
    # FIXME currently hardcoded
    ip_addrs = [ '10.1.1.2', '10.1.2.2', '10.1.3.2', '10.1.4.2' ]
    ssd_dirs = [ '/scratch' ]
    
    tcp_connections_per_ip_address = 1
    downsampling_threads_per_cpu = 8
    write_threads_per_ssd = 4

    no_flags = not (args.d or args.c or args.s or args.n or args.H or args.h2g or args.g2h)
    server = FakeServer('Node test')
    hardware = server.hardware

    if no_flags:
        print("No flags passed to test_node.run() -- by default, all tasks except hmem will be run")

    if args.H:
        # FIXME -- currently submit one thread per vcpu (should do something better)
        for icpu in range(hardware.num_cpus):
            for v in hardware.vcpu_list_from_cpu(icpu):
                server.add_memcpy_thread(-1, -1, cpu=icpu)
                
    if no_flags or args.c:
        for icpu in range(hardware.num_cpus):
            for _ in range(downsampling_threads_per_cpu):
                server.add_downsampling_thread(icpu)

    if no_flags or args.s:
        for issd,ssd_dir in enumerate(ssd_dirs):
            for thread in range(write_threads_per_ssd):
                server.add_ssd_writer(f'{ssd_dir}/thread{thread}', issd)

    if no_flags or args.h2g:
        for gpu in range(hardware.num_gpus):
            server.add_memcpy_thread(-1, gpu)  # h2g
    
    if no_flags or args.g2h:
        for gpu in range(hardware.num_gpus):
            server.add_memcpy_thread(gpu, -1)  # g2h
    
    if no_flags or args.d:
        for gpu in range(hardware.num_gpus):
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


def send(args):
    # FIXME currently hardcoded
    ip_addrs = [ '10.1.1.2', '10.1.2.2', '10.1.3.2', '10.1.4.2' ]
    tcp_connections_per_ip_address = 1

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
