import sys
import argparse

from .Hardware import Hardware
from .FakeServer import FakeServer
from .FakeCorrelator import FakeCorrelator


#####################################   show_hardware command  #####################################


def parse_show_hardware(subparsers):
    subparsers.add_parser("show_hardware", help="Show hardware information")
    
def show_hardware(args):
    h = Hardware()
    Hardware.show()


#######################################   test_node command  #######################################


def parse_test_node(subparsers):
    parser = subparsers.add_parser("test_node", help="Run test server (if no flags are specified, then all tasks will be run by default)")
    parser.add_argument('-d', '--dedisperse', dest='d', action='store_true', help='Run GPU dedispersion')
    parser.add_argument('-c', '--cpu', dest='c', action='store_true', help='Run AVX2 downsampling kernels on CPU')
    parser.add_argument('-s', '--ssd', dest='s', action='store_true', help='Write files to SSDs')
    parser.add_argument('-n', '--net', dest='n', action='store_true', help='Receive data over the network')
    parser.add_argument('--h2g', dest='h2g', action='store_true', help='Copy host->GPU')
    parser.add_argument('--g2h', dest='g2h', action='store_true', help='Copy GPU->host')


def test_node(args):
    # FIXME currently hardcoded
    ip_addrs = [ '10.1.1.2', '10.1.2.2', '10.1.3.2', '10.1.4.2' ]
    ssd_dirs = [ '/scratch' ]
    
    tcp_connections_per_ip_address = 1
    downsampling_threads_per_cpu = 8
    write_threads_per_ssd = 4

    no_flags = not (args.d or args.c or args.s or args.n or args.h2g or args.g2h)
    server = FakeServer('Node test')
    hardware = server.hardware

    if no_flags:
        print("No flags passed to test_node.run() -- by default, all tasks will be run")

    if no_flags or args.c:
        for icpu in range(hardware.num_cpus):
            for _ in range(downsampling_threads_per_cpu):
                server.add_downsampling_thread(icpu)

    if no_flags or args.s:
        for ssd_dir in ssd_dirs:
            for thread in range(write_threads_per_ssd):
                server.add_ssd_writer(f'{ssd_dir}/thread{thread}')

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

    server.run()


#########################################   send command  ##########################################


def parse_send(subparsers):
    subparsers.add_parser("send", help='Send data to test server (his is the "other half" of "python -m pirate_frb test_node")')


def send(args):
    # FIXME currently hardcoded
    ip_addrs = [ '10.1.1.2', '10.1.2.2', '10.1.3.2', '10.1.4.2' ]
    tcp_connections_per_ip_address = 1
    gbps_per_ip_address = 20.0

    correlator = FakeCorrelator()

    for ip_addr in ip_addrs:
        correlator.add_endpoint(ip_addr, tcp_connections_per_ip_address, gbps_per_ip_address)

    correlator.run()


####################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pirate_frb command-line driver (use --help for more info)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_show_hardware(subparsers)
    parse_test_node(subparsers)
    parse_send(subparsers)

    args = parser.parse_args()

    if args.command == "show_hardware":
        show_hardware(args)
    elif args.command == "test_node":
        test_node(args)
    elif args.command == "send":
        send(args)
    else:
        print(f"Command '{args.command}' not recognized", file=sys.stderr)
        sys.exit(2)
