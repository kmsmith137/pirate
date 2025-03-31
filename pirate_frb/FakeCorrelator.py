from . import pirate_pybind11

from .Hardware import Hardware


class FakeCorrelator:
    def __init__(self, send_bufsize=512*1024, use_zerocopy=True, use_mmap=False, use_hugepages=True):
        # Low-level C++ correlator, exported to python via pybind11.
        self.cpp_correlator = pirate_pybind11.FakeCorrelator(send_bufsize, use_zerocopy, use_mmap, use_hugepages)
        
        # The Hardware class provides member functions for querying hardware, in particular
        # for determining which cores are associated with PCIe devices (GPUs, NICs).
        self.hardware = Hardware()


    def add_endpoint(self, src_ip_addr, dst_ip_addr, num_tcp_connections, total_gbps):
        # FIXME 'src_ip_addr' should not be necessary!
        vcpu_list = self.hardware.vcpu_list_from_ip_addr(src_ip_addr)
        self.cpp_correlator.add_endpoint(dst_ip_addr, num_tcp_connections, total_gbps, vcpu_list)
                                  
        
    def run(self):
        self.cpp_correlator.run()
