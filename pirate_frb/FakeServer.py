import os
import time

from . import pirate_pybind11

from .Hardware import Hardware


class FakeServer:
    def __init__(self, server_name, use_hugepages=True):
        # Low-level C++ server, exported to python via pybind11.
        self.cpp_server = pirate_pybind11.FakeServer(server_name, use_hugepages)
        
        # The Hardware class provides member functions for querying hardware, in particular
        # for determining which cores are associated with PCIe devices (GPUs, NICs).
        self.hardware = Hardware()


    def add_receiver(self, ip_addr, num_tcp_connections, recv_bufsize = 512*1024, use_epoll=True, network_sync_cadence=16*1024**2):
        vcpu_list = self.hardware.vcpu_list_from_ip_addr(ip_addr)
        self.cpp_server.add_receiver(ip_addr, num_tcp_connections, recv_bufsize, use_epoll, network_sync_cadence, vcpu_list)
                                  
    
    def add_memcpy_worker(self, src_device, dst_device, nbytes_per_iteration, cpu=None, blocksize=1024**3):
        """
        Represents either a host->host, host->GPU, or GPU->host copy.
        The 'src_device' and 'dst_device' args are GPU indices, or (-1) for "host".
        For a host->host copy, the memory bandwidth is (2 * nbytes_per_iteration).

        We use a default blocksize of 2 GiB, since (surprisingly) cudaMemcpy() runs slow
        for sizes >4 GiB. (Empirically, any blocksize between 1MiB and 4GiB works pretty well.)
        """

        m = max(src_device, dst_device)
        
        if m < 0:  # host->host
            assert cpu is not None
            vcpu_list = self.hardware.vcpu_list_from_cpu(cpu)
        else:      # host->gpu or gpu->host
            assert cpu is None
            vcpu_list = self.hardware.vcpu_list_from_gpu(m)
        
        self.cpp_server.add_memcpy_worker(src_device, dst_device, nbytes_per_iteration, blocksize, vcpu_list)


    def add_gpu_copy_kernel(self, gpu, nbytes=128*1024**3, blocksize=1024**3):
        vcpu_list = self.hardware.vcpu_list_from_gpu(gpu)
        self.cpp_server.add_gpu_copy_kernel(gpu, nbytes, blocksize, vcpu_list)
        
        
    def add_ssd_worker(self, root_dir, nfiles_per_iteration, nbytes_per_file, nbytes_per_write):
        os.makedirs(root_dir, exist_ok=True)
        vcpu_list = self.hardware.vcpu_list_from_dirname(root_dir)
        self.cpp_server.add_ssd_worker(root_dir, nfiles_per_iteration, nbytes_per_file, nbytes_per_write, vcpu_list)
    

    def add_downsampling_worker(self, src_bit_depth, src_nelts, cpu):
        vcpu_list = self.hardware.vcpu_list_from_cpu(cpu)
        self.cpp_server.add_downsampling_worker(src_bit_depth, src_nelts, vcpu_list)
        

    def run(self):
        self.cpp_server.start()

        while True:
            time.sleep(1)
            if self.cpp_server.show_stats() > 20:
                break   # server has been running longer than 30 sec

        self.cpp_server.stop()
        self.cpp_server.join_threads()
