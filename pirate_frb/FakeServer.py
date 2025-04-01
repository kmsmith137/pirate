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


    def add_tcp_receiver(self, ip_addr, num_tcp_connections, recv_bufsize=512*1024, use_epoll=True):
        vcpu_list = self.hardware.vcpu_list_from_ip_addr(ip_addr)
        self.cpp_server.add_receiver(ip_addr, num_tcp_connections, recv_bufsize, use_epoll, vcpu_list)


    def add_chime_dedipserser(self, gpu, use_copy_engine=False, num_active_batches=3, beams_per_batch=1, beams_per_gpu=None):
        if beams_per_gpu is None:
            beams_per_gpu = num_active_batches * beams_per_batch

        vcpu_list = self.hardware.vcpu_list_from_gpu(gpu)
        self.cpp_server.add_chime_dedisperser(gpu, beams_per_gpu, num_active_batches, beams_per_batch, use_copy_engine, vcpu_list)
                                  
    
    def add_memcpy_thread(self, src_device, dst_device, blocksize=1024**3, cpu=None, use_copy_engine=False):
        """
        Represents either a host->host, host->GPU, or GPU->host copy.
        The 'src_device' and 'dst_device' args are GPU indices, or (-1) for "host".

        We use a default blocksize of 1 GiB, since (surprisingly) cudaMemcpy() runs slow
        for sizes >4 GiB. (Empirically, any blocksize between 1MiB and 4GiB works pretty well.)

        The 'cpu' argument is required for host->host copies, in order to pin threads to a CPU.

        The 'use_copy_engine' argument is only meaningful for GPU->GPU copies. If use_copy_engine=False,
        then the copy is done with a GPU kernel, rather than cudaMemcpyAsync(). This is useful in a situation
        where both GPU "compute engines" are being used for GPU->host and host->GPU transfers.
        """

        host_to_host = ((src_device < 0) and (dst_device < 0))

        if host_to_host:
            if cpu is None:
                raise RuntimeError("FakeServer.add_memcpy() thread: for a host->host copy, the 'cpu' arg must be specified")
            vcpu_list = self.hardware.vcpu_list_from_cpu(cpu)

        else:
            if cpu is not None:
                raise RuntimeError("FakeServer.add_memcpy_thread(): for a copy involving the GPU, the 'cpu' arg must not be specified")
            if (src_device >= 0) and (dst_device >= 0) and (src_device != dst_device):
                raise RuntimeError("FakeServer.add_memcpy_thread(): GPU->GPU copies between different GPUs are not currently supported")

            gpu = max(src_device, dst_device)
            vcpu_list = self.hardware.vcpu_list_from_gpu(gpu)
        
        self.cpp_server.add_memcpy_worker(src_device, dst_device, blocksize, use_copy_engine, vcpu_list)
        
        
    def add_ssd_writer(self, root_dir, nbytes_per_file = 64 * 1024**2)
        os.makedirs(root_dir, exist_ok=True)
        vcpu_list = self.hardware.vcpu_list_from_dirname(root_dir)
        self.cpp_server.add_ssd_worker(root_dir, nbytes_per_file, vcpu_list)
    

    def add_downsampling_thread(self, src_bit_depth, src_nelts, cpu):
        """Runs the AVX2 downsampling kernel."""
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
