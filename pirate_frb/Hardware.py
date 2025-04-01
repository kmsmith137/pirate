#!/usr/bin/env python

import os
import re
import ksgpu
import functools
import subprocess


class Hardware:
    def __init__(self):
        pass

    @functools.cached_property
    def num_cpus(self):
        return len(self._vcpu_list_per_cpu())
        
    @functools.cached_property
    def num_gpus(self):
        return ksgpu.get_cuda_num_devices()

    @functools.cache
    def vcpu_list_from_cpu(self, cpu):
        assert 0 <= cpu < self.num_cpus
        return self._vcpu_list_per_cpu()[cpu]
        
    @functools.cache
    def vcpu_list_from_gpu(self, gpu):
        bus_id = self._pcie_bus_id_from_gpu(gpu)
        return self._vcpu_list_from_pcie_bus_id(bus_id)

    @functools.cache
    def vcpu_list_from_ip_addr(self, ip_addr):
        nic = self._ip_addr_show_output[ip_addr]
        bus_id = self._pcie_bus_id_from_nic(nic)   # can be None, for loopback interface 
        return self._vcpu_list_from_pcie_bus_id(bus_id, allow_none=True)
        
    @functools.cache
    def vcpu_list_from_disk(self, disk):
        bus_id = self._pcie_bus_id_from_block_device(disk)
        return self._vcpu_list_from_pcie_bus_id(bus_id)

    def vcpu_list_from_dirname(self, dirname):
        dev_id = os.stat(dirname).st_dev  # Device ID (major:minor)
        disk = self._dev_id_to_disk_dict[dev_id]
        return self.vcpu_list_from_disk(disk)

    @functools.cached_property
    def ip_addrs(self):
        return sorted(self._ip_addr_show_output.keys())
    
    @functools.cached_property
    def disks(self):
        disks = []

        # Iterate over block devices.
        for device in os.listdir('/sys/class/block'):
            # Exclude loop devices.
            if device.startswith('loop'):
                continue
            
            # Read the uevent file to determine the device type.
            uevent_file = f'/sys/class/block/{device}/uevent'
            if os.path.exists(uevent_file):
                with open(uevent_file, "r") as f:
                    uevent_data = f.read()
                    if "DEVTYPE=disk" in uevent_data:  # Filter by 'disk' type
                        disks.append(device)

        return disks
    
        
    def show(self):
        for cpu in range(self.num_cpus):
            print(f'CPU {cpu}: vcpu_list = {self.vcpu_list_from_cpu(cpu)}')
        print()
        
        for gpu in range(self.num_gpus):
            bus_id = self._pcie_bus_id_from_gpu(gpu)
            description = self._description_from_pcie_bus_id(bus_id)
            vcpu_list = self.vcpu_list_from_gpu(gpu)
            print(f'GPU {gpu}')
            print(f'   pcie = {bus_id}  ({description})')
            print(f'   {vcpu_list = }\n')

        for ip_addr in self.ip_addrs:
            nic = self._ip_addr_show_output[ip_addr]
            bus_id = self._pcie_bus_id_from_nic(nic)
            description = self._description_from_pcie_bus_id(bus_id)
            vcpu_list = self.vcpu_list_from_ip_addr(ip_addr)
            print(f'IP addr {ip_addr}')
            print(f'   nic = {nic}, pcie = {bus_id}  ({description})')
            print(f'   {vcpu_list = }\n')

        for disk in self.disks:
            bus_id = self._pcie_bus_id_from_block_device(disk)
            description = self._description_from_pcie_bus_id(bus_id)
            vcpu_list = self.vcpu_list_from_disk(disk)
            print(f'Disk {disk}')
            print(f'   pcie = {bus_id}  ({description})')
            print(f'   {vcpu_list = }\n')


    ################################################################################################


    @functools.cache
    def _vcpu_list_per_cpu(self):
        """Returns list of lists, containing vcpu list for each physical cpu."""
        
        ret = [ ]
    
        for d in os.listdir("/sys/devices/system/cpu/"):
            if (not d.startswith("cpu")) or (not d[3:].isdigit()):
                continue

            with open(f"/sys/devices/system/cpu/{d}/topology/physical_package_id") as f:
                cpu_id = int(f.read().strip())
                vcpu_id = int(d[3:])
                while len(ret) <= cpu_id:
                    ret.append(list())
                ret[cpu_id].append(vcpu_id)

        assert len(ret) > 0
        assert all((len(x) > 0) for x in ret)
        return [ sorted(x) for x in ret ]

    
    @functools.cache
    def _vcpu_list_from_pcie_bus_id(self, bus_id, allow_none=False):
        """The 'bus_id' arg should be a string e.g. '0000:E1:00.0' (case insenstive)."""

        if allow_none and (bus_id is None):
            return list(range(os.cpu_count()))
        
        filename = f"/sys/bus/pci/devices/{bus_id.lower()}/local_cpulist"
        return self._parse_vcpu_list(open(filename).read())

    
    @functools.cache
    def _pcie_bus_id_from_gpu(self, gpu):
        return ksgpu.get_cuda_pcie_bus_id(gpu).lower()


    @functools.cache
    def _pcie_bus_id_from_sys_subdir(self, pathname):
        """The 'pathname' arg is e.g. '/sys/class/net/eno8303' or '/sys/class/block/nvme0n1p1'."""
        
        pathname = os.path.realpath(pathname)
        if not pathname.startswith('/sys/devices/pci'):
            return None

        pcie_regex = r'^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$'  # thanks chatgpt
        for x in pathname.split('/')[::-1]:
            if re.match(pcie_regex, x):
                return x

        return None
    
    
    @functools.cache
    def _pcie_bus_id_from_nic(self, nic):
        """
        The 'nic' argument should be e.g. 'eth0'. Returns a PCIe bus id (e.g. '0000:03:00.0'),
        or None if not a PCIe device (e.g. loopback, docker).
        """
        return self._pcie_bus_id_from_sys_subdir(f'/sys/class/net/{nic}')


    @functools.cache
    def _pcie_bus_id_from_block_device(self, device_name):
        """
        Given a block device (e.g., '/dev/nvme0n1' or '/dev/nvme0n1p1'),
        returns the PCIe bus id (e.g. '0000:03:00.0').
        """
        return self._pcie_bus_id_from_sys_subdir(f'/sys/class/block/{os.path.basename(device_name)}')


    @functools.cache
    def _description_from_pcie_bus_id(self, bus_id):
        if bus_id is None:
            return 'Not a PCIe device'
        for abbreviated_bus_id, description in self._lspci_output:
            if bus_id.endswith(abbreviated_bus_id):
                return description
        return "Not found in 'lspci'"

    
    def _parse_vcpu_list(self, cpu_str):
        """
        Parses a CPU list string like '0-3,8-11' and returns a list of integers
        such as [0,1,2,3,8,9,10,11].
        """
        
        result = []

        # Split the input string by commas
        for part in cpu_str.strip().split(','):
            # Match ranges like '1-3' or single values like '7'
            match = re.match(r'^(\d+)(?:-(\d+))?$', part.strip())
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else start  # If no range, end = start
                result.extend(range(start, end + 1))  # Add numbers to the list

        return result

    
    @functools.cached_property
    def _ip_addr_show_output(self):
        """
        Parses the output of 'ip -o addr show' and returns a dictionary mapping
        IPv4 addresses to their associated network interfaces.
   
        Example return value: { '192.168.1.100': 'eth0', '10.0.0.5': 'ens3', ... }
        """
        
        ip_to_interface = {}

        # Run the command to get all network interfaces and IPs
        result = subprocess.run(
            ["ip", "-o", "addr", "show"],
            capture_output=True,
            text=True,
            check=True
        )

        # Regex pattern to match IPv4 addresses
        ipv4_pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+)/\d+")

        # Parse the output line by line
        for line in result.stdout.splitlines():
            parts = line.split()
            interface = parts[1]  # Extract interface name

            # Scan for IP addresses in the line
            for part in parts:
                match = ipv4_pattern.match(part)
                if match:
                    ip_address = match.group(1)  # Extract the actual IP address
                    ip_to_interface[ip_address] = interface  # Map IP to interface

        return ip_to_interface


    @functools.cached_property
    def _lspci_output(self):
        """Parses the output of 'lspci' and returns a list of pairs (abbreviated_bus_id, description)."""

        pairs = [ ]
        result = subprocess.run(['lspci'], capture_output=True, text=True, check=True)
        
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if parts and (len(parts) == 2):
                pairs.append(parts)

        return pairs
    

    @functools.cached_property
    def _dev_id_to_disk_dict(self):
        ret = { }
        with open("/proc/mounts") as f:
            for line in f:
                device, mountpoint, *_ = line.split()
                try:
                    dev_id = os.stat(mountpoint).st_dev  # Device ID (major:minor)
                    ret[dev_id] = device
                except:
                    pass
        return ret


if __name__ == '__main__':
    h = Hardware()
    h.show()
