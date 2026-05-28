import os
import re
import ksgpu
import socket
import itertools
import functools
import subprocess


class Hardware:
    def __init__(self):
        """
        I hacked up some python code (`class Hardware`) to query hardware
        and work out which devices are associated with each CPU.
        You can run this code with `python -m pirate_frb show_hardware`.
        This code is currently pretty terrible -- feel free to improve it.
        
        The `Hardware` class is used when starting the real-time server,
        to decide which hardware to associate with each of the two `FrbServer`
        instances. This is all currently done from python.
        """
        
        pass

    @functools.cached_property
    def num_cpus(self):
        return max(self._parse_cpu_topology) + 1

    @functools.cached_property
    def num_vcpus(self):
        return len(self._parse_cpu_topology)
        
    @functools.cached_property
    def num_gpus(self):
        return ksgpu.get_cuda_num_devices()


    # ----- NUMA node accessors -----

    @functools.cached_property
    def numa_nodes(self):
        """Sorted list of NUMA node ids found in /sys/devices/system/node/.
        Returns [] on a system without a NUMA sysfs tree."""

        base = '/sys/devices/system/node'
        if not os.path.isdir(base):
            return []
        ret = []
        for name in os.listdir(base):
            m = re.fullmatch(r'node(\d+)', name)
            if m:
                ret.append(int(m.group(1)))
        return sorted(ret)

    @functools.cache
    def vcpu_list_from_numa_node(self, node):
        """vCPU list (e.g. [0,1,2,...]) for the given NUMA node id."""
        assert node in self.numa_nodes, f'unknown NUMA node {node}'
        with open(f'/sys/devices/system/node/node{node}/cpulist') as f:
            return self._parse_vcpu_list(f.read())


    # ----- Hugepage accessors -----
    #
    # Read directly from sysfs each time the cache is cold. Values can
    # change at runtime (admin writes to sysfs), but show_hardware is a
    # snapshot tool so caching once is fine.

    @functools.cached_property
    def hugepage_sizes(self):
        """Sorted list of hugepage sizes (BYTES) found in
        /sys/kernel/mm/hugepages/. Returns [] if HugeTLB is unconfigured."""

        base = '/sys/kernel/mm/hugepages'
        if not os.path.isdir(base):
            return []
        ret = []
        for name in os.listdir(base):
            m = re.fullmatch(r'hugepages-(\d+)kB', name)
            if m:
                ret.append(int(m.group(1)) * 1024)
        return sorted(ret)

    @functools.cache
    def hugepage_pool(self, size):
        """System-wide hugepage pool for the given page size (in bytes).
        Returns dict {'nr', 'free', 'surplus'}. Raises if 'size' is not
        in self.hugepage_sizes."""

        assert size in self.hugepage_sizes, f'unknown hugepage size {size}'
        d = f'/sys/kernel/mm/hugepages/hugepages-{size // 1024}kB'
        return self._read_hugepage_dir(d)

    @functools.cache
    def hugepage_pool_per_node(self, size, node):
        """Per-NUMA-node hugepage pool. Returns dict {'nr', 'free',
        'surplus'} -- same keys as hugepage_pool()."""

        assert size in self.hugepage_sizes, f'unknown hugepage size {size}'
        assert node in self.numa_nodes,     f'unknown NUMA node {node}'
        d = f'/sys/devices/system/node/node{node}/hugepages/hugepages-{size // 1024}kB'
        return self._read_hugepage_dir(d)

    @staticmethod
    def _read_hugepage_dir(dirname):
        # Helper for hugepage_pool[*_per_node]. surplus_hugepages exists in
        # both system-wide and per-node trees; the other admin/policy
        # files (resv_hugepages, nr_overcommit_hugepages,
        # nr_hugepages_mempolicy) are system-wide only and not surfaced
        # here.
        def _read_int(fn):
            with open(fn) as f:
                return int(f.read().strip())
        return {
            'nr':      _read_int(f'{dirname}/nr_hugepages'),
            'free':    _read_int(f'{dirname}/free_hugepages'),
            'surplus': _read_int(f'{dirname}/surplus_hugepages'),
        }

    @staticmethod
    def _fmt_bytes(n):
        """Format byte count as e.g. '2 MiB', '1.50 GiB', '768 GiB', '0 B'.
        Integers stay integer; non-integers print with 2 decimals."""
        if n == 0:
            return '0 B'
        for unit_bytes, unit_name in [(1<<30, 'GiB'), (1<<20, 'MiB'), (1<<10, 'KiB')]:
            if abs(n) >= unit_bytes:
                q, r = divmod(n, unit_bytes)
                if r == 0:
                    return f'{q} {unit_name}'
                return f'{n / unit_bytes:.2f} {unit_name}'
        return f'{n} B'


    @functools.cache
    def vcpu_list_from_cpu(self, cpu):
        assert 0 <= cpu < self.num_cpus
        ret = [ v for v,c in enumerate(self._parse_cpu_topology) if c == cpu ]
        assert len(ret) > 0
        return ret

    def cpu_from_vcpu_list(self, vcpu_list):
        """Returns None if the vcpu_list is either empty, or spans multiple CPUs."""
        
        ret = None
        for v in vcpu_list:
            assert 0 <= v < self.num_vcpus
            c = self._parse_cpu_topology[v]
            if (ret is not None) and (ret != c):
                return None
            ret = c
        return ret

    
    @functools.cache
    def vcpu_list_from_gpu(self, gpu):
        bus_id = self._pcie_bus_id_from_gpu(gpu)
        return self._vcpu_list_from_pcie_bus_id(bus_id)

    
    @functools.cached_property
    def nics(self):
        # FIXME only returns NICs which have been assigned an IP address
        return [ nic for nic,ip in self._parse_ip_addr_show ]

    @functools.cached_property
    def ip_addrs(self):
        return [ ip for nic,ip in self._parse_ip_addr_show ]

    @functools.cache
    def ip_addr_from_nic(self, nic):
        for n,ip in self._parse_ip_addr_show:
            if n == nic:
                return ip
        raise RuntimeError(f"Couldn't associate NIC {ip_addr} with a NIC")

    @functools.cache
    def nic_from_ip_addr(self, ip_addr, is_dst_addr=False):
        for nic,ip in self._parse_ip_addr_show:
            if ip == ip_addr:
                return nic

        if is_dst_addr:
            # To associate (dst_addr) -> (src_addr), use a UDP socket.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((ip_addr, 80))             # Doesn't actually send data
                source_ip_addr = s.getsockname()[0]   # Get the source IP used for routing
            for nic,ip in self._parse_ip_addr_show:
                if ip == source_ip_addr:
                    return nic

        raise RuntimeError(f"Couldn't associate IP address {ip_addr} with a NIC")

    @functools.cache
    def mtu_from_nic(self, nic):
        """MTU for a NIC (e.g. 'eno8303' or 'lo'). Reads /sys/class/net/<nic>/mtu."""
        with open(f'/sys/class/net/{nic}/mtu') as f:
            return int(f.read().strip())

    @functools.cache
    def vcpu_list_from_nic(self, nic):
        bus_id = self._pcie_bus_id_from_nic(nic)   # can be None, for loopback interface 
        return self._vcpu_list_from_pcie_bus_id(bus_id, allow_none=True)
    
    @functools.cache
    def vcpu_list_from_ip_addr(self, ip_addr, is_dst_addr=False):
        nic = self.nic_from_ip_addr(ip_addr, is_dst_addr)
        return self.vcpu_list_from_nic(nic)

    
    @functools.cache
    def vcpu_list_from_disk(self, disk):
        bus_id = self._pcie_bus_id_from_block_device(disk)
        return self._vcpu_list_from_pcie_bus_id(bus_id)
    
    def vcpu_list_from_dirname(self, dirname):
        disk = self.disk_from_dirname(dirname)
        return self.vcpu_list_from_disk(disk)

    def disk_from_dirname(self, dirname):
        dev_id = os.stat(dirname).st_dev  # Device ID (major:minor)

        for d_name, d_mountpoint, d_id in self._parse_proc_mounts:
            if dev_id == d_id:
                return d_name

        raise RuntimeError(f"Couldn't find disk for dirname {dirname} (by searching /proc/mounts for st_dev={dev_id})")
        
    @functools.cache
    def mount_point_from_device(self, device_name):
        """The 'device_name' is e.g. /dev/nvme0n1p2 or just 'nvme0n1p2'."""

        for d_name, d_mountpoint, d_id in self._parse_proc_mounts:
            if os.path.basename(device_name) == os.path.basename(d_name):
                return d_mountpoint

        raise RuntimeError(f"Couldn't find mount point for device {device_name}")
    
    @functools.cached_property
    def disks(self):
        disks = []

        # Iterate over block devices.
        for device in os.listdir('/sys/class/block'):
            # Exclude loop devices.
            if device.startswith('loop'):
                continue

            # Exclude NVMe controller-specific paths (e.g. nvme0c0n1).
            # These duplicate the namespace device (e.g. nvme0n1) with the same PCIe address.
            if re.fullmatch(r'nvme\d+c\d+n\d+', device):
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

        for nic, ip_addr in self._parse_ip_addr_show:
            bus_id = self._pcie_bus_id_from_nic(nic)
            description = self._description_from_pcie_bus_id(bus_id)
            vcpu_list = self.vcpu_list_from_ip_addr(ip_addr)
            mtu = self.mtu_from_nic(nic)
            print(f'IP addr {ip_addr}')
            print(f'   nic = {nic}, mtu = {mtu}, pcie = {bus_id}  ({description})')
            print(f'   {vcpu_list = }\n')

        for disk in self.disks:
            bus_id = self._pcie_bus_id_from_block_device(disk)
            description = self._description_from_pcie_bus_id(bus_id)
            vcpu_list = self.vcpu_list_from_disk(disk)
            print(f'Disk {disk}')
            print(f'   pcie = {bus_id}  ({description})')
            print(f'   {vcpu_list = }\n')

        self._show_hugepages()


    def _show_hugepages(self):
        """Print hugepage pool sizes per page-size and per-NUMA-node.

        'unpinned' is the system-wide nr_hugepages minus the sum over
        NUMA nodes. The kernel always places each hugepage on some node,
        so this should be 0 in a healthy configuration; surfacing the
        discrepancy makes misconfigurations visible at a glance."""

        sizes = self.hugepage_sizes
        if not sizes:
            print('hugepages: not configured (no /sys/kernel/mm/hugepages tree)\n')
            return

        nodes = self.numa_nodes
        print('hugepages:')
        # Label width: 'system total', 'system surplus', 'unpinned',
        # 'node<N>'. 'system surplus' is the longest at 14 chars.
        lw = 14
        for size in sizes:
            size_str = self._fmt_bytes(size)
            pool = self.hugepage_pool(size)
            total_bytes = pool['nr']   * size
            free_bytes  = pool['free'] * size

            print(f'  page_size = {size_str}:')
            print(f'    {"system total":<{lw}} = {pool["nr"]} ({self._fmt_bytes(total_bytes)}),'
                  f' free = {pool["free"]} ({self._fmt_bytes(free_bytes)})')

            if pool['nr'] == 0 and pool['surplus'] == 0:
                # Skip per-node breakdown when nothing is configured.
                continue

            sum_nr = 0
            for node in nodes:
                npool = self.hugepage_pool_per_node(size, node)
                sum_nr += npool['nr']
                label = f'node{node}'
                nr_bytes = npool['nr'] * size
                line = (f'    {label:<{lw}} = {npool["nr"]} ({self._fmt_bytes(nr_bytes)}),'
                        f' free = {npool["free"]}')
                if npool['surplus']:
                    line += f', surplus = {npool["surplus"]}'
                print(line)

            unpinned = pool['nr'] - sum_nr
            print(f'    {"unpinned":<{lw}} = {unpinned}  '
                  f'(system total - sum over NUMA nodes)')

            if pool['surplus']:
                print(f'    {"system surplus":<{lw}} = {pool["surplus"]}')
        print()


    ################################################################################################


    @functools.cached_property
    def _parse_cpu_topology(self):
        """List of length (num_vcpus), containing physical CPU associated with each vCPU."""

        ret = [ ]
        
        for n in itertools.count():
            dirname = f'/sys/devices/system/cpu/cpu{n}'
            if not os.path.exists(dirname):
                assert n > 0
                return ret

            with open(f'{dirname}/topology/physical_package_id') as f:
                cpu_id = int(f.read().strip())
                ret.append(cpu_id)

    
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
        """
        The 'pathname' arg is e.g. '/sys/class/net/eno8303' or '/sys/class/block/nvme0n1p1'.
        Note that this function can return None -- caller should check return value.
        """

        pathname = os.path.realpath(pathname)

        # FIXME hack around NVMe namespace madness on the the CHORD FRB nodes.
        # Revisit this in the future and try to find a sane approach.
        
        while pathname.startswith('/sys/devices/'):
            if os.path.exists(pathname):
                rp = os.path.realpath(pathname)

                if rp.startswith('/sys/devices/pci'):
                    # regex to match PCIe bus IDs (e.g. '0000:03:00.0'), courtesy of chatgpt
                    pcie_regex = r'^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$'
                    for x in rp.split('/')[::-1]:
                        if re.match(pcie_regex, x):
                            return x

                    raise RuntimeError(f"Couldn't get PCIe bus id from sysfs path {rp} (maybe regex failure?)")

            # remove one character from the pathname and try again
            pathname = pathname[:-1]

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
        Given a block device (e.g., '/dev/nvme0n1' or '/dev/nvme0n1p1'), returns
        the PCIe bus id (e.g. '0000:03:00.0'). Raises exception on failure.
        """

        sys_subdir = f'/sys/class/block/{os.path.basename(device_name)}'
        ret = self._pcie_bus_id_from_sys_subdir(sys_subdir)

        if ret is not None:
            return ret

        rp = os.path.realpath(sys_subdir)
        raise RuntimeError(f"Couldn't get PCIe bus ID for block device {device_name} ({sys_subdir=}, realpath={rp}")


    @functools.cache
    def _description_from_pcie_bus_id(self, bus_id):
        if bus_id is None:
            return 'Not a PCIe device'
        for abbreviated_bus_id, description in self._parse_lspci:
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
    def _parse_ip_addr_show(self):
        """Parses the output of 'ip -o addr show' and returns a list of (nic, ip) pairs."""
        
        ret = []

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
                    ret.append((interface, ip_address))

        return ret


    @functools.cached_property
    def _parse_lspci(self):
        """Parses the output of 'lspci' and returns a list of pairs (abbreviated_bus_id, description)."""

        pairs = [ ]
        result = subprocess.run(['lspci'], capture_output=True, text=True, check=True)
        
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if parts and (len(parts) == 2):
                pairs.append(parts)

        return pairs
    

    @functools.cached_property
    def _parse_proc_mounts(self):
        """Parses /proc/mounts and returns a list of triples (device_name, mount_point, device_id).
        
        Here, the 'device_name' is e.g. '/dev/nvme0n1p2', and the device_id is the numerical ID
        returned by os.stat(dirname).st_dev."""
        
        ret = [ ]

        # FIXME figure out how remove entries that don't correspond to real block devices.
        # (/proc/mounts contains a bunch of weird stuff like /sys/kernel/tracing.)
        
        with open("/proc/mounts") as f:
            for line in f:
                device, mountpoint, *_ = line.split()
                try:
                    dev_id = os.stat(mountpoint).st_dev  # Device ID (major:minor)
                    ret.append((device, mountpoint, dev_id))
                except:
                    pass
        
        return ret


if __name__ == '__main__':
    h = Hardware()
    h.show()
