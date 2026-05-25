"""Implementation of 'pirate_frb run_server' subcommand."""

import os
import re
import time
import datetime

import yaml
import ksgpu

from .Hardware import Hardware
from .utils import ThreadAffinity, extract_ip, check_mtu
from .core import BumpAllocator, SlabAllocator, AssembledFrameAllocator, FileWriter, Receiver
from .pirate_pybind11 import DedispersionConfig


def _parse_memory_string(s):
    """Parse a string like '256 GB' or '10 MB' into a byte count."""

    m = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]*)?)\s*(TB|GB|MB)\s*', s, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Could not parse memory string: {s!r} (expected e.g. '256 GB')")

    value = float(m.group(1))
    unit = m.group(2).upper()
    multiplier = {'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}[unit]
    return int(value * multiplier)


def _resolve_nfs_dir(template):
    """Interpolate {user} and {date} in an NFS directory template."""

    user = os.environ.get('USER', 'unknown')
    date = datetime.date.today().strftime('%Y-%m-%d')
    return template.replace('{user}', user).replace('{date}', date)


def _parse_config(filename):
    """Parse and validate a run_server YAML config file. Returns a dict."""

    with open(filename) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise RuntimeError(f"{filename}: expected YAML mapping at top level, got {type(config).__name__}")

    # --- Required keys and their types ---
    #
    # Note: 'time_samples_per_chunk' used to live here, but now belongs to the
    # DedispersionConfig. run_server() reads it from the dedispersion YAML;
    # run_fake_xengine() queries it from the running server via GetConfig RPC.

    required_keys = [
        'num_servers', 'server_cpus',
        'memory_per_server', 'use_hugepages', 'data_ip_addrs', 'rpc_ip_addrs',
        'ssd_dirs', 'ssd_devices', 'ssd_threads_per_server',
        'nfs_dir', 'nfs_threads_per_server',
    ]

    missing = [k for k in required_keys if k not in config]
    if missing:
        raise RuntimeError(f"{filename}: missing required key(s): {', '.join(missing)}")

    n = config['num_servers']

    # --- Validate individual keys ---

    if not isinstance(n, int) or n <= 0:
        raise RuntimeError(f"{filename}: 'num_servers' must be a positive integer, got {n!r}")

    sc = config['server_cpus']
    if not isinstance(sc, list) or len(sc) != n:
        raise RuntimeError(f"{filename}: 'server_cpus' must be a list of length {n}")
    for i, cpu in enumerate(sc):
        if not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0:
            raise RuntimeError(f"{filename}: server_cpus[{i}] must be a non-negative integer, got {cpu!r}")

    mps = config['memory_per_server']
    if not isinstance(mps, str):
        raise RuntimeError(f"{filename}: 'memory_per_server' must be a string like '256 GB', got {mps!r}")
    config['memory_per_server_bytes'] = _parse_memory_string(mps)

    if not isinstance(config['use_hugepages'], bool):
        raise RuntimeError(f"{filename}: 'use_hugepages' must be a boolean, got {config['use_hugepages']!r}")

    # data_ip_addrs: list of length num_servers, each element is a string or list of strings.
    dia = config['data_ip_addrs']
    if not isinstance(dia, list) or len(dia) != n:
        raise RuntimeError(f"{filename}: 'data_ip_addrs' must be a list of length {n}, got length {len(dia) if isinstance(dia, list) else type(dia).__name__}")

    # Normalize: bare string -> one-element list.
    for i in range(n):
        if isinstance(dia[i], str):
            dia[i] = [dia[i]]
        elif isinstance(dia[i], list):
            for j, addr in enumerate(dia[i]):
                if not isinstance(addr, str):
                    raise RuntimeError(f"{filename}: data_ip_addrs[{i}][{j}] must be a string, got {type(addr).__name__}")
        else:
            raise RuntimeError(f"{filename}: data_ip_addrs[{i}] must be a string or list of strings, got {type(dia[i]).__name__}")

    rpc = config['rpc_ip_addrs']
    if not isinstance(rpc, list) or len(rpc) != n:
        raise RuntimeError(f"{filename}: 'rpc_ip_addrs' must be a list of length {n}")
    for i, addr in enumerate(rpc):
        if not isinstance(addr, str):
            raise RuntimeError(f"{filename}: rpc_ip_addrs[{i}] must be a string, got {type(addr).__name__}")

    for key in ('ssd_dirs', 'ssd_devices'):
        val = config[key]
        if not isinstance(val, list) or len(val) != n:
            raise RuntimeError(f"{filename}: '{key}' must be a list of length {n}")
        for i, v in enumerate(val):
            if not isinstance(v, str):
                raise RuntimeError(f"{filename}: {key}[{i}] must be a string, got {type(v).__name__}")

    for key in ('ssd_threads_per_server', 'nfs_threads_per_server',
                'min_data_mtu', 'min_rpc_mtu', 'ringbuf_nchunks'):
        val = config[key]
        if not isinstance(val, int) or val <= 0:
            raise RuntimeError(f"{filename}: '{key}' must be a positive integer, got {val!r}")

    if not isinstance(config['nfs_dir'], str):
        raise RuntimeError(f"{filename}: 'nfs_dir' must be a string, got {type(config['nfs_dir']).__name__}")

    return config


def _validate_hardware(config, hw):
    """Check that all data_ip_addrs and ssd_dirs are consistent with server_cpus.

    For each server, verifies that the NIC(s) and SSD are on the physical CPU
    specified by server_cpus[i]. Devices with no CPU affinity (loopback NICs,
    tmpfs directories) automatically pass the consistency check.
    """

    n = config['num_servers']
    cpus = config['server_cpus']

    for i in range(n):
        expected_cpu = cpus[i]

        # Check data IP addresses.
        for addr in config['data_ip_addrs'][i]:
            ip = extract_ip(addr)
            vcpus = hw.vcpu_list_from_ip_addr(ip)
            cpu = hw.cpu_from_vcpu_list(vcpus)
            # cpu is None for loopback (no PCIe affinity, spans all CPUs).
            if (cpu is not None) and (cpu != expected_cpu):
                raise RuntimeError(
                    f"Server {i}: data IP {addr} is on CPU {cpu}, "
                    f"but server_cpus[{i}] = {expected_cpu}"
                )

        # Check SSD dir CPU affinity.
        ssd_dir = config['ssd_dirs'][i]
        try:
            ssd_vcpus = hw.vcpu_list_from_dirname(ssd_dir)
            ssd_cpu = hw.cpu_from_vcpu_list(ssd_vcpus)
        except RuntimeError:
            # Non-PCIe filesystem (e.g. tmpfs) has no CPU affinity.
            ssd_cpu = None

        if (ssd_cpu is not None) and (ssd_cpu != expected_cpu):
            raise RuntimeError(
                f"Server {i}: SSD dir {ssd_dir!r} is on CPU {ssd_cpu}, "
                f"but server_cpus[{i}] = {expected_cpu}"
            )

        # Verify SSD device matches.
        actual_dev = hw.disk_from_dirname(ssd_dir)
        expected_dev = config['ssd_devices'][i]
        if os.path.basename(actual_dev) != os.path.basename(expected_dev):
            raise RuntimeError(
                f"Server {i}: ssd_dirs[{i}]={ssd_dir!r} is backed by "
                f"device {actual_dev!r}, but ssd_devices[{i}]={expected_dev!r} (mismatch)"
            )


class RunServerHelper:
    """Encapsulates state and logic for 'pirate_frb run_server'.

    Constructed once per invocation; call .run() to drive the full
    lifecycle (config parsing happens in __init__; hardware checks,
    construction, and the Ctrl-C wait loop happen in run()).
    """

    def __init__(self, server_config_filename, dedispersion_config_filename):
        self.config = _parse_config(server_config_filename)
        self.dedisp_config = DedispersionConfig.from_yaml(dedispersion_config_filename)
        self.n = self.config['num_servers']
        self.hw = Hardware()
        # Populated later by run() -> _prepare_directories / _setup_memory.
        self.nfs_dir = None
        self.capacity = None
        self.aflags = None
        self.servers = []
        self._print_config_summary(server_config_filename, dedispersion_config_filename)

    def run(self):
        """Top-level lifecycle: prepare, build all servers, wait for Ctrl-C,
        then stop everything (including any partially-built state)."""
        self._prepare_directories()
        self._run_hardware_check()
        self._setup_memory()
        try:
            self._build_all_servers()
            self._print_help_lines()
            self._wait_forever()
        except KeyboardInterrupt:
            print("\nStopping servers...")
        finally:
            self._stop_all_servers()

    def _print_config_summary(self, server_config_filename, dedispersion_config_filename):
        print(f"Parsed server config: {server_config_filename}")
        print(f"Parsed dedispersion config: {dedispersion_config_filename}")
        print(f"  num_servers = {self.n}")
        print(f"  time_samples_per_chunk = {self.dedisp_config.time_samples_per_chunk}  (from dedispersion config)")
        print(f"  memory_per_server = {self.config['memory_per_server']}")
        print(f"  use_hugepages = {self.config['use_hugepages']}")

    def _prepare_directories(self):
        # Resolve NFS dir and create SSD/NFS dirs if needed.
        # (Must happen before _run_hardware_check, which calls os.stat on ssd_dirs.)
        self.nfs_dir = _resolve_nfs_dir(self.config['nfs_dir'])
        os.makedirs(self.nfs_dir, exist_ok=True)
        for ssd_dir in self.config['ssd_dirs']:
            os.makedirs(ssd_dir, exist_ok=True)
        print(f"  nfs_dir = {self.nfs_dir}")

    def _run_hardware_check(self):
        _validate_hardware(self.config, self.hw)
        print(f"Hardware validation passed: server CPUs = {self.config['server_cpus']}")

    def _setup_memory(self):
        self.capacity = self.config['memory_per_server_bytes']
        self.aflags = ksgpu.af_uhost
        if self.config['use_hugepages']:
            self.aflags |= ksgpu.af_mmap_huge

    def _build_all_servers(self):
        for i in range(self.n):
            self._build_server(i)

    def _build_server(self, i):
        cpus = self.config['server_cpus']
        vcpu_list = self.hw.vcpu_list_from_cpu(cpus[i])

        self._print_server_details(i, vcpu_list)
        self._check_mtus_for_server(i)

        # Pin the calling thread to this CPU's vCPUs. Objects created
        # within the context manager (BumpAllocator, SlabAllocator,
        # AssembledFrameAllocator, FileWriter) will inherit this affinity
        # for any worker threads they spawn.
        with ThreadAffinity(vcpu_list):
            num_addrs = len(self.config['data_ip_addrs'][i])
            # BumpAllocator: pre-allocates memory (NUMA-aware due to thread
            # affinity). No threads spawned.
            bump_allocator = BumpAllocator(self.aflags, self.capacity)
            # SlabAllocator: carves bump allocator into fixed-size slabs.
            # No threads spawned.
            slab_allocator = SlabAllocator(bump_allocator, self.capacity)
            # AssembledFrameAllocator: manages frame allocation for receivers.
            # Spawns 1 worker thread (inherits vCPU affinity).
            allocator = AssembledFrameAllocator(
                slab_allocator,
                num_consumers=num_addrs,
                time_samples_per_chunk=self.dedisp_config.time_samples_per_chunk,
            )
            # FileWriter: writes frames to SSD and copies to NFS.
            # Spawns ssd_threads + nfs_threads worker threads.
            file_writer = FileWriter(
                self.config['ssd_dirs'][i],
                self.nfs_dir,
                num_ssd_threads=self.config['ssd_threads_per_server'],
                num_nfs_threads=self.config['nfs_threads_per_server'],
            )
            # Receivers: one per data IP address. No threads spawned in ctor.
            receivers = [
                Receiver(address=addr, allocator=allocator, consumer_id=j)
                for j, addr in enumerate(self.config['data_ip_addrs'][i])
            ]
            # FrbServer: ties together receivers, file writer, and RPC.
            # (Imported here to avoid circular import with __init__.py.)
            from . import FrbServer
            server = FrbServer(self.dedisp_config, receivers, file_writer,
                               self.config['rpc_ip_addrs'][i],
                               self.config['ringbuf_nchunks'],
                               min_data_mtu=self.config['min_data_mtu'])
            # server.start(): spawns worker/reaper/processing threads and
            # calls receiver.start() for each receiver. All threads inherit
            # the vCPU affinity for this CPU.
            server.start()

        self.servers.append(server)
        print(f"  Server {i} started.")

    def _print_server_details(self, i, vcpu_list):
        cpus = self.config['server_cpus']
        print(f"\nServer {i}: CPU {cpus[i]}, vcpu_list = {vcpu_list}")
        print(f"  data_ip_addrs = {self.config['data_ip_addrs'][i]}")
        print(f"  rpc_ip_addr = {self.config['rpc_ip_addrs'][i]}")
        print(f"  ssd_dir = {self.config['ssd_dirs'][i]}")

    def _check_mtus_for_server(self, i):
        # Enforce minimum MTU on the local data and RPC NICs.
        for j, addr in enumerate(self.config['data_ip_addrs'][i]):
            check_mtu(self.hw, f"FrbServer {i} data[{j}]", extract_ip(addr),
                      self.config['min_data_mtu'], 'min_data_mtu')
        check_mtu(self.hw, f"FrbServer {i} rpc",
                  extract_ip(self.config['rpc_ip_addrs'][i]),
                  self.config['min_rpc_mtu'], 'min_rpc_mtu')

    def _print_help_lines(self):
        rpc_addrs = ' '.join(self.config['rpc_ip_addrs'])
        print(f"\nTo send fake data to server(s):  pirate_frb run_fake_xengine {rpc_addrs}")
        print(f"To monitor status:               pirate_frb rpc_status {rpc_addrs}")
        print(f"To write random data:            pirate_frb rpc_write {rpc_addrs}")

        print(f"\nReminder: the only way to interact with running server(s) is via RPC, see above.")
        print(f"All {self.n} server(s) started. Press Ctrl-C to stop.")

    def _wait_forever(self):
        while True:
            time.sleep(1)

    def _stop_all_servers(self):
        for server in self.servers:
            server.stop()
        print("All servers stopped.")


def run_server(server_config_filename, dedispersion_config_filename):
    """Main entry point for 'pirate_frb run_server'."""
    helper = RunServerHelper(server_config_filename, dedispersion_config_filename)
    helper.run()

