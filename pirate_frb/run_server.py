"""Implementation of 'pirate_frb run_server <config.yml>' subcommand."""

import os
import re
import time
import datetime

import yaml
import ksgpu

from .Hardware import Hardware
from .utils.ThreadAffinity import ThreadAffinity
from .core import BumpAllocator, SlabAllocator, AssembledFrameAllocator, FileWriter, Receiver
from .core import FakeXEngine, XEngineMetadata


def _parse_memory_string(s):
    """Parse a string like '256 GB' or '10 MB' into a byte count."""

    m = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]*)?)\s*(TB|GB|MB)\s*', s, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Could not parse memory string: {s!r} (expected e.g. '256 GB')")

    value = float(m.group(1))
    unit = m.group(2).upper()
    multiplier = {'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}[unit]
    return int(value * multiplier)


def _extract_ip(addr):
    """Extract the IP part from an 'ip:port' string (splits on last ':')."""

    i = addr.rfind(':')
    if i < 0:
        raise RuntimeError(f"Expected 'ip:port' string, got {addr!r}")
    return addr[:i]


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

    required_keys = [
        'num_servers', 'server_cpus', 'time_samples_per_chunk',
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

    tsc = config['time_samples_per_chunk']
    if not isinstance(tsc, int) or tsc <= 0:
        raise RuntimeError(f"{filename}: 'time_samples_per_chunk' must be a positive integer, got {tsc!r}")
    if tsc % 256 != 0:
        raise RuntimeError(f"{filename}: 'time_samples_per_chunk' must be a multiple of 256, got {tsc}")

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

    for key in ('ssd_threads_per_server', 'nfs_threads_per_server'):
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
            ip = _extract_ip(addr)
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


def run_server(config_filename):
    """Main entry point for 'pirate_frb run_server'."""

    config = _parse_config(config_filename)
    n = config['num_servers']

    print(f"Parsed config: {config_filename}")
    print(f"  num_servers = {n}")
    print(f"  time_samples_per_chunk = {config['time_samples_per_chunk']}")
    print(f"  memory_per_server = {config['memory_per_server']}")
    print(f"  use_hugepages = {config['use_hugepages']}")

    # Resolve NFS dir and create SSD/NFS dirs if needed.
    # (Must happen before _validate_hardware, which calls os.stat on ssd_dirs.)
    nfs_dir = _resolve_nfs_dir(config['nfs_dir'])
    os.makedirs(nfs_dir, exist_ok=True)
    for ssd_dir in config['ssd_dirs']:
        os.makedirs(ssd_dir, exist_ok=True)
    print(f"  nfs_dir = {nfs_dir}")

    hw = Hardware()
    cpus = config['server_cpus']
    _validate_hardware(config, hw)
    print(f"Hardware validation passed: server CPUs = {cpus}")

    capacity = config['memory_per_server_bytes']
    aflags = ksgpu.af_uhost
    if config['use_hugepages']:
        aflags |= ksgpu.af_mmap_huge

    servers = []

    try:
        for i in range(n):
            vcpu_list = hw.vcpu_list_from_cpu(cpus[i])
            num_addrs = len(config['data_ip_addrs'][i])

            print(f"\nServer {i}: CPU {cpus[i]}, vcpu_list = {vcpu_list}")
            print(f"  data_ip_addrs = {config['data_ip_addrs'][i]}")
            print(f"  rpc_ip_addr = {config['rpc_ip_addrs'][i]}")
            print(f"  ssd_dir = {config['ssd_dirs'][i]}")

            # Pin the calling thread to this CPU's vCPUs. Objects created
            # within the context manager (BumpAllocator, SlabAllocator,
            # AssembledFrameAllocator, FileWriter) will inherit this affinity
            # for any worker threads they spawn.
            with ThreadAffinity(vcpu_list):
                # BumpAllocator: pre-allocates memory (NUMA-aware due to thread affinity).
                # No threads spawned.
                bump_allocator = BumpAllocator(aflags, capacity)

                # SlabAllocator: carves bump allocator into fixed-size slabs.
                # No threads spawned.
                slab_allocator = SlabAllocator(bump_allocator, capacity)

                # AssembledFrameAllocator: manages frame allocation for receivers.
                # Spawns 1 worker thread (inherits vCPU affinity for CPU {cpus[i]}).
                allocator = AssembledFrameAllocator(slab_allocator, num_consumers=num_addrs)

                # FileWriter: writes frames to SSD and copies to NFS.
                # Spawns ssd_threads + nfs_threads worker threads
                # (inherit vCPU affinity for CPU {cpus[i]}).
                file_writer = FileWriter(
                    config['ssd_dirs'][i],
                    nfs_dir,
                    num_ssd_threads=config['ssd_threads_per_server'],
                    num_nfs_threads=config['nfs_threads_per_server'],
                )

                # Receivers: one per data IP address. No threads spawned in constructor.
                receivers = []
                for j, addr in enumerate(config['data_ip_addrs'][i]):
                    receiver = Receiver(
                        address=addr,
                        time_samples_per_chunk=config['time_samples_per_chunk'],
                        allocator=allocator,
                        consumer_id=j,
                    )
                    receivers.append(receiver)

                # FrbServer: ties together receivers, file writer, and RPC.
                # No threads spawned in constructor.
                # (Imported here to avoid circular import with __init__.py.)
                from . import FrbServer
                server = FrbServer(receivers, file_writer, config['rpc_ip_addrs'][i])

                # server.start(): spawns worker/reaper threads and calls
                # receiver.start() for each receiver (3 threads each).
                # All threads inherit vCPU affinity for CPU {cpus[i]}.
                server.start()

            servers.append(server)
            print(f"  Server {i} started.")

        rpc_addrs = ' '.join(config['rpc_ip_addrs'])
        print(f"\nTo send fake data to server(s):  pirate_frb run_server -s {config_filename}")
        print(f"To monitor status:               pirate_frb rpc_status {rpc_addrs}")
        print(f"To write random data:            pirate_frb rpc_write {rpc_addrs}")
        
        print(f"\nReminder: the only way to interact with running server(s) is via RPC, see above.")
        print(f"All {n} server(s) started. Press Ctrl-C to stop.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        for server in servers:
            server.stop()
        print("All servers stopped.")


def _parse_fake_xengine_config(filename, config):
    """Parse and validate the 'Fake X-engine' section of a run_server config.

    Returns a dict with keys: beams_per_server, base_beam_id,
    tcp_connections_per_server, zone_nfreq, zone_freq_edges.
    """

    n = config['num_servers']
    fxe_keys = ['beams_per_server', 'base_beam_id', 'tcp_connections_per_server',
                'zone_nfreq', 'zone_freq_edges']

    missing = [k for k in fxe_keys if k not in config]
    if missing:
        raise RuntimeError(
            f"{filename}: fake X-engine config requires key(s): {', '.join(missing)}\n"
            f"  (needed for 'pirate_frb run_server -s')"
        )

    bps = config['beams_per_server']
    if not isinstance(bps, int) or bps <= 0:
        raise RuntimeError(f"{filename}: 'beams_per_server' must be a positive integer, got {bps!r}")

    bbi = config['base_beam_id']
    if not isinstance(bbi, list) or len(bbi) != n:
        raise RuntimeError(f"{filename}: 'base_beam_id' must be a list of length {n}")
    for i, v in enumerate(bbi):
        if not isinstance(v, int) or isinstance(v, bool):
            raise RuntimeError(f"{filename}: base_beam_id[{i}] must be an integer, got {v!r}")

    tcs = config['tcp_connections_per_server']
    if not isinstance(tcs, int) or tcs <= 0:
        raise RuntimeError(f"{filename}: 'tcp_connections_per_server' must be a positive integer, got {tcs!r}")

    znf = config['zone_nfreq']
    if not isinstance(znf, list) or len(znf) == 0:
        raise RuntimeError(f"{filename}: 'zone_nfreq' must be a non-empty list of positive integers")
    for i, v in enumerate(znf):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise RuntimeError(f"{filename}: zone_nfreq[{i}] must be a positive integer, got {v!r}")

    zfe = config['zone_freq_edges']
    if not isinstance(zfe, list) or len(zfe) != len(znf) + 1:
        raise RuntimeError(f"{filename}: 'zone_freq_edges' must be a list of length {len(znf)+1} (len(zone_nfreq)+1)")
    for i, v in enumerate(zfe):
        if not isinstance(v, (int, float)):
            raise RuntimeError(f"{filename}: zone_freq_edges[{i}] must be a number, got {v!r}")
    for i in range(len(zfe) - 1):
        if zfe[i] >= zfe[i+1]:
            raise RuntimeError(f"{filename}: 'zone_freq_edges' must be monotonically increasing, "
                               f"but zone_freq_edges[{i}]={zfe[i]} >= zone_freq_edges[{i+1}]={zfe[i+1]}")

    return {k: config[k] for k in fxe_keys}


def _make_xengine_metadata(fxe_config, server_index):
    """Create an XEngineMetadata for one FakeXEngine instance."""

    xmd = XEngineMetadata()
    xmd.version = 1
    xmd.zone_nfreq = fxe_config['zone_nfreq']
    xmd.zone_freq_edges = [float(x) for x in fxe_config['zone_freq_edges']]

    nbeams = fxe_config['beams_per_server']
    base = fxe_config['base_beam_id'][server_index]
    xmd.nbeams = nbeams
    xmd.beam_ids = list(range(base, base + nbeams))

    xmd.validate()
    return xmd


def run_fake_xengine(config_filename):
    """Main entry point for 'pirate_frb run_server -s'."""

    config = _parse_config(config_filename)
    n = config['num_servers']
    fxe_config = _parse_fake_xengine_config(config_filename, config)

    hw = Hardware()
    nthreads = fxe_config['tcp_connections_per_server']

    print(f"Parsed config: {config_filename}")
    print(f"  num_servers = {n}")
    print(f"  tcp_connections_per_server = {nthreads}")

    fake_xengines = []

    try:
        for i in range(n):
            ip_addrs = config['data_ip_addrs'][i]

            # Check that all destination IPs for this server route through
            # NICs on the same physical CPU.
            vcpu_list = None
            for addr in ip_addrs:
                ip = _extract_ip(addr)
                vl = hw.vcpu_list_from_ip_addr(ip, is_dst_addr=True)
                cpu = hw.cpu_from_vcpu_list(vl)
                if vcpu_list is None:
                    vcpu_list = vl
                    first_cpu = cpu
                elif cpu != first_cpu:
                    raise RuntimeError(
                        f"FakeXEngine {i}: destination IPs {ip_addrs} route through "
                        f"NICs on different CPUs (need all on one CPU)"
                    )

            xmd = _make_xengine_metadata(fxe_config, i)

            print(f"\nFakeXEngine {i}:")
            print(f"  ip_addrs = {ip_addrs}")
            print(f"  nthreads = {nthreads}")
            print(f"  nbeams = {xmd.nbeams}, beam_ids = {xmd.beam_ids}")
            print(f"  total_nfreq = {xmd.get_total_nfreq()}")

            # Pin thread to sender NIC's CPU for NUMA-local thread creation.
            with ThreadAffinity(vcpu_list):
                fxe = FakeXEngine(xmd, ip_addrs, nthreads)
                fxe.start()

            fake_xengines.append(fxe)
            print(f"  FakeXEngine {i} started.")

        print(f"\nAll {n} FakeXEngine(s) started. Press Ctrl-C to stop.")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for fxe in fake_xengines:
            fxe.stop()
        print("All FakeXEngine(s) stopped.")
