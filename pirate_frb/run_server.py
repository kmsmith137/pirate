"""Implementation of 'pirate_frb run_server' and 'pirate_frb run_fake_xengine' subcommands."""

import os
import re
import time
import datetime
import threading
from contextlib import closing

import yaml
import ksgpu

from .Hardware import Hardware
from .utils.ThreadAffinity import ThreadAffinity
from .core import BumpAllocator, SlabAllocator, AssembledFrameAllocator, FileWriter, Receiver
from .core import FakeXEngine, XEngineMetadata
from .pirate_pybind11 import DedispersionConfig
from .rpc import FrbClient


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


def _check_mtu(hw, label, ip_addr, min_mtu, min_mtu_param, is_dst_addr=False):
    """Raise RuntimeError if the NIC routing for 'ip_addr' has MTU below min_mtu.

    'label' is a free-form descriptor (e.g. 'FrbServer 0 data[1]') shown in
    the exception text. 'min_mtu_param' is the YAML key name (e.g.
    'min_data_mtu') so the error message points the user at the right knob.
    Set is_dst_addr=True for FakeXEngine destinations.
    """
    nic = hw.nic_from_ip_addr(ip_addr, is_dst_addr=is_dst_addr)
    mtu = hw.mtu_from_nic(nic)
    if mtu < min_mtu:
        raise RuntimeError(
            f"{label}: NIC {nic!r} ({ip_addr}) has MTU {mtu}, below the required "
            f"minimum {min_mtu} (config param {min_mtu_param!r}).\n"
            f"  - If the small MTU is intentional, lower {min_mtu_param!r} in the "
            f"server YAML config to <= {mtu}.\n"
            f"  - If the small MTU is unintentional, reconfigure the NIC to MTU "
            f">= {min_mtu} (e.g. 'sudo ip link set {nic} mtu {min_mtu}')."
        )


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


def run_server(server_config_filename, dedispersion_config_filename):
    """Main entry point for 'pirate_frb run_server'."""

    config = _parse_config(server_config_filename)
    dedisp_config = DedispersionConfig.from_yaml(dedispersion_config_filename)
    n = config['num_servers']

    print(f"Parsed server config: {server_config_filename}")
    print(f"Parsed dedispersion config: {dedispersion_config_filename}")
    print(f"  num_servers = {n}")
    print(f"  time_samples_per_chunk = {dedisp_config.time_samples_per_chunk}  (from dedispersion config)")
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

            # Enforce minimum MTU on the local data and RPC NICs.
            for j, addr in enumerate(config['data_ip_addrs'][i]):
                _check_mtu(hw, f"FrbServer {i} data[{j}]", _extract_ip(addr),
                           config['min_data_mtu'], 'min_data_mtu')
            _check_mtu(hw, f"FrbServer {i} rpc",
                       _extract_ip(config['rpc_ip_addrs'][i]),
                       config['min_rpc_mtu'], 'min_rpc_mtu')

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
                allocator = AssembledFrameAllocator(
                    slab_allocator,
                    num_consumers=num_addrs,
                    time_samples_per_chunk=dedisp_config.time_samples_per_chunk,
                )

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
                        allocator=allocator,
                        consumer_id=j,
                    )
                    receivers.append(receiver)

                # FrbServer: ties together receivers, file writer, and RPC.
                # No threads spawned in constructor.
                # (Imported here to avoid circular import with __init__.py.)
                from . import FrbServer
                server = FrbServer(dedisp_config, receivers, file_writer,
                                   config['rpc_ip_addrs'][i],
                                   config['ringbuf_nchunks'],
                                   min_data_mtu=config['min_data_mtu'])

                # server.start(): spawns worker/reaper threads and calls
                # receiver.start() for each receiver (3 threads each).
                # All threads inherit vCPU affinity for CPU {cpus[i]}.
                server.start()

            servers.append(server)
            print(f"  Server {i} started.")

        rpc_addrs = ' '.join(config['rpc_ip_addrs'])
        print()
        for addr in config['rpc_ip_addrs']:
            print(f"To send fake data to {addr}:  pirate_frb run_fake_xengine {addr}")
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


def _fake_xengine_controller_main(fxe):
    """
    Drive one FakeXEngine in 'send-junk-forever' mode.

    Reproduces the cross-worker "minichunk N waits for (N-2)" serialization
    that the C++ FakeXEngine used to enforce internally with its barrier.
    Runs until fxe.stop() is called (from anywhere), at which point the
    next wait_until_processed() / enqueue_send_junk() call raises
    RuntimeError and the function returns via exception.
    """
    nworkers = fxe.nworkers
    n = 0
    while True:
        # Wait for every worker to have caught up to (n-2). Negative
        # indices return immediately (per-worker last_processed_minichunk
        # starts at -1).
        for w in range(nworkers):
            fxe.wait_until_processed(w, n - 2)
        # Then submit minichunk n on every worker.
        for w in range(nworkers):
            fxe.enqueue_send_junk(w, n)
        n += 1


def _fake_xengine_controller_wrapper(fxe, exc_list, exc_lock):
    """Wraps _fake_xengine_controller_main. On exit (normal or exceptional),
    stops the FakeXEngine and captures any exception under exc_lock for
    the main thread to re-raise.
    """
    try:
        _fake_xengine_controller_main(fxe)
    except BaseException as e:
        with exc_lock:
            exc_list.append(e)
    finally:
        try:
            fxe.stop()
        except Exception:
            pass


def run_fake_xengine(rpc_addr, nworkers=128):
    """Main entry point for 'pirate_frb run_fake_xengine'.

    Connects to the receiver at 'rpc_addr', sends a GetConfig RPC to
    learn data_ip_addrs, time_samples_per_chunk, min_data_mtu, and the
    fake_* fields needed to synthesize an XEngineMetadata. Then pins to
    the data NIC's CPU and spawns one FakeXEngine plus one controller
    thread that drives the SEND_JUNK loop.
    """

    # ---- Phase 1: GetConfig ----
    print(f"Connecting to receiver at {rpc_addr} ...")
    with closing(FrbClient(rpc_addr)) as c:
        cfg = c.get_config()

    ip_addrs = list(cfg.data_ip_addrs)
    if not ip_addrs:
        raise RuntimeError(f"run_fake_xengine: receiver at {rpc_addr} reported empty data_ip_addrs")
    time_samples_per_chunk = cfg.time_samples_per_chunk
    min_data_mtu = cfg.min_data_mtu

    # Synthesize XEngineMetadata from the receiver's prefilled config.
    # beam_ids = {0, 1, ..., nbeams-1} -- the receiver records whatever
    # we send, so no cross-X-engine consensus to satisfy.
    nbeams = cfg.fake_nbeams
    beam_ids = list(range(nbeams))
    xmd = XEngineMetadata.make_test_instance(
        list(cfg.fake_zone_nfreq),
        list(cfg.fake_zone_freq_edges),
        beam_ids,
        cfg.fake_time_sample_ms,
    )

    actual_time_sample_ms = (xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample) / 1.0e6
    print(f"  data_ip_addrs = {ip_addrs}")
    print(f"  time_samples_per_chunk = {time_samples_per_chunk}")
    print(f"  min_data_mtu = {min_data_mtu}")
    print(f"  nbeams = {nbeams}, total_nfreq = {xmd.get_total_nfreq()}")
    print(f"  time_sample_ms = {actual_time_sample_ms}")
    print(f"  dt_ns_per_seq = {xmd.dt_ns_per_seq}")
    print(f"  seq_per_frb_time_sample = {xmd.seq_per_frb_time_sample}")

    # ---- Phase 2: hardware checks + CPU affinity ----
    hw = Hardware()
    vcpu_list = None
    first_cpu = None
    for addr in ip_addrs:
        ip = _extract_ip(addr)
        vl = hw.vcpu_list_from_ip_addr(ip, is_dst_addr=True)
        _check_mtu(hw, f"FakeXEngine -> {addr}", ip,
                   min_data_mtu, 'min_data_mtu', is_dst_addr=True)
        cpu = hw.cpu_from_vcpu_list(vl)
        if vcpu_list is None:
            vcpu_list = vl
            first_cpu = cpu
        elif cpu != first_cpu:
            raise RuntimeError(
                f"FakeXEngine: destination IPs {ip_addrs} route through "
                f"NICs on different CPUs (need all on one CPU)"
            )

    # ---- Phase 3: build the FakeXEngine + controller under affinity ----
    exc_list = []   # list[BaseException]
    exc_lock = threading.Lock()
    with ThreadAffinity(vcpu_list):
        fxe = FakeXEngine(
            xmd, ip_addrs, nworkers,
            time_samples_per_chunk=time_samples_per_chunk)
        t = threading.Thread(
            target=_fake_xengine_controller_wrapper,
            args=(fxe, exc_list, exc_lock),
            daemon=True,
        )
        t.start()

    print(f"\nFakeXEngine running ({nworkers} workers). Press Ctrl-C to stop.")

    try:
        while t.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")

    try:
        fxe.stop()
    except Exception:
        pass
    t.join(timeout=5.0)

    with exc_lock:
        # A "called on stopped instance" RuntimeError is the controller's
        # natural teardown artefact: stop() invalidates the next
        # enqueue_send_junk() call. Re-raise only "real" exceptions.
        real = [e for e in exc_list if "called on stopped instance" not in str(e)]
        if real:
            raise real[0]

    print("FakeXEngine stopped.")
