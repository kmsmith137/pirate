"""Implementation of 'pirate_frb run_server' subcommand."""

import os
import re
import time
import datetime

import cupy as cp
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


def compute_async_bump_nthreads(vcpu_list, nbytes):
    """nthreads = max(2, min(len(vcpu_list)//2, nbytes // 128MiB)).

    Worker-count formula shared between run_server and time_dedisperser
    when constructing an async BumpAllocator. The 128 MiB lower bound on
    bytes-per-thread keeps tiny allocators from spawning more workers
    than they have work for; the len(vcpu_list)//2 cap leaves vCPUs free
    for the actual server / timing-loop work.
    """
    nthreads = len(vcpu_list) // 2
    nthreads = min(nthreads, nbytes // (128 * 2**20))
    return max(2, nthreads)


def _cuda_device_from_cpu(hw, cpu_id):
    """Return the unique CUDA device on physical CPU `cpu_id`.

    Throws a verbose RuntimeError if zero or two-or-more GPUs are
    associated with the CPU.

    Assumes a one-GPU-per-CPU topology, which is true for the CHORD FRB
    nodes today. If a future deployment has multiple GPUs per CPU (or
    zero GPUs on a server's pinned CPU), the operator will need a
    per-server `cuda_devices` YAML field; until then we auto-derive.
    """
    matches = []
    for gpu in range(hw.num_gpus):
        vcpu_list = hw.vcpu_list_from_gpu(gpu)
        if hw.cpu_from_vcpu_list(vcpu_list) == cpu_id:
            matches.append(gpu)

    if len(matches) == 0:
        gpu_summary = ", ".join(
            f"GPU {g}->CPU {hw.cpu_from_vcpu_list(hw.vcpu_list_from_gpu(g))}"
            for g in range(hw.num_gpus)
        )
        raise RuntimeError(
            f"_cuda_device_from_cpu(cpu_id={cpu_id}): no CUDA devices "
            f"found on CPU {cpu_id}. "
            f"Available: {gpu_summary or '(no GPUs)'}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"_cuda_device_from_cpu(cpu_id={cpu_id}): expected exactly "
            f"one CUDA device on CPU {cpu_id}, found {matches} "
            f"(this helper assumes one GPU per CPU)"
        )

    return matches[0]


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
        'rb_host_memory_per_server', 'dd_host_memory_per_server',
        'gpu_memory_per_server',
        'use_hugepages', 'data_ip_addrs', 'rpc_ip_addrs', 'grouper_ip_addrs',
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

    for key in ('rb_host_memory_per_server',
                'dd_host_memory_per_server',
                'gpu_memory_per_server'):
        val = config[key]
        if not isinstance(val, str):
            raise RuntimeError(
                f"{filename}: {key!r} must be a string like '256 GB', got {val!r}"
            )
        config[f'{key}_bytes'] = _parse_memory_string(val)

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

    # grouper_ip_addrs: same shape as rpc_ip_addrs (one loopback 'ip:port' per
    # server). The FrbServer is the gRPC *client* of the FrbGrouper; the grouper
    # (downstream consumer) is the server. '--no-grouper' overrides these to ''.
    gia = config['grouper_ip_addrs']
    if not isinstance(gia, list) or len(gia) != n:
        raise RuntimeError(f"{filename}: 'grouper_ip_addrs' must be a list of length {n}")
    for i, addr in enumerate(gia):
        if not isinstance(addr, str):
            raise RuntimeError(f"{filename}: grouper_ip_addrs[{i}] must be a string, got {type(addr).__name__}")

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

        # Fail-fast check that there's exactly one CUDA device on this CPU.
        # The same call happens again in _build_server (cheap; the underlying
        # Hardware accessors are cached) when we actually need the value.
        _cuda_device_from_cpu(hw, expected_cpu)

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

    def __init__(self, server_config_filename, dedispersion_config_filename,
                 processing_delay_sec=0.0, no_grouper=False):
        self.config = _parse_config(server_config_filename)
        self.dedisp_config = DedispersionConfig.from_yaml(dedispersion_config_filename)
        self.n = self.config['num_servers']
        self.hw = Hardware()
        self.processing_delay_sec = processing_delay_sec
        self.no_grouper = no_grouper
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
        print(f"  rb_host_memory_per_server = {self.config['rb_host_memory_per_server']}")
        print(f"  dd_host_memory_per_server = {self.config['dd_host_memory_per_server']}")
        print(f"  gpu_memory_per_server    = {self.config['gpu_memory_per_server']}")
        print(f"  use_hugepages = {self.config['use_hugepages']}")
        if self.processing_delay_sec > 0.0:
            print(f"  processing_delay_sec = {self.processing_delay_sec}  (artificial per-frame delay)")

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
        # All three host/GPU BumpAllocators are constructed in async mode so
        # they initialize concurrently (zeroing + cudaHostRegister + cudaMalloc
        # overlap across servers). Async mode requires af_zero, so we add it
        # to rb_host (the receiver overwrites the memory, so zeroing isn't
        # functionally required -- but with async zeroing the cost is hidden
        # behind concurrent inits, so the previous concern about startup
        # stalls no longer applies).
        self.rb_host_aflags = ksgpu.af_rhost | ksgpu.af_zero
        # dd_host (GpuDedisperser host buffers): af_rhost + af_zero, per
        # GpuDedisperser::allocate()'s requirement.
        self.dd_host_aflags = ksgpu.af_rhost | ksgpu.af_zero
        if self.config['use_hugepages']:
            self.rb_host_aflags |= ksgpu.af_mmap_huge
            self.dd_host_aflags |= ksgpu.af_mmap_huge
        # gpu: af_gpu + af_zero, per GpuDedisperser::allocate()'s requirement.
        self.gpu_aflags = ksgpu.af_gpu | ksgpu.af_zero

    def _build_all_servers(self):
        # Phase 1 happens inside _build_server: all 3 async BumpAllocators per
        # server kick off and return immediately. SlabAllocator and
        # AssembledFrameAllocator are also constructed (their ctors don't
        # block because the SlabAllocator defers its allocate_bytes call to
        # first get_slab(), which won't happen until server.start()).
        for i in range(self.n):
            self._build_server(i)

        # Phase 2: wait for all 3 * num_servers async BumpAllocators to
        # finish initializing. Any async-init failures surface here with a
        # clean stack trace through run_server rather than from a server
        # worker thread later.
        for i, server in enumerate(self.servers):
            print(f"Waiting for server {i}'s async BumpAllocators to initialize...")
            server.host_allocator.wait_until_initialized()
            server.gpu_allocator.wait_until_initialized()
            # rb_bump is held in self._rb_bumps[i] so we can wait on it here.
            self._rb_bumps[i].wait_until_initialized()
            print(f"  Server {i} ready.")

        # Phase 3: start all servers.
        for i, server in enumerate(self.servers):
            server.start()
            print(f"  Server {i} started.")

    def _build_server(self, i):
        cpus = self.config['server_cpus']
        vcpu_list = self.hw.vcpu_list_from_cpu(cpus[i])
        cuda_device_id = _cuda_device_from_cpu(self.hw, cpus[i])

        self._print_server_details(i, vcpu_list, cuda_device_id)
        self._check_mtus_for_server(i)

        # Pin the calling thread to this CPU's vCPUs AND set the current
        # CUDA device. Objects created within the context manager
        # (BumpAllocators, SlabAllocator, AssembledFrameAllocator,
        # FileWriter) will inherit this affinity for any worker threads
        # they spawn; the gpu_alloc's cudaMalloc will land on the correct
        # device. cp.cuda.Device restores the previous device on exit, so
        # subsequent server builds (num_servers > 1) start clean.
        with ThreadAffinity(vcpu_list), cp.cuda.Device(cuda_device_id):
            num_addrs = len(self.config['data_ip_addrs'][i])

            rb_nbytes = self.config['rb_host_memory_per_server_bytes']
            dd_nbytes = self.config['dd_host_memory_per_server_bytes']
            gpu_nbytes = self.config['gpu_memory_per_server_bytes']

            # Ring-buffer host memory: async, so the ctor returns immediately
            # and workers handle zero (+ register, in hugepage case).
            rb_bump = BumpAllocator(
                self.rb_host_aflags, rb_nbytes,
                is_async=True,
                nthreads=compute_async_bump_nthreads(vcpu_list, rb_nbytes),
                cuda_device=cuda_device_id)
            # SlabAllocator: async-aware. Returns immediately; defers the
            # allocate_bytes() call to first get_slab() (which happens after
            # server.start()).
            slab_allocator = SlabAllocator(rb_bump, rb_nbytes)
            # AssembledFrameAllocator: spawns its own worker thread; that
            # worker calls into slab_allocator and will block on the
            # BumpAllocator's init if it tries to call get_slab() before
            # phase 2 (our wait loop) completes. In practice the AFA worker
            # is waiting for a request to land, so it's fine.
            allocator = AssembledFrameAllocator(
                slab_allocator,
                num_consumers=num_addrs,
                time_samples_per_chunk=self.dedisp_config.time_samples_per_chunk,
            )

            # Dedispersion host memory (passed to FrbServer; used by the
            # processing thread's GpuDedisperser::allocate call). Async.
            host_alloc = BumpAllocator(
                self.dd_host_aflags, dd_nbytes,
                is_async=True,
                nthreads=compute_async_bump_nthreads(vcpu_list, dd_nbytes),
                cuda_device=cuda_device_id)

            # GPU memory (cudaMalloc + cudaMemset). Async; nthreads ignored
            # for case 3 but we pass it for uniformity.
            gpu_alloc = BumpAllocator(
                self.gpu_aflags, gpu_nbytes,
                is_async=True,
                nthreads=compute_async_bump_nthreads(vcpu_list, gpu_nbytes),
                cuda_device=cuda_device_id)

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
            grouper_ip_addr = '' if self.no_grouper else self.config['grouper_ip_addrs'][i]
            server = FrbServer(self.dedisp_config, receivers, file_writer,
                               self.config['rpc_ip_addrs'][i],
                               self.config['ringbuf_nchunks'],
                               min_data_mtu=self.config['min_data_mtu'],
                               host_allocator=host_alloc,
                               gpu_allocator=gpu_alloc,
                               cuda_device_id=cuda_device_id,
                               processing_delay_sec=self.processing_delay_sec,
                               grouper_ip_addr=grouper_ip_addr)
            # server.start() is NOT called here. We defer all server.start()
            # calls to _build_all_servers's phase 3 so that the async
            # BumpAllocators across all servers can initialize concurrently
            # (rather than getting blocked on the first server's startup).

        self.servers.append(server)
        # Stash rb_bump for the wait-until-initialized loop in phase 2.
        # (FrbServer doesn't store rb_bump itself; the SlabAllocator above
        # holds a shared_ptr to it which would keep it alive transitively,
        # but we want explicit access for the wait loop.)
        if not hasattr(self, '_rb_bumps'):
            self._rb_bumps = []
        self._rb_bumps.append(rb_bump)
        print(f"  Server {i} built (async init in progress).")

    def _print_server_details(self, i, vcpu_list, cuda_device_id):
        cpus = self.config['server_cpus']
        print(f"\nServer {i}: CPU {cpus[i]}, vcpu_list = {vcpu_list}")
        print(f"  cuda_device_id = {cuda_device_id}")
        print(f"  data_ip_addrs = {self.config['data_ip_addrs'][i]}")
        print(f"  rpc_ip_addr = {self.config['rpc_ip_addrs'][i]}")
        grouper = '' if self.no_grouper else self.config['grouper_ip_addrs'][i]
        print(f"  grouper_ip_addr = {grouper!r}" + ("  (disabled via --no-grouper)" if self.no_grouper else ""))
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
        # Poll for two things: Ctrl-C (KeyboardInterrupt, raised out of
        # time.sleep), and any FrbServer transitioning to is_stopped=True
        # via an internal error (in which case the C++ side has already
        # printed the exception message to stderr).
        while True:
            for i, server in enumerate(self.servers):
                if server.is_stopped:
                    print(f"FrbServer {i} stopped (internal error -- see stderr above). Exiting.")
                    return
            time.sleep(1)

    def _stop_all_servers(self):
        for server in self.servers:
            server.stop()
        print("All servers stopped.")


def run_server(server_config_filename, dedispersion_config_filename,
               processing_delay_sec=0.0, no_grouper=False):
    """Main entry point for 'pirate_frb run_server'.

    processing_delay_sec (default 0): artificial per-frame delay (seconds)
    injected by the FrbServer processing thread. Used to simulate slow
    GPU work for testing the FakeXEngine pacing path.

    no_grouper (default False): if True, disable the FrbGrouper RPC even when
    'grouper_ip_addrs' is set in the config (GpuDedisperser runs with
    num_consumers=0, receivers start immediately).
    """
    helper = RunServerHelper(server_config_filename, dedispersion_config_filename,
                             processing_delay_sec=processing_delay_sec,
                             no_grouper=no_grouper)
    helper.run()

