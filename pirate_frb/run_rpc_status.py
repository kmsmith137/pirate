"""Implementation of 'pirate_frb rpc_status' subcommand."""

import sys
import math
import time
import textwrap
import threading

import grpc
import yaml

from .rpc import FrbClient


class _ThroughputTracker:
    """Per-connection helper that turns successive get_status() samples into
    the bandwidth and 'real-time beams' figures shown on each status line.

    'rt_beams' is the number of beams the server processes in real time, based
    on measured throughput: if it processes N AssembledFrames (= delta
    rb_processed) in delta_time seconds, the instantaneous value is
    N * chunk_dur / delta_time, where chunk_dur is the seconds of data per
    processed chunk. We smooth this with an EMA (rt_tau-second scale), and
    don't start sampling until rb_processed is nonzero: rb_processed can jump
    discontinuously at startup (if initial_time_chunk != 0), so we wait for
    steady-state before sampling the rate. rt_beams stays omitted from the
    status line until xengine_metadata arrives and set_chunk_dur() is called.
    """

    def __init__(self, rt_tau=10.0):
        self.rt_tau = rt_tau          # EMA smoothing scale (seconds)
        self.chunk_dur = None         # seconds of data per processed chunk (set once metadata arrives)
        self.prev_time = None
        self.prev_bytes = None
        self.prev_processed = None    # previous rb_processed (tracked only once nonzero == steady-state)
        self.rt_ema = None            # EMA of "real-time beams" throughput

    def set_chunk_dur(self, chunk_dur):
        """Supply seconds-of-data-per-chunk, enabling rt_beams once the
        server's xengine_metadata is known."""
        self.chunk_dur = chunk_dur

    def update(self, status, now):
        """Fold one (status, monotonic-timestamp) sample into the tracker and
        return the formatted status-line suffix, e.g.
        ", bw=1.23 Gbps, rt_beams=4.5". Each piece is omitted until it can be
        computed (bandwidth needs a previous sample; rt_beams needs metadata
        and steady-state)."""
        bw_str = self._bandwidth_str(status, now)
        rt_str = self._rt_beams_str(status, now)
        self.prev_time = now
        self.prev_bytes = status.num_bytes
        return bw_str + rt_str

    def _bandwidth_str(self, status, now):
        if self.prev_time is None or (now - self.prev_time) <= 0:
            return ""
        delta_bytes = status.num_bytes - self.prev_bytes
        delta_time = now - self.prev_time
        gbps = (delta_bytes * 8) / (delta_time * 1e9)
        return f", bw={gbps:.2f} Gbps"

    def _rt_beams_str(self, status, now):
        if (self.chunk_dur is not None) and (status.rb_processed > 0):
            if (self.prev_processed is not None) and (now - self.prev_time) > 0:
                delta_time = now - self.prev_time
                inst = (status.rb_processed - self.prev_processed) * self.chunk_dur / delta_time
                alpha = 1.0 - math.exp(-delta_time / self.rt_tau)
                self.rt_ema = inst if (self.rt_ema is None) else (alpha * inst + (1.0 - alpha) * self.rt_ema)
            self.prev_processed = status.rb_processed
        else:
            self.prev_processed = None
        if self.rt_ema is None:
            return ""
        return f", rt_beams={self.rt_ema:.1f}"


class _ServerMonitor:
    """Owns the two daemon threads for a single FrbServer connection: one
    polls get_status() once per second and prints a summary line; the other
    prints filenames as the server reports them. An error in either loop sets
    the shared stop_event, which tears down every connection.
    """

    def __init__(self, addr, client, stop_event):
        self.addr = addr
        self.client = client
        self.stop_event = stop_event

    def status_loop(self):
        """Poll get_status() once per second and print a summary. Also tries
        get_xengine_metadata() once per second until it returns a non-empty
        YAML string, then prints it (once) and stops trying."""
        try:
            cfg = self.client.get_config()
            tracker = _ThroughputTracker()
            tsamp_per_chunk = cfg.time_samples_per_chunk
            metadata_printed = False

            while not self.stop_event.is_set():
                if not metadata_printed:
                    metadata_printed = self._try_print_metadata(tracker, tsamp_per_chunk)

                status = self.client.get_status()
                suffix = tracker.update(status, time.monotonic())
                print(f"[{self.addr}] connections={status.num_connections}{suffix}, "
                      f"rb=[{status.rb_start},{status.rb_reaped},{status.rb_processed},{status.rb_assembled},{status.rb_end}], "
                      f"free={status.num_free_frames}")

                self._sleep_one_second()
        except Exception as e:
            print(f"[{self.addr}] ERROR: {e}", file=sys.stderr)
            self.stop_event.set()

    def _try_print_metadata(self, tracker, tsamp_per_chunk):
        """Try once to fetch and print xengine_metadata. Returns True once it
        has been printed (caller then stops trying), False while it is still
        unavailable. On success, derives chunk_dur and hands it to the tracker.
        """
        xmd_yaml = self.client.get_xengine_metadata(verbose=False)
        if not xmd_yaml:
            return False

        print()
        print(f"[{self.addr}] xengine_metadata:")
        print(textwrap.indent(xmd_yaml.rstrip(), "  "))
        print()

        # Seconds of data per processed chunk = (time samples per chunk) x
        # (seconds per FRB time sample). The tracker uses this to convert the
        # rb_processed rate into "real-time beams".
        xmd = yaml.safe_load(xmd_yaml)
        chunk_dur = (1.0e-9 * tsamp_per_chunk
                     * xmd['seq_per_frb_time_sample'] * xmd['dt_ns_per_seq'])
        tracker.set_chunk_dur(chunk_dur)
        return True

    def _sleep_one_second(self):
        """Sleep ~1 second, waking every 0.1s to check stop_event so the loop
        exits promptly on Ctrl-C or a sibling thread's error."""
        for _ in range(10):
            if self.stop_event.is_set():
                return
            time.sleep(0.1)

    def subscribe_loop(self):
        """Subscribe to filenames and print them as they arrive."""
        try:
            # subscribe_files() returns a FileSubscriber whose constructor has
            # already opened the stream and consumed the server's ready
            # sentinel; iteration yields (filename, error_message) pairs.
            with self.client.subscribe_files() as sub:
                for filename, error_message in sub:
                    if self.stop_event.is_set():
                        return
                    if error_message:
                        print(f"[{self.addr}] {filename} failed: {error_message}")
                    else:
                        print(f"[{self.addr}] {filename} received")
        except grpc.RpcError as e:
            # CANCELLED here is from something OTHER than our own close()
            # (which the FileSubscriber converts to clean StopIteration). In
            # practice: server graceful shutdown. Silence it; surface anything
            # else.
            if e.code() != grpc.StatusCode.CANCELLED:
                print(f"[{self.addr}] subscribe_files ERROR: {e}", file=sys.stderr)
                self.stop_event.set()
        except Exception as e:
            print(f"[{self.addr}] subscribe_files ERROR: {e}", file=sys.stderr)
            self.stop_event.set()


def _print_config(addr, cfg):
    """Print the one-shot GetConfig dump for a single server."""
    print(f"[{addr}] config:")
    print(f"  rpc_ip_addr = {cfg.rpc_ip_addr}")
    print(f"  data_ip_addrs = {list(cfg.data_ip_addrs)}")
    print(f"  time_samples_per_chunk = {cfg.time_samples_per_chunk}")
    print(f"  ringbuf_nchunks = {cfg.ringbuf_nchunks}")
    print(f"  ssd_dir = {cfg.ssd_dir}")
    print(f"  nfs_dir = {cfg.nfs_dir}")
    print(f"  ssd_threads = {cfg.ssd_threads}")
    print(f"  nfs_threads = {cfg.nfs_threads}")
    print(f"  tree_rank = {cfg.tree_rank}")
    print(f"  beams_per_batch = {cfg.beams_per_batch}")
    print(f"  min_data_mtu = {cfg.min_data_mtu}")
    print(f"  fake_zone_nfreq = {list(cfg.fake_zone_nfreq)}")
    print(f"  fake_zone_freq_edges = {list(cfg.fake_zone_freq_edges)}")
    print(f"  fake_time_sample_ms = {cfg.fake_time_sample_ms}")
    print(f"  fake_nbeams = {cfg.fake_nbeams}")


def run_rpc_status(ip_addrs):
    """Connect to one or more FrbServers and stream status + filenames.

    Prints a one-shot config dump for each server, then -- per server -- runs
    two daemon threads: one polls get_status() once per second (printing
    connection count, instantaneous bandwidth, an EMA of "real-time beams"
    throughput, and the ringbuffer counters), the other prints filenames as
    the server reports them over subscribe_files().

    Blocks until Ctrl-C or until any thread hits an error (the first error
    sets a shared stop_event that tears down every connection). Exits the
    process with status 1 if any thread errored.

    Args:
        ip_addrs: non-empty list[str] of server "ip:port" addresses
            (e.g. ["127.0.0.1:6000"]).
    """
    # Strings are iterable, so a caller passing a bare string would silently
    # iterate it character-by-character (FrbClient('1'), FrbClient('2'), ...).
    # Short-circuit with a clear error (mirrors run_fake_xengine).
    if isinstance(ip_addrs, str):
        raise RuntimeError(
            f"run_rpc_status: ip_addrs must be a list of strings, "
            f"not a single string ({ip_addrs!r})"
        )
    if not ip_addrs:
        raise RuntimeError("run_rpc_status: ip_addrs is empty")

    clients = [(addr, FrbClient(addr)) for addr in ip_addrs]

    print(f"RPC client(s) connected to {', '.join(ip_addrs)}")
    print()

    # One-shot startup dump: print each server's configuration (GetConfig).
    for addr, client in clients:
        _print_config(addr, client.get_config())
    print()

    print("Running get_status (1/sec) and subscribe_files. Press Ctrl-C to stop.")
    print()

    stop_event = threading.Event()
    threads = []
    for addr, client in clients:
        monitor = _ServerMonitor(addr, client, stop_event)
        for loop_fn in (monitor.status_loop, monitor.subscribe_loop):
            t = threading.Thread(target=loop_fn, daemon=True)
            t.start()
            threads.append(t)

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()

    for t in threads:
        t.join(timeout=1.0)
    for _, client in clients:
        client.close()
    print("RPC client(s) stopped.")

    if stop_event.is_set():
        sys.exit(1)
