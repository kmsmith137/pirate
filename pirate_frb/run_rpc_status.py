"""Implementation of 'pirate_frb rpc_status' subcommand."""

import sys
import time
import textwrap
import threading

import grpc

from .rpc import FrbSearchClient


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
            metadata_printed = False

            while not self.stop_event.is_set():
                if not metadata_printed:
                    metadata_printed = self._try_print_metadata()

                status = self.client.get_status()
                print(f"[{self.addr}] connections={status.num_connections}, "
                      f"rb=[{status.rb_start},{status.rb_reaped},{status.rb_processed},{status.rb_assembled},{status.rb_end}], "
                      f"free={status.num_free_frames}")

                self._sleep_one_second()
        except Exception as e:
            print(f"[{self.addr}] ERROR: {e}", file=sys.stderr)
            self.stop_event.set()

    def _try_print_metadata(self):
        """Try once to fetch and print xengine_metadata. Returns True once it
        has been printed (caller then stops trying), False while it is still
        unavailable.
        """
        xmd_yaml = self.client.get_xengine_metadata(verbose=False)
        if not xmd_yaml:
            return False

        print()
        print(f"[{self.addr}] xengine_metadata:")
        print(textwrap.indent(xmd_yaml.rstrip(), "  "))
        print()
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
    print(f"  frequency_subband_counts = {list(cfg.frequency_subband_counts)}")
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
    # iterate it character-by-character (FrbSearchClient('1'), FrbSearchClient('2'), ...).
    # Short-circuit with a clear error (mirrors run_fake_xengine).
    if isinstance(ip_addrs, str):
        raise RuntimeError(
            f"run_rpc_status: ip_addrs must be a list of strings, "
            f"not a single string ({ip_addrs!r})"
        )
    if not ip_addrs:
        raise RuntimeError("run_rpc_status: ip_addrs is empty")

    clients = [(addr, FrbSearchClient(addr)) for addr in ip_addrs]

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
