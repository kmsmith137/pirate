"""Implementation of 'pirate_frb rpc_status' subcommand."""

import sys
import time
import textwrap
import threading

import grpc

from .rpc import FrbSearchClient


class _ServerMonitor:
    """Owns the daemon threads for a single FrbServer connection: one polls
    get_status() once per second and prints a summary line; one waits for the
    X-engine metadata and prints it once it arrives; one prints filenames as
    the server reports them. An error in any loop sets the shared stop_event,
    which tears down every connection.
    """

    def __init__(self, addr, client, stop_event):
        self.addr = addr
        self.client = client
        self.stop_event = stop_event

    def status_loop(self):
        """Poll get_status() once per second and print a summary line."""
        try:
            while not self.stop_event.is_set():
                status = self.client.get_status()
                print(f"[{self.addr}] connections={status.num_connections}, "
                      f"rb=[{status.rb_start},{status.rb_reaped},{status.rb_processed},{status.rb_streamed},{status.rb_assembled},{status.rb_end}], "
                      f"free={status.num_free_frames}")

                self._sleep_one_second()
        except Exception as e:
            print(f"[{self.addr}] ERROR: {e}", file=sys.stderr)
            self.stop_event.set()

    def metadata_loop(self):
        """Poll for the server's X-engine metadata once per second; print it
        once it becomes available, then stop. Runs in its own thread so the
        wait does not stall the status polling above, and stays responsive to
        stop_event via _sleep_one_second()."""
        try:
            while not self.stop_event.is_set():
                xmd_yaml = self.client._try_xengine_metadata()
                if xmd_yaml is not None:
                    print()
                    print(f"[{self.addr}] xengine_metadata:")
                    print(textwrap.indent(xmd_yaml.rstrip(), "  "))
                    print()
                    return

                self._sleep_one_second()
        except Exception as e:
            print(f"[{self.addr}] ERROR: {e}", file=sys.stderr)
            self.stop_event.set()

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
            # sentinel; iteration yields (filename, error_message, stream_name)
            # triples. subscribe_streams=True, so files written by streams
            # (nonempty stream_name) are reported here too, alongside the usual
            # WriteFiles-triggered files (stream_name == "").
            with self.client.subscribe_files(subscribe_streams=True) as sub:
                for filename, error_message, stream_name in sub:
                    if self.stop_event.is_set():
                        return
                    tag = f" (stream {stream_name})" if stream_name else ""
                    if error_message:
                        print(f"[{self.addr}] {filename} failed: {error_message}{tag}")
                    else:
                        print(f"[{self.addr}] {filename} received{tag}")
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
    print(f"  toplevel_tree_rank = {cfg.toplevel_tree_rank}")
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
    three daemon threads: one polls get_status() once per second (printing the
    connection count, ring-buffer counters, and free-frame count), one waits
    for the X-engine metadata and prints it once it arrives, and one prints
    filenames as the server reports them over subscribe_files().

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
        _print_config(addr, client.config)
    print()

    print("Running get_status (1/sec) and subscribe_files. Press Ctrl-C to stop.")
    print()

    stop_event = threading.Event()
    threads = []
    for addr, client in clients:
        monitor = _ServerMonitor(addr, client, stop_event)
        for loop_fn in (monitor.status_loop, monitor.subscribe_loop, monitor.metadata_loop):
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
