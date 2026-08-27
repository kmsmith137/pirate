"""Implementation of 'pirate_frb rpc status' subcommand."""

import sys
import time
import textwrap
import threading

import grpc

from .rpc import FrbSearchClient
from .utils import atomic_print
from .pirate_pybind11 import constants


class _ServerMonitor:
    """Owns the daemon threads for a single FrbServer connection.

    One thread polls get_status() once per second and prints a summary line; one
    waits for the X-engine metadata and prints it once it arrives; one prints
    filenames as the server reports them. An error in any loop sets the shared
    stop_event, which tears down every connection.
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
                atomic_print(f"[{self.addr}] connections={status.num_connections}, "
                             f"rb=[{status.rb_start},{status.rb_reaped},{status.rb_processed},{status.rb_streamed},{status.rb_assembled},{status.rb_end}]")

                self._wait_between_polls()
        except Exception as e:
            atomic_print(f"[{self.addr}] ERROR: {e}", fd=2)
            self.stop_event.set()

    def metadata_loop(self):
        """Poll for the server's X-engine metadata, print it once, then stop.

        Polls once per second. Runs in its own thread so the
        wait does not stall the status polling above, and stays responsive to
        stop_event via _wait_between_polls()."""
        try:
            while not self.stop_event.is_set():
                xmd_yaml = self.client._try_xengine_metadata()
                if xmd_yaml is not None:
                    # One call, so the whole block stays contiguous.
                    atomic_print(f"\n[{self.addr}] xengine_metadata:\n"
                                 f"{textwrap.indent(xmd_yaml.rstrip(), '  ')}\n\n")
                    return

                self._wait_between_polls()
        except Exception as e:
            atomic_print(f"[{self.addr}] ERROR: {e}", fd=2)
            self.stop_event.set()

    def _wait_between_polls(self):
        """Wait one print/refresh interval (constants.default_print_cadence_sec).

        Wakes every constants.default_poll_cadence_ms ms to check stop_event so
        the loop exits promptly on Ctrl-C or a sibling thread's error."""
        step = constants.default_poll_cadence_ms / 1000
        interval = constants.default_print_cadence_sec
        slept = 0.0
        while slept < interval:
            if self.stop_event.is_set():
                return
            dt = min(step, interval - slept)
            time.sleep(dt)
            slept += dt

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
                        atomic_print(f"[{self.addr}] {filename} failed: {error_message}{tag}")
                    else:
                        atomic_print(f"[{self.addr}] {filename} received{tag}")
        except grpc.RpcError as e:
            # CANCELLED here is from something OTHER than our own close()
            # (which the FileSubscriber converts to clean StopIteration). In
            # practice: server graceful shutdown. Silence it; surface anything
            # else.
            if e.code() != grpc.StatusCode.CANCELLED:
                atomic_print(f"[{self.addr}] subscribe_files ERROR: {e}", fd=2)
                self.stop_event.set()
        except Exception as e:
            atomic_print(f"[{self.addr}] subscribe_files ERROR: {e}", fd=2)
            self.stop_event.set()


def _print_config(addr, cfg):
    """Print the one-shot GetConfig dump for a single server, as one block."""
    atomic_print(
        f"[{addr}] config:\n"
        f"  rpc_ip_addr = {cfg.rpc_ip_addr}\n"
        f"  data_ip_addrs = {list(cfg.data_ip_addrs)}\n"
        f"  time_samples_per_chunk = {cfg.time_samples_per_chunk}\n"
        f"  ringbuf_nchunks = {cfg.ringbuf_nchunks}\n"
        f"  ssd_dir = {cfg.ssd_dir}\n"
        f"  nfs_dir = {cfg.nfs_dir}\n"
        f"  ssd_threads = {cfg.ssd_threads}\n"
        f"  nfs_threads = {cfg.nfs_threads}\n"
        f"  toplevel_tree_rank = {cfg.toplevel_tree_rank}\n"
        f"  beams_per_batch = {cfg.beams_per_batch}\n"
        f"  frequency_subband_counts = {list(cfg.frequency_subband_counts)}\n"
        f"  min_data_mtu = {cfg.min_data_mtu}\n"
        f"  fake_zone_nfreq = {list(cfg.fake_zone_nfreq)}\n"
        f"  fake_zone_freq_edges = {list(cfg.fake_zone_freq_edges)}\n"
        f"  fake_time_sample_ms = {cfg.fake_time_sample_ms}\n"
        f"  fake_nbeams = {cfg.fake_nbeams}")


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

    atomic_print(f"RPC client(s) connected to {', '.join(ip_addrs)}\n\n")

    # One-shot startup dump: print each server's configuration (GetConfig).
    for addr, client in clients:
        _print_config(addr, client.config)
    atomic_print("\n")

    atomic_print("Running get_status (1/sec) and subscribe_files. Press Ctrl-C to stop.\n\n")

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
            time.sleep(constants.default_poll_cadence_ms / 1000)
    except KeyboardInterrupt:
        # Monitor threads may still be printing here, and join() below can time
        # out with one still alive -- so these go through atomic_print too.
        atomic_print("\nStopping...")
        stop_event.set()

    for t in threads:
        t.join(timeout=constants.default_shutdown_timeout_sec)
    for _, client in clients:
        client.close()
    atomic_print("RPC client(s) stopped.")

    if stop_event.is_set():
        sys.exit(1)
