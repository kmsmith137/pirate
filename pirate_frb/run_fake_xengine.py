"""Implementation of 'pirate_frb run_fake_xengine' subcommand."""

import time
import threading

from .Hardware import Hardware
from .utils import ThreadAffinity, extract_ip, check_mtu
from .core import FakeXEngine, XEngineMetadata
from .rpc import FrbClient


class RunFakeXEngineHelper:
    """Encapsulates state and logic for 'pirate_frb run_fake_xengine'.

    Constructed once per invocation; call .run() to drive the full
    lifecycle (Phase 1: per-receiver GetConfig + FakeXEngine; Phase 2:
    spawn controllers; wait for Ctrl-C; cleanup).
    """

    def __init__(self, rpc_addrs, nworkers=128):
        # Strings are iterable, so a caller who passes a bare string would
        # silently iterate character-by-character. Short-circuit with a clear
        # error.
        if isinstance(rpc_addrs, str):
            raise RuntimeError(
                f"run_fake_xengine: rpc_addrs must be a list of strings, "
                f"not a single string ({rpc_addrs!r})"
            )
        if not rpc_addrs:
            raise RuntimeError("run_fake_xengine: rpc_addrs is empty")
        if len(set(rpc_addrs)) != len(rpc_addrs):
            raise RuntimeError(
                f"run_fake_xengine: duplicate rpc_addrs in {rpc_addrs!r}"
            )

        self.rpc_addrs = rpc_addrs
        self.nworkers = nworkers
        self.hw = Hardware()
        self.fake_xengines = []   # parallel to rpc_addrs (Phase 1 fills this)
        self.fxe_vcpus = []       # parallel
        self.controllers = []     # parallel (Phase 2 fills this)
        self.exc_list = []        # list[(rpc_addr, BaseException)]
        self.exc_lock = threading.Lock()

    def run(self):
        """Top-level lifecycle: Phase 1 build, Phase 2 spawn, wait, cleanup.

        If any controller errors out, the cascade in _controller_wrapper
        stops every FakeXEngine; the main thread then surfaces the first
        non-cascade exception.
        """
        try:
            self._build_all()
            self._spawn_all_controllers()
            print(f"\nAll {len(self.rpc_addrs)} FakeXEngine(s) running. "
                  f"Press Ctrl-C to stop.")
            self._wait_for_controllers()
            self._stop_and_join_all()
            self._surface_real_exceptions()
        finally:
            # Defensive cleanup -- harmless if the main path already ran it.
            self._stop_all_fxes()
            print("All FakeXEngine(s) stopped.")

    def _build_all(self):
        # Doing all per-receiver builds before spawning any controllers means
        # a misconfigured/unreachable receiver fails fast, with no half-built
        # state running in the background.
        for rpc_addr in self.rpc_addrs:
            self._build_one(rpc_addr)

    def _build_one(self, rpc_addr):
        """Per-receiver Phase 1: GetConfig -> XMD -> NIC checks -> FakeXEngine."""
        print(f"\n[{rpc_addr}] Connecting ...")
        with FrbClient(rpc_addr) as c:
            cfg = c.get_config()
        if not list(cfg.data_ip_addrs):
            raise RuntimeError(f"[{rpc_addr}] reported empty data_ip_addrs")

        # Synthesize XEngineMetadata from the receiver's prefilled config.
        # beam_ids = {0, 1, ..., nbeams-1} -- the receiver records whatever
        # we send, so no cross-X-engine consensus to satisfy.
        beam_ids = list(range(cfg.fake_nbeams))
        xmd = XEngineMetadata.make_fiducial(
            list(cfg.fake_zone_nfreq),
            list(cfg.fake_zone_freq_edges),
            beam_ids,
            cfg.fake_time_sample_ms,
        )

        self._print_receiver_details(rpc_addr, cfg, xmd)
        vcpu_list = self._verify_nics_and_mtu(rpc_addr, cfg)

        with ThreadAffinity(vcpu_list):
            fxe = FakeXEngine(
                xmd, list(cfg.data_ip_addrs), self.nworkers,
                time_samples_per_chunk=cfg.time_samples_per_chunk)
        self.fake_xengines.append(fxe)
        self.fxe_vcpus.append(vcpu_list)
        print(f"[{rpc_addr}] FakeXEngine started ({self.nworkers} workers).")

    def _print_receiver_details(self, rpc_addr, cfg, xmd):
        ip_addrs = list(cfg.data_ip_addrs)
        actual_time_sample_ms = (xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample) / 1.0e6
        print(f"[{rpc_addr}]   data_ip_addrs = {ip_addrs}")
        print(f"[{rpc_addr}]   time_samples_per_chunk = {cfg.time_samples_per_chunk}")
        print(f"[{rpc_addr}]   min_data_mtu = {cfg.min_data_mtu}")
        print(f"[{rpc_addr}]   nbeams = {cfg.fake_nbeams}, total_nfreq = {xmd.get_total_nfreq()}")
        print(f"[{rpc_addr}]   time_sample_ms = {actual_time_sample_ms}")
        print(f"[{rpc_addr}]   dt_ns_per_seq = {xmd.dt_ns_per_seq}")
        print(f"[{rpc_addr}]   seq_per_frb_time_sample = {xmd.seq_per_frb_time_sample}")

    def _verify_nics_and_mtu(self, rpc_addr, cfg):
        """Check MTU on each data NIC and verify all NICs are on one CPU.
        Returns the (consistent) vcpu_list for use with ThreadAffinity.
        """
        ip_addrs = list(cfg.data_ip_addrs)
        vcpu_list = None
        first_cpu = None
        for addr in ip_addrs:
            ip = extract_ip(addr)
            vl = self.hw.vcpu_list_from_ip_addr(ip, is_dst_addr=True)
            check_mtu(self.hw, f"[{rpc_addr}] FakeXEngine -> {addr}", ip,
                      cfg.min_data_mtu, 'min_data_mtu', is_dst_addr=True)
            cpu = self.hw.cpu_from_vcpu_list(vl)
            if vcpu_list is None:
                vcpu_list = vl
                first_cpu = cpu
            elif cpu != first_cpu:
                raise RuntimeError(
                    f"[{rpc_addr}] destination IPs {ip_addrs} route through "
                    f"NICs on different CPUs (need all on one CPU)"
                )
        return vcpu_list

    def _spawn_all_controllers(self):
        # Spawning all controllers after Phase 1 completes means the cascade
        # in _controller_wrapper sees the complete self.fake_xengines list
        # the moment the first controller runs.
        for rpc_addr, fxe, vcpu_list in zip(
                self.rpc_addrs, self.fake_xengines, self.fxe_vcpus):
            with ThreadAffinity(vcpu_list):
                t = threading.Thread(
                    target=self._controller_wrapper,
                    args=(rpc_addr, fxe),
                    daemon=True,
                )
                t.start()
            self.controllers.append(t)

    def _controller_wrapper(self, rpc_addr, fxe):
        """Wraps _controller_main. On exit (normal or exceptional),
        cascade-stops every FakeXEngine in self.fake_xengines so sibling
        controllers exit promptly via "called on stopped instance". Captures
        any exception (paired with its rpc_addr) under self.exc_lock for the
        main thread to surface.
        """
        try:
            self._controller_main(fxe)
        except BaseException as e:
            with self.exc_lock:
                self.exc_list.append((rpc_addr, e))
        finally:
            for other in self.fake_xengines:
                try:
                    other.stop()
                except Exception:
                    pass

    def _controller_main(self, fxe):
        """Drive one FakeXEngine in 'send-junk-forever' mode.

        Reproduces the cross-worker "minichunk N waits for (N-2)"
        serialization that the C++ FakeXEngine used to enforce internally
        with its barrier. Runs until fxe.stop() is called (from anywhere),
        at which point the next wait_until_processed() / enqueue_send_junk()
        call raises RuntimeError and the function returns via exception.
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

    def _wait_for_controllers(self):
        try:
            while any(t.is_alive() for t in self.controllers):
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopping...")

    def _stop_and_join_all(self):
        for fxe in self.fake_xengines:
            try:
                fxe.stop()
            except Exception:
                pass
        for t in self.controllers:
            t.join(timeout=5.0)

    def _surface_real_exceptions(self):
        with self.exc_lock:
            # A "called on stopped instance" RuntimeError is the controller's
            # natural teardown artefact (or cascade artefact across receivers
            # when one of them errors); re-raise only "real" exceptions.
            real = [(addr, e) for (addr, e) in self.exc_list
                    if "called on stopped instance" not in str(e)]
            if real:
                raise real[0][1]

    def _stop_all_fxes(self):
        """Defensive idempotent stop (also called from `finally:`)."""
        for fxe in self.fake_xengines:
            try:
                fxe.stop()
            except Exception:
                pass


def run_fake_xengine(rpc_addrs, nworkers=128):
    """Main entry point for 'pirate_frb run_fake_xengine'.

    For each rpc_addr in rpc_addrs, sends a GetConfig RPC, synthesizes an
    XEngineMetadata, runs the MTU + NIC-CPU consistency checks, and spawns
    one FakeXEngine + controller thread pinned to that receiver's data-NIC
    CPU. Blocks until Ctrl-C or every controller exits.

    Args:
        rpc_addrs: non-empty list[str] of "ip:port" strings (one per
            receiver). Duplicates are rejected, and a bare string is
            rejected (use [addr], not addr).
        nworkers: worker threads per FakeXEngine (not aggregate across
            receivers).
    """
    helper = RunFakeXEngineHelper(rpc_addrs, nworkers)
    helper.run()
