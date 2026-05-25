"""Implementation of 'pirate_frb run_fake_xengine' subcommand."""

import time
import threading

from .Hardware import Hardware
from .utils import ThreadAffinity, extract_ip, check_mtu
from .core import FakeXEngine, XEngineMetadata
from .rpc import FrbClient


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


def _fake_xengine_controller_wrapper(rpc_addr, fxe, all_fxes, exc_list, exc_lock):
    """Wraps _fake_xengine_controller_main. On exit (normal or exceptional),
    cascade-stops every FakeXEngine in all_fxes so sibling controllers exit
    promptly via "called on stopped instance". Captures any exception (paired
    with its rpc_addr) under exc_lock for the main thread to surface.
    """
    try:
        _fake_xengine_controller_main(fxe)
    except BaseException as e:
        with exc_lock:
            exc_list.append((rpc_addr, e))
    finally:
        for other in all_fxes:
            try:
                other.stop()
            except Exception:
                pass


def run_fake_xengine(rpc_addrs, nworkers=128):
    """Main entry point for 'pirate_frb run_fake_xengine'.

    For each rpc_addr in rpc_addrs, sends a GetConfig RPC, synthesizes an
    XEngineMetadata, runs the MTU + NIC-CPU consistency checks, and spawns
    one FakeXEngine + controller thread pinned to that receiver's data-NIC
    CPU. Blocks until Ctrl-C or every controller exits.

    If any controller errors out, the cascade in
    _fake_xengine_controller_wrapper stops every FakeXEngine; the main
    thread then surfaces the first non-cascade exception.

    Args:
        rpc_addrs: non-empty list[str] of "ip:port" strings (one per
            receiver). Duplicates are rejected, and a bare string is
            rejected (use [addr], not addr).
        nworkers: worker threads per FakeXEngine (not aggregate across
            receivers).
    """
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

    hw = Hardware()
    fake_xengines = []   # parallel to rpc_addrs (index-aligned)
    fxe_vcpus = []       # vcpu_list per fxe
    controllers = []
    exc_list = []        # list[(rpc_addr, BaseException)]
    exc_lock = threading.Lock()

    try:
        # ---- Phase 1: per-receiver GetConfig + checks + FakeXEngine ----
        # Doing all of these before spawning any controllers means a
        # misconfigured/unreachable receiver fails fast, with no
        # half-built state running in the background.
        for rpc_addr in rpc_addrs:
            print(f"\n[{rpc_addr}] Connecting ...")
            with FrbClient(rpc_addr) as c:
                cfg = c.get_config()

            ip_addrs = list(cfg.data_ip_addrs)
            if not ip_addrs:
                raise RuntimeError(f"[{rpc_addr}] reported empty data_ip_addrs")
            time_samples_per_chunk = cfg.time_samples_per_chunk
            min_data_mtu = cfg.min_data_mtu

            # Synthesize XEngineMetadata from the receiver's prefilled config.
            # beam_ids = {0, 1, ..., nbeams-1} -- the receiver records
            # whatever we send, so no cross-X-engine consensus to satisfy.
            nbeams = cfg.fake_nbeams
            beam_ids = list(range(nbeams))
            xmd = XEngineMetadata.make_test_instance(
                list(cfg.fake_zone_nfreq),
                list(cfg.fake_zone_freq_edges),
                beam_ids,
                cfg.fake_time_sample_ms,
            )

            actual_time_sample_ms = (xmd.dt_ns_per_seq * xmd.seq_per_frb_time_sample) / 1.0e6
            print(f"[{rpc_addr}]   data_ip_addrs = {ip_addrs}")
            print(f"[{rpc_addr}]   time_samples_per_chunk = {time_samples_per_chunk}")
            print(f"[{rpc_addr}]   min_data_mtu = {min_data_mtu}")
            print(f"[{rpc_addr}]   nbeams = {nbeams}, total_nfreq = {xmd.get_total_nfreq()}")
            print(f"[{rpc_addr}]   time_sample_ms = {actual_time_sample_ms}")
            print(f"[{rpc_addr}]   dt_ns_per_seq = {xmd.dt_ns_per_seq}")
            print(f"[{rpc_addr}]   seq_per_frb_time_sample = {xmd.seq_per_frb_time_sample}")

            # Hardware: MTU + NIC-on-one-CPU consistency.
            vcpu_list = None
            first_cpu = None
            for addr in ip_addrs:
                ip = extract_ip(addr)
                vl = hw.vcpu_list_from_ip_addr(ip, is_dst_addr=True)
                check_mtu(hw, f"[{rpc_addr}] FakeXEngine -> {addr}", ip,
                          min_data_mtu, 'min_data_mtu', is_dst_addr=True)
                cpu = hw.cpu_from_vcpu_list(vl)
                if vcpu_list is None:
                    vcpu_list = vl
                    first_cpu = cpu
                elif cpu != first_cpu:
                    raise RuntimeError(
                        f"[{rpc_addr}] destination IPs {ip_addrs} route through "
                        f"NICs on different CPUs (need all on one CPU)"
                    )

            with ThreadAffinity(vcpu_list):
                fxe = FakeXEngine(
                    xmd, ip_addrs, nworkers,
                    time_samples_per_chunk=time_samples_per_chunk)
            fake_xengines.append(fxe)
            fxe_vcpus.append(vcpu_list)
            print(f"[{rpc_addr}] FakeXEngine started ({nworkers} workers).")

        # ---- Phase 2: spawn all controllers ----
        # Spawning all controllers after Phase 1 completes means the
        # cascade in _fake_xengine_controller_wrapper sees the complete
        # fake_xengines list the moment the first controller runs.
        for rpc_addr, fxe, vcpu_list in zip(rpc_addrs, fake_xengines, fxe_vcpus):
            with ThreadAffinity(vcpu_list):
                t = threading.Thread(
                    target=_fake_xengine_controller_wrapper,
                    args=(rpc_addr, fxe, fake_xengines, exc_list, exc_lock),
                    daemon=True,
                )
                t.start()
            controllers.append(t)

        print(f"\nAll {len(rpc_addrs)} FakeXEngine(s) running. Press Ctrl-C to stop.")

        try:
            while any(t.is_alive() for t in controllers):
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopping...")

        for fxe in fake_xengines:
            try:
                fxe.stop()
            except Exception:
                pass
        for t in controllers:
            t.join(timeout=5.0)

        with exc_lock:
            # A "called on stopped instance" RuntimeError is the controller's
            # natural teardown artefact (or cascade artefact across receivers
            # when one of them errors); re-raise only "real" exceptions.
            real = [(addr, e) for (addr, e) in exc_list
                    if "called on stopped instance" not in str(e)]
            if real:
                raise real[0][1]

    finally:
        # Defensive cleanup -- harmless if the main path already ran it.
        for fxe in fake_xengines:
            try:
                fxe.stop()
            except Exception:
                pass
        print("All FakeXEngine(s) stopped.")
