"""Toy FrbSifter gRPC server: print a one-line summary of each received message.

Pure-Python (no C++) implementation of the FrbSifter service from
frb_sifter.proto. The real sifter (chord-frb-sifter) aggregates FRB events from
the groupers on all search nodes; this toy version simply receives each RPC and
prints a one-line summary of the message, then returns a trivial ok=True reply.
Useful for smoke-testing a grouper's sifter-client path without running the real
sifter.
"""

from concurrent import futures

import grpc

from .rpc.grpc import frb_sifter_pb2
from .rpc.grpc import frb_sifter_pb2_grpc


def _peer_ip(context):
    """Return the client IP from grpc ServicerContext.peer().

    peer() looks like 'ipv4:127.0.0.1:54321' or 'ipv6:[::1]:54321'; we strip the
    'ipvN:' scheme and the trailing ':port', returning just the address. Any
    other form (e.g. a 'unix:' socket) is returned unchanged.
    """
    peer = context.peer()
    if peer.startswith("ipv4:"):
        return peer[len("ipv4:"):].rsplit(":", 1)[0]
    if peer.startswith("ipv6:"):
        rest = peer[len("ipv6:"):]   # '[addr]:port'
        if rest.startswith("[") and "]" in rest:
            return rest[1:rest.index("]")]
        return rest.rsplit(":", 1)[0]
    return peer


class ToyFrbSifterServicer(frb_sifter_pb2_grpc.FrbSifterServicer):
    """Implements FrbSifter; prints one summary line per received RPC."""

    def CheckConfiguration(self, request, context):
        # Reject any client whose wire-protocol version does not match ours.
        # context.abort() raises to terminate the RPC, so the client gets a
        # clean FAILED_PRECONDITION error naming the mismatch (rather than a
        # silent ok=True). proto3 defaults an unset protocol_version to 0, so an
        # old client that never set the field is rejected here too.
        if request.protocol_version != frb_sifter_pb2.PROTOCOL_VERSION_CURRENT:
            msg = (f"protocol version mismatch: client sent protocol_version="
                   f"{request.protocol_version}, but this sifter requires "
                   f"{frb_sifter_pb2.PROTOCOL_VERSION_CURRENT}")
            print(f"{_peer_ip(context)}  CheckConfiguration REJECTED ({msg})", flush=True)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, msg)

        # One-line summary: the size of each (possibly empty) config YAML field,
        # rather than dumping the multi-line YAML itself.
        sizes = ", ".join(
            f"{name}={len(val)}B"
            for name, val in (
                ("pirate", request.pirate_yaml),
                ("xengine", request.xengine_yaml),
                ("dedispersion_plan", request.dedispersion_plan_yaml),
                ("grouper", request.grouper_yaml),
            )
        )
        print(f"{_peer_ip(context)}  CheckConfiguration({sizes})", flush=True)
        return frb_sifter_pb2.ConfigReply(ok=True)

    def FrbEvents(self, request, context):
        # 'chunk_fpga_count' is the proto field; the grouper populates it with the
        # chunk's FPGA-seq start (FrbSifterEvents.chunk_fpga_start).
        n_events = len(request.events)
        line = (f"beamset={request.beam_set_id}, "
                f"fpga={request.chunk_fpga_count}, "
                f"nevents={n_events}")
        if n_events > 0:
            # Report the single highest-SNR event.
            ev = max(request.events, key=lambda e: e.snr)
            line += (f" (snr={ev.snr:.4g}, beam={ev.beam_id}, dm={ev.dm:.4g}, "
                     f"fpga={ev.fpga_timestamp}, rfiprob={ev.rfi_prob:.4g})")
        print(line, flush=True)
        return frb_sifter_pb2.FrbEventsReply(ok=True, message="")


def run_toy_sifter(addr, max_workers=4):
    """Run a toy FrbSifter gRPC server listening at 'addr' (e.g. '127.0.0.1:7100').

    Implements the FrbSifter service (CheckConfiguration, FrbEvents) in pure
    Python. For each received RPC, prints a one-line summary of the message, then
    returns a trivial ok=True reply. Blocks until Ctrl-C.

    'addr' is passed to grpc's add_insecure_port(): use 'ip:port' for a specific
    interface, or '[::]:port' / '0.0.0.0:port' to listen on all interfaces.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    frb_sifter_pb2_grpc.add_FrbSifterServicer_to_server(ToyFrbSifterServicer(), server)

    if server.add_insecure_port(addr) == 0:
        raise RuntimeError(f"run_toy_sifter: failed to bind {addr!r} "
                           "(already in use, or malformed 'ip:port'?)")

    server.start()
    print(f"run_toy_sifter: listening on {addr} (Ctrl-C to stop)", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nrun_toy_sifter: interrupted; shutting down", flush=True)
        server.stop(grace=1.0).wait()
