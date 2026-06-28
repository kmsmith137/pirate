import grpc
import numpy as np
import yaml

from .grpc import frb_sifter_pb2
from .grpc import frb_sifter_pb2_grpc


class FrbSifterClient:
    """Python client for the FrbSifter gRPC service (grpc/frb_sifter.proto).

    The "sifter" is a central process that receives FRB events (and coarse-grained
    per-beam SNRs) from the groupers on all search nodes. This client lets other
    code open a connection to a sifter and send it messages. It wraps the two RPCs
    currently used by the prototype grouper (pirate_frb/run_chord_grouper.py):

      - check_configuration() -- the initial ConfigMessage (config YAML strings).
      - send_events()         -- a per-time-period FrbEventsMessage (FRB events
                                 and/or coarse-grained per-beam SNRs).

    Both methods return None and raise a verbose RuntimeError on failure (a gRPC
    transport error, or a not-ok reply from the sifter).

    Usage::

        with FrbSifterClient("localhost:7100") as sifter:
            sifter.check_configuration(pirate_yaml, xengine_yaml,
                                       dedispersion_plan_yaml, grouper_yaml)
            sifter.send_events(has_injections, beam_set_id, chunk_fpga_count,
                               event_beam_ids, event_fpga_timestamps, event_dms,
                               event_dm_errors, event_snrs, event_rfi_probs,
                               coarsegrain_start_fpga_count,
                               coarsegrain_end_fpga_count, coarsegrain_snr)
    """

    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = frb_sifter_pb2_grpc.FrbSifterStub(self.channel)

    def check_configuration(self, pirate_yaml, xengine_yaml,
                            dedispersion_plan_yaml, grouper_yaml):
        """Send the initial ConfigMessage (CheckConfiguration RPC).

        Each of the four yaml arguments may be either a str (sent unmodified) or
        any yaml-serializable object (yaml.dump()'d to a string before sending).

        Returns None; raises a verbose RuntimeError on failure (yaml-serialization
        error, gRPC transport error, or a not-ok reply).
        """
        request = frb_sifter_pb2.ConfigMessage(
            pirate_yaml = self._to_yaml("pirate_yaml", pirate_yaml),
            xengine_yaml = self._to_yaml("xengine_yaml", xengine_yaml),
            dedispersion_plan_yaml = self._to_yaml("dedispersion_plan_yaml", dedispersion_plan_yaml),
            grouper_yaml = self._to_yaml("grouper_yaml", grouper_yaml),
        )
        try:
            reply = self.stub.CheckConfiguration(request)
        except grpc.RpcError as e:
            raise RuntimeError(f"FrbSifterClient.check_configuration: RPC to sifter at "
                               f"{self.server_address!r} failed: {e}") from e
        if not reply.ok:
            raise RuntimeError(f"FrbSifterClient.check_configuration: sifter at "
                               f"{self.server_address!r} returned ok=False")

    def send_events(self, has_injections, beam_set_id, chunk_fpga_count,
                    event_beam_ids, event_fpga_timestamps, event_dms,
                    event_dm_errors, event_snrs, event_rfi_probs,
                    coarsegrain_start_fpga_count, coarsegrain_end_fpga_count,
                    coarsegrain_snr):
        """Send an FrbEventsMessage (FrbEvents RPC). Args are in proto field order.

        The per-event fields are passed as six parallel 1-d arrays -- one FrbEvent
        is built per element, so all six must have equal length. These six and
        coarsegrain_snr may each be a numpy or cupy 1-d array (or any 1-d iterable):
        a cupy array is copied to the host in a single shot via .get() (cupy raises
        on implicit np.asarray()), then cast to the proto field's dtype.

        Returns None; raises a verbose RuntimeError on failure (malformed input,
        gRPC transport error, or a not-ok reply).
        """
        beam_ids   = self._to_1d("event_beam_ids", event_beam_ids, np.int32)
        timestamps = self._to_1d("event_fpga_timestamps", event_fpga_timestamps, np.int64)
        dms        = self._to_1d("event_dms", event_dms, np.float32)
        dm_errors  = self._to_1d("event_dm_errors", event_dm_errors, np.float32)
        snrs       = self._to_1d("event_snrs", event_snrs, np.float32)
        rfi_probs  = self._to_1d("event_rfi_probs", event_rfi_probs, np.float32)

        event_arrays = [beam_ids, timestamps, dms, dm_errors, snrs, rfi_probs]
        if len({len(a) for a in event_arrays}) != 1:
            raise RuntimeError(f"FrbSifterClient.send_events: the six event_* arrays must "
                               f"have equal length, got {[len(a) for a in event_arrays]}")

        cg_snr = self._to_1d("coarsegrain_snr", coarsegrain_snr, np.float32)

        # Build one FrbEvent per element. .tolist() yields native Python scalars.
        events = [
            frb_sifter_pb2.FrbEvent(beam_id=b, fpga_timestamp=t, dm=dm,
                                    dm_error=de, snr=s, rfi_prob=r)
            for b, t, dm, de, s, r in zip(beam_ids.tolist(), timestamps.tolist(),
                                          dms.tolist(), dm_errors.tolist(),
                                          snrs.tolist(), rfi_probs.tolist())
        ]

        request = frb_sifter_pb2.FrbEventsMessage(
            has_injections = has_injections,
            beam_set_id = beam_set_id,
            chunk_fpga_count = chunk_fpga_count,
            events = events,
            coarsegrain_start_fpga_count = coarsegrain_start_fpga_count,
            coarsegrain_end_fpga_count = coarsegrain_end_fpga_count,
            coarsegrain_snr = cg_snr,
        )
        try:
            reply = self.stub.FrbEvents(request)
        except grpc.RpcError as e:
            raise RuntimeError(f"FrbSifterClient.send_events: RPC to sifter at "
                               f"{self.server_address!r} failed: {e}") from e
        if not reply.ok:
            raise RuntimeError(f"FrbSifterClient.send_events: sifter at {self.server_address!r} "
                               f"rejected the message: {reply.message!r}")

    @staticmethod
    def _to_yaml(name, value):
        """Return 'value' unchanged if it is a str, else yaml.dump() it to a string."""
        if isinstance(value, str):
            return value
        try:
            return yaml.dump(value)
        except Exception as e:
            raise RuntimeError(f"FrbSifterClient: argument {name!r} (type "
                               f"{type(value).__name__}) is neither a str nor "
                               f"yaml-serializable: {e}") from e

    @staticmethod
    def _to_1d(name, x, np_dtype):
        """Convert a 1-d numpy/cupy array (or any 1-d iterable) to a contiguous 1-d
        numpy array of dtype 'np_dtype'. A cupy array is first copied to the host in
        one shot via .get() (np.asarray() can't be applied directly: cupy raises on
        implicit conversion)."""
        arr = x
        if type(arr).__module__.split('.', 1)[0] == 'cupy':
            arr = arr.get()   # single bulk device -> host copy
        try:
            arr = np.asarray(arr, dtype=np_dtype)
        except Exception as e:
            raise RuntimeError(f"FrbSifterClient: could not convert argument {name!r} to a "
                               f"1-d numpy {np.dtype(np_dtype).name} array: {e}") from e
        if arr.ndim != 1:
            raise RuntimeError(f"FrbSifterClient: argument {name!r} must be 1-d, "
                               f"got shape {arr.shape}")
        return arr

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"FrbSifterClient({self.server_address!r})"
