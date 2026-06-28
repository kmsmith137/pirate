import grpc
import numpy as np
import yaml

from .grpc import frb_sifter_pb2
from .grpc import frb_sifter_pb2_grpc


class FrbSifterEvents:
    """A batch of FRB events to send to the sifter (FrbSifterClient.send_events).

    Holds the per-event fields of the FrbEvent message (grpc/frb_sifter.proto) as
    parallel numpy arrays (NOT cupy) -- one FrbEvent is sent per array element --
    plus the scalar ``chunk_fpga_count`` carried by the enclosing FrbEventsMessage.

    The per-event arrays (cast to the indicated dtype) are:

    - ``beam_ids`` (int32): X-engine beam id of each event
    - ``fpga_timestamps`` (int64): absolute FPGA-counter timestamp of each event
    - ``dms`` (float32): dispersion measure
    - ``dm_errors`` (float32): dispersion-measure uncertainty
    - ``snrs`` (float32): signal-to-noise ratio
    - ``rfi_probs`` (float32): RFI probability

    The from-scratch constructor takes these six arrays (each a numpy array or any
    array-like; a cupy array is rejected -- use FrbGrouper.create_events to build
    from GPU arrays) plus ``chunk_fpga_count`` (a non-negative int). All six arrays
    must have the same shape; events are emitted in row-major (C) order.
    """

    # (attribute, proto FrbEvent field, numpy dtype) -- order follows the proto.
    _FIELDS = (
        ("beam_ids",        "beam_id",        np.int32),
        ("fpga_timestamps", "fpga_timestamp", np.int64),
        ("dms",             "dm",             np.float32),
        ("dm_errors",       "dm_error",       np.float32),
        ("snrs",            "snr",            np.float32),
        ("rfi_probs",       "rfi_prob",       np.float32),
    )

    def __init__(self, beam_ids, fpga_timestamps, dms, dm_errors, snrs, rfi_probs,
                 chunk_fpga_count):
        try:
            self.chunk_fpga_count = int(chunk_fpga_count)
        except (TypeError, ValueError) as e:
            raise ValueError(f"FrbSifterEvents: chunk_fpga_count must be an integer, "
                             f"got {chunk_fpga_count!r}") from e
        if self.chunk_fpga_count < 0:
            raise ValueError(f"FrbSifterEvents: chunk_fpga_count must be >= 0, "
                             f"got {self.chunk_fpga_count}")

        values = dict(beam_ids=beam_ids, fpga_timestamps=fpga_timestamps, dms=dms,
                      dm_errors=dm_errors, snrs=snrs, rfi_probs=rfi_probs)
        for attr, _, dtype in self._FIELDS:
            val = values[attr]
            if type(val).__module__.split('.', 1)[0] == 'cupy':
                raise TypeError(f"FrbSifterEvents: argument {attr!r} is a cupy array; this "
                                f"class stores numpy arrays. Use FrbGrouper.create_events() "
                                f"to build a FrbSifterEvents from GPU arrays.")
            try:
                arr = np.asarray(val, dtype=dtype)
            except Exception as e:
                raise ValueError(f"FrbSifterEvents: could not convert argument {attr!r} to a "
                                 f"numpy {np.dtype(dtype).name} array: {e}") from e
            setattr(self, attr, arr)

        shapes = {attr: getattr(self, attr).shape for attr, _, _ in self._FIELDS}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"FrbSifterEvents: all event arrays must have the same shape, "
                             f"got {shapes}")
        self.shape = next(iter(shapes.values()))

    def to_proto(self):
        """Return a list of frb_sifter_pb2.FrbEvent, one per event (row-major order).

        Arrays of any shape are flattened; .tolist() yields the native Python
        scalars required by protobuf.
        """
        cols = {proto: getattr(self, attr).reshape(-1).tolist()
                for attr, proto, _ in self._FIELDS}
        return [frb_sifter_pb2.FrbEvent(**{f: cols[f][i] for f in cols})
                for i in range(len(self))]

    def __len__(self):
        """Number of events (== number of FrbEvent protos that will be sent)."""
        return int(self.beam_ids.size)

    def __repr__(self):
        return (f"FrbSifterEvents(num_events={len(self)}, shape={self.shape}, "
                f"chunk_fpga_count={self.chunk_fpga_count})")


class FrbSifterClient:
    """Python client for the FrbSifter gRPC service (grpc/frb_sifter.proto).

    The "sifter" is a central process that receives FRB events (and coarse-grained
    per-beam SNRs) from the groupers on all search nodes. This client lets other
    code open a connection to a sifter and send it messages. It wraps the two RPCs
    currently used by the prototype grouper (pirate_frb/run_chord_grouper.py):

      - send_configuration()  -- the initial ConfigMessage (config YAML strings).
      - send_events()         -- a per-time-period FrbEventsMessage (FRB events
                                 and/or coarse-grained per-beam SNRs).

    Both methods return None and raise a verbose RuntimeError on failure (a gRPC
    transport error, or a not-ok reply from the sifter).

    Usage::

        with FrbSifterClient("localhost:7100") as sifter:
            sifter.send_configuration(pirate_yaml, xengine_yaml,
                                       dedispersion_plan_yaml, grouper_yaml)
            sifter.send_events(has_injections, beam_set_id, events,
                               coarsegrain_start_fpga_count,
                               coarsegrain_end_fpga_count, coarsegrain_snr)

    where 'events' is a FrbSifterEvents (or None).
    """

    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = frb_sifter_pb2_grpc.FrbSifterStub(self.channel)

    def send_configuration(self, pirate_yaml, xengine_yaml,
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
            raise RuntimeError(f"FrbSifterClient.send_configuration: RPC to sifter at "
                               f"{self.server_address!r} failed: {e}") from e
        if not reply.ok:
            raise RuntimeError(f"FrbSifterClient.send_configuration: sifter at "
                               f"{self.server_address!r} returned ok=False")

    def send_events(self, has_injections, beam_set_id, events,
                    coarsegrain_start_fpga_count, coarsegrain_end_fpga_count,
                    coarsegrain_snr):
        """Send an FrbEventsMessage (FrbEvents RPC).

        'events' is a FrbSifterEvents (one FrbEvent is sent per element, and its
        chunk_fpga_count populates the message's chunk_fpga_count field), or None
        to send no per-event triggers (with chunk_fpga_count = 0).

        'coarsegrain_snr' may be a numpy or cupy 1-d array (or any 1-d iterable): a
        cupy array is copied to the host in a single shot via .get() (cupy raises on
        implicit np.asarray()), then cast to float32.

        Returns None; raises a verbose RuntimeError on failure (malformed input,
        gRPC transport error, or a not-ok reply).
        """
        if events is None:
            proto_events = []
            chunk_fpga_count = 0
        elif isinstance(events, FrbSifterEvents):
            proto_events = events.to_proto()
            chunk_fpga_count = events.chunk_fpga_count
        else:
            raise RuntimeError(f"FrbSifterClient.send_events: 'events' must be a "
                               f"FrbSifterEvents or None, got {type(events).__name__}")

        cg_snr = self._to_1d("coarsegrain_snr", coarsegrain_snr, np.float32)

        request = frb_sifter_pb2.FrbEventsMessage(
            has_injections = has_injections,
            beam_set_id = beam_set_id,
            chunk_fpga_count = chunk_fpga_count,
            events = proto_events,
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
