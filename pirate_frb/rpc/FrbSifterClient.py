import grpc
import numpy as np
import yaml

from .grpc import frb_sifter_pb2
from .grpc import frb_sifter_pb2_grpc


class FrbSifterEvents:
    """Helper class, representing FRB events to send to the sifter.

    Returned by FrbGrouper.create_events(), and passed to FrbSifterClient.send_events().

    Attributes (numpy arrays, shapes must be equal):
    
    - ``beam_ids`` (int32) -- X-engine beam id of each event
    - ``fpga_timestamps`` (int64) -- absolute FPGA-counter timestamp of each event
    - ``dms`` (float32) -- dispersion measure
    - ``snrs`` (float32) -- signal-to-noise ratio
    - ``rfi_probs`` (float32) -- RFI probability

    Other attributes:

    - ``chunk_fpga_start`` (int) -- FPGA count at the start of the time chunk.
    - ``chunk_fpga_end`` (int) -- FPGA count at the end of the time chunk (= start of the next chunk).
    - ``shape`` (tuple) -- common shape of the per-event arrays.
    """

    # (attribute, proto FrbEvent field, numpy dtype) -- order follows the proto.
    _FIELDS = (
        ("beam_ids",        "beam_id",        np.int32),
        ("fpga_timestamps", "fpga_timestamp", np.int64),
        ("dms",             "dm",             np.float32),
        ("snrs",            "snr",            np.float32),
        ("rfi_probs",       "rfi_prob",       np.float32),
    )

    @staticmethod
    def _check_fpga_count(name, value):
        """Coerce an FPGA-count scalar to a non-negative int (raises ValueError otherwise)."""
        try:
            value = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"FrbSifterEvents: {name} must be an integer, got {value!r}") from e
        if value < 0:
            raise ValueError(f"FrbSifterEvents: {name} must be >= 0, got {value}")
        return value

    def __init__(self, beam_ids, fpga_timestamps, dms, snrs, rfi_probs,
                 chunk_fpga_start, chunk_fpga_end):
        self.chunk_fpga_start = self._check_fpga_count("chunk_fpga_start", chunk_fpga_start)
        self.chunk_fpga_end = self._check_fpga_count("chunk_fpga_end", chunk_fpga_end)
        if self.chunk_fpga_end < self.chunk_fpga_start:
            raise ValueError(f"FrbSifterEvents: chunk_fpga_end ({self.chunk_fpga_end}) must be "
                             f">= chunk_fpga_start ({self.chunk_fpga_start})")

        values = dict(beam_ids=beam_ids, fpga_timestamps=fpga_timestamps, dms=dms,
                      snrs=snrs, rfi_probs=rfi_probs)
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
                f"chunk_fpga_start={self.chunk_fpga_start}, chunk_fpga_end={self.chunk_fpga_end})")


class FrbSifterClient:
    """
    The FrbSifterClient manages grouper-sifter communication (from the grouper side).

    The grouper sends two types of messages to the sifter: a ConfigMessage
    (sent once), and FrbEventsMessages (sent once per FPGA window, typically
    one per time chunk).

    If FrbSifterClient is used as a context manager, then close() is automatically called.

    Attributes (read-only):

    - ``server_address`` (str) -- the sifter's ``ip:port`` address.

    Usage::

        with FrbSifterClient("localhost:7100") as sifter:
            sifter.send_configuration(pirate_yaml, xengine_yaml,
                                       dedispersion_plan_yaml, grouper_yaml,
                                       search_ip_addr)
            sifter.send_events(beam_set_id, events, coarsegrain_snr)

    where 'events' is a FrbSifterEvents.
    """

    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = frb_sifter_pb2_grpc.FrbSifterStub(self.channel)

    def send_configuration(self, pirate_yaml, xengine_yaml,
                            dedispersion_plan_yaml, grouper_yaml,
                            search_ip_addr):
        """Send the initial ConfigMessage (CheckConfiguration RPC).

        Sent once, before any events. Each yaml argument may be either a str (sent
        unmodified) or any yaml-serializable object (``yaml.dump()``'d to a string
        before sending).

        Parameters
        ----------
        pirate_yaml : str or object
            Pirate dedispersion-configuration yaml.
        xengine_yaml : str or object
            X-engine metadata yaml.
        dedispersion_plan_yaml : str or object
            Dedispersion-plan yaml.
        grouper_yaml : str or object
            Grouper-specific configuration yaml.
        search_ip_addr : str
            The producer FrbServer's FrbSearch RPC endpoint (an ``ip:port`` string,
            typically ``grouper.search_ip_addr``), so the sifter can call back to
            pirate. Sent as-is (not yaml-serialized).

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            On failure: a yaml-serialization error, a gRPC transport error, or a
            not-ok reply from the sifter.
        """
        request = frb_sifter_pb2.ConfigMessage(
            # Always advertise the protocol version we were built against; the
            # sifter rejects the config if it does not match its own.
            protocol_version = frb_sifter_pb2.PROTOCOL_VERSION_CURRENT,
            pirate_yaml = self._to_yaml("pirate_yaml", pirate_yaml),
            xengine_yaml = self._to_yaml("xengine_yaml", xengine_yaml),
            dedispersion_plan_yaml = self._to_yaml("dedispersion_plan_yaml", dedispersion_plan_yaml),
            grouper_yaml = self._to_yaml("grouper_yaml", grouper_yaml),
            search_ip_addr = search_ip_addr,   # plain 'ip:port' string (sifter -> pirate callbacks)
        )
        try:
            reply = self.stub.CheckConfiguration(request)
        except grpc.RpcError as e:
            raise RuntimeError(f"FrbSifterClient.send_configuration: RPC to sifter at "
                               f"{self.server_address!r} failed: {e}") from e
        if not reply.ok:
            raise RuntimeError(f"FrbSifterClient.send_configuration: sifter at "
                               f"{self.server_address!r} returned ok=False")

    def send_events(self, beam_set_id, events, coarsegrain_snr, from_simulator=False):
        """Send an FrbEventsMessage (FrbEvents RPC).

        Sent once per FPGA window, after send_configuration().

        Parameters
        ----------
        beam_set_id : int
            X-engine beam-set id.
        events : :class:`~pirate_frb.rpc.FrbSifterEvents`
            The per-event triggers (one FrbEvent per array element) AND the FPGA
            window ``[chunk_fpga_start, chunk_fpga_end]`` this message covers -- the
            window is taken from this object. For a coarse-grain-only message, pass an
            FrbSifterEvents with empty per-event arrays but a valid window.
        coarsegrain_snr : array_like
            Per-beam coarse-grained max SNR over the same FPGA window, as a 1-d numpy
            or cupy array (or any 1-d iterable). A cupy array is copied to the host in
            a single shot via ``.get()`` (cupy raises on implicit ``np.asarray()``),
            then cast to float32.
        from_simulator : bool, optional
            False (default) for events produced by the real search pipeline; True for
            the parallel "ideal" events the fake X-engine emits when simulating pulses
            upstream (so the sifter can tell the two apart).

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            On failure: malformed input, a gRPC transport error, or a not-ok reply
            from the sifter.
        """
        if not isinstance(events, FrbSifterEvents):
            raise RuntimeError(f"FrbSifterClient.send_events: 'events' must be a "
                               f"FrbSifterEvents, got {type(events).__name__}")

        cg_snr = self._to_1d("coarsegrain_snr", coarsegrain_snr, np.float32)

        request = frb_sifter_pb2.FrbEventsMessage(
            from_simulator = from_simulator,
            beam_set_id = beam_set_id,
            chunk_fpga_start = events.chunk_fpga_start,
            chunk_fpga_end = events.chunk_fpga_end,
            events = events.to_proto(),
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
        """Close the gRPC channel.

        Automatically called if the FrbSifterClient is used as a context manager.
        """
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"FrbSifterClient({self.server_address!r})"
