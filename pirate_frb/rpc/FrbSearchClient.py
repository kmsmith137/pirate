"""FrbSearchClient - Python client for FrbServer gRPC service."""

import functools
import datetime

import yaml
import grpc
from .grpc import frb_search_pb2
from .grpc import frb_search_pb2_grpc
from .FileSubscriber import FileSubscriber

# Our wire-protocol version, stamped onto every request's protocol_version
# field (see notes/grpc.md). The server rejects any RPC whose value does not
# match its own PROTOCOL_VERSION_CURRENT, catching out-of-sync pirate builds.
_PROTOCOL_VERSION = frb_search_pb2.PROTOCOL_VERSION_CURRENT


class FrbSearchClient:
    """Client for querying FrbServer via gRPC.

    Usage (as a context manager)::

        with FrbSearchClient("127.0.0.1:6000") as client:
            status = client.get_status()
            print(f"Connections: {status.num_connections}, Bytes: {status.num_bytes}")

    Or without a context manager::

        client = FrbSearchClient("127.0.0.1:6000")
        status = client.get_status()
        client.close()
    """
    
    def __init__(self, server_address: str):
        """Create a client connected to the given server address.

        Every RPC issued by this client stamps its request with our
        _PROTOCOL_VERSION; the server rejects an RPC on a version mismatch
        (raising grpc.RpcError) if the client and server pirate builds are
        out of sync.

        Args:
            server_address: gRPC server address (e.g. "127.0.0.1:6000")
        """
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = frb_search_pb2_grpc.FrbSearchStub(self.channel)
        # Cache for the X-engine metadata YAML string, populated by
        # _try_xengine_metadata() once the server has it (static thereafter).
        self._xengine_metadata_yaml = None

    def get_status(self):
        """Query the server for current status, by sending a GetStatus RPC.

        Returns:
            GetStatusResponse protobuf message with the following int fields:
            - num_connections: Total number of active TCP connections (summed over receivers)
            - num_bytes: Total bytes received (summed over receivers)
            - rb_start: First frame_id in ring buffer
            - rb_reaped: (Last reaped frame_id) + 1
            - rb_processed: (Last GPU-processed frame_id) + 1
            - rb_streamed: Defines split between WriteFiles and streams
            - rb_assembled: (Last fully-assembled frame_id) + 1
            - rb_end: (Last frame_id in ring buffer) + 1
            - num_free_frames: Number of available frames in AssembledFrameAllocator
        """
        request = frb_search_pb2.GetStatusRequest(protocol_version=_PROTOCOL_VERSION)
        return self.stub.GetStatus(request)

    @functools.cached_property
    def config(self):
        """The server's run-time configuration (cached).

        Static for the server's lifetime, so it is fetched from the server once
        on first access (a GetConfig RPC) and reused thereafter. A
        GetConfigResponse protobuf message with fields:

        - rpc_ip_addr: "ip:port" this server's RPC is bound to
        - data_ip_addrs: list of "ip:port" strings, one per Receiver
        - time_samples_per_chunk: int
        - ringbuf_nchunks: logical ring buffer length (in time chunks)
        - ssd_dir: SSD cache directory
        - nfs_dir: NFS output directory (already interpolated for {user}/{date}/{home})
        - ssd_threads: number of FileWriter SSD threads
        - nfs_threads: number of FileWriter NFS threads
        - toplevel_tree_rank: from config_prefilled
        - beams_per_batch: from config_prefilled
        - frequency_subband_counts: list[int] from config_prefilled (a real
          search-config value, not a ``fake_`` default)
        - min_data_mtu: minimum data-NIC MTU expected on the sender side
        - fake_zone_nfreq: list[int] from config_prefilled.zone_nfreq (pre-metadata)
        - fake_zone_freq_edges: list[float] from config_prefilled.zone_freq_edges (pre-metadata)
        - fake_time_sample_ms: float from config_prefilled.time_sample_ms (pre-metadata)
        - fake_nbeams: int from config_prefilled.beams_per_gpu (pre-metadata)

        The 'fake_*' fields are what a fake X-engine sender should mimic;
        they're the pre-metadata values the receiver was started with, not
        what a real X-engine subsequently sent.
        """
        request = frb_search_pb2.GetConfigRequest(protocol_version=_PROTOCOL_VERSION)
        return self.stub.GetConfig(request)

    def _try_xengine_metadata(self):
        """Fetch the server's X-engine metadata YAML string, by sending a
        GetXEngineMetadata RPC (only when the value is not already cached).

        Returns the (terse, non-verbose) YAML string, or None if the server
        has not yet received metadata from the X-engine. The first non-empty
        result is cached, so later calls return it without another RPC; a None
        result is NOT cached, so a subsequent call retries. This is the
        non-raising primitive a caller polls (e.g. run_rpc_status); the
        xengine_* / beam_ids accessors below wrap it and raise when it is None.

        Always the terse form: the parsed tiers ignore YAML comments, and
        nothing needs the commented dump over RPC.

        Raises grpc.RpcError on transport failure (a server that is up but has
        no metadata yet returns None, which is not an error).
        """
        if self._xengine_metadata_yaml is None:
            request = frb_search_pb2.GetXEngineMetadataRequest(
                protocol_version=_PROTOCOL_VERSION, verbose=False)
            yaml_string = self.stub.GetXEngineMetadata(request).yaml_string
            if not yaml_string:
                return None   # not yet available; do NOT cache
            self._xengine_metadata_yaml = yaml_string
        return self._xengine_metadata_yaml

    @property
    def xengine_metadata_yaml_string(self) -> str:
        """The server's X-engine metadata as a YAML string.

        On the first access that finds the value uncached, this sends a
        GetXEngineMetadata RPC (via _try_xengine_metadata(), which holds the
        cache); later accesses do not re-hit the server.

        The single "not-ready" choke point shared by xengine_metadata_yaml,
        xengine_metadata, and beam_ids: it raises RuntimeError if the metadata
        is not available yet. For a non-raising, pollable form (returns None
        while unavailable), use _try_xengine_metadata() -- which also holds the
        cache, so repeated access here does not re-hit the server.

        Raises:
            RuntimeError: metadata not yet available.
            grpc.RpcError: transport failure.
        """
        yaml_string = self._try_xengine_metadata()
        if yaml_string is None:
            raise RuntimeError(
                f"{self!r}: X-engine metadata not yet available "
                "(the server has not received it from the X-engine yet)")
        return yaml_string

    @functools.cached_property
    def xengine_metadata_yaml(self) -> dict:
        """The server's X-engine metadata, parsed to a plain dict (cached).

        The parsed form of xengine_metadata_yaml_string. Keys include
        dt_ns_per_seq, unix_ns_at_seq_0, seq_per_frb_time_sample, zone_nfreq,
        zone_freq_edges, beamset, beam_ids, and version.

        On first access, may send a GetXEngineMetadata RPC (via
        xengine_metadata_yaml_string).

        Raises RuntimeError / grpc.RpcError; see xengine_metadata_yaml_string.
        """
        return yaml.safe_load(self.xengine_metadata_yaml_string)

    @functools.cached_property
    def xengine_metadata(self):
        """The server's X-engine metadata, parsed to a typed
        pirate_frb.core.XEngineMetadata (cached).

        beam_ids and beam_positions_{x,y} ARE populated (the server serializes
        them via XEngineMetadata::to_yaml_string). freq_channels may be empty:
        it is only meaningful as a per-sender channel subset, and the server's
        aggregated metadata does not carry one.

        On first access, may send a GetXEngineMetadata RPC (via
        xengine_metadata_yaml_string).

        Raises RuntimeError / grpc.RpcError; see xengine_metadata_yaml_string.
        """
        from ..pirate_pybind11 import XEngineMetadata
        return XEngineMetadata.from_yaml_string(self.xengine_metadata_yaml_string)

    @functools.cached_property
    def beam_ids(self) -> tuple[int, ...]:
        """The beam IDs processed by this server (cached).

        Static for the server's lifetime. Sourced from the X-engine metadata
        (the same values ShowStreams reports); returned as a tuple so the
        shared cached value cannot be mutated by a caller.

        On first access, may send a GetXEngineMetadata RPC (via
        xengine_metadata_yaml_string).

        Raises RuntimeError / grpc.RpcError; see xengine_metadata_yaml_string.
        """
        return tuple(self.xengine_metadata.beam_ids)

    def write_files(
        self,
        beams: list[int],
        fpga_seq_start: int,
        fpga_seq_end: int,
        acqdir: str
    ) -> list[str]:
        """Request the server to write files to disk, by sending a WriteFiles RPC.

        Files are written to {nfs_root}/{acqdir}/frame_b(BEAM)_t(CHUNK).asdf,
        where nfs_root comes from the server config (self.config.nfs_dir).

        The range may extend into the future ("future writes"): chunks not
        yet processed are scheduled automatically, up to the server config's
        future_write_max_samples (rounded up to a whole chunk) past the
        current processing threshold (get_status().rb_streamed); the excess
        is silently truncated. Future files appear in the returned list
        immediately -- the list is a promise whose tail lands later -- and
        are written (and reported to subscribe_files() subscribers, like any
        write_files-triggered write) as the data is processed.

        Note: I recommend using a unique acqdir for each multibeam event (or stream).
        Our postprocessing tools generally assume contiguous time chunks (or close to
        that) within an acqdir. If the same file ends up in multiple acqdirs,
        hardlinks will be used to avoid copying and save space.

        Args:
            beams: List of beam IDs to write (duplicates are rejected).
            fpga_seq_start: Start of the fpga-seq range (inclusive).
            fpga_seq_end: End of the fpga-seq range (exclusive). Files are
                written for all chunks overlapping
                fpga_seq_start <= f < fpga_seq_end.
            acqdir: Acquisition directory: a nonempty, normalized relative
                path (no leading/trailing '/', no '//', no '.' or '..'
                components); may be multi-level, e.g. "foo/bar/baz".

        Returns:
            List of filenames that will be written (nfs_root-relative),
            sorted ascending by (time chunk, then order of the beam in
            'beams'). Truncation of the requested range (past data that has
            left the ring buffer, or future data beyond
            future_write_max_samples) is reported only implicitly, via the
            missing filenames.
        """
        request = frb_search_pb2.WriteFilesRequest(
            protocol_version=_PROTOCOL_VERSION,
            beams=beams,
            fpga_seq_start=fpga_seq_start,
            fpga_seq_end=fpga_seq_end,
            acqdir=acqdir
        )
        response = self.stub.WriteFiles(request)
        return list(response.filename_list)

    def start_stream(
        self,
        beam_ids: list[int],
        stream_name: str = None,
        acqdir: str = None,
        fpga_seq_start: int = 0,
        fpga_seq_end: int = None
    ) -> tuple[str, str]:
        """Register a "stream", by sending a StartStream RPC: data matching
        (beam_ids x fpga-seq range) is queued for disk writing automatically as
        it flows through the server.

        Files are written to {nfs_root}/{acqdir}/frame_b(BEAM)_t(CHUNK).asdf,
        where nfs_root comes from the server config (self.config.nfs_dir).

        Complements write_files() (one-shot; retroactive within the ring
        buffer, plus a bounded look-ahead via future_write_max_samples): a
        stream captures each frame at the moment it is processed, can run
        arbitrarily far into the future, and its fpga_seq range can be
        open-ended -- but chunks that were already processed when
        StartStream arrives are NOT captured retroactively, even if
        fpga_seq_start is in the past.

        Note: I recommend using a unique acqdir for each multibeam event (or stream).
        Our postprocessing tools generally assume contiguous time chunks (or close to
        that) within an acqdir. If the same file ends up in multiple acqdirs,
        hardlinks will be used to avoid copying and save space.

        Args:
            beam_ids: Nonempty list of beam IDs (no all-beams convention;
                list beams explicitly -- show_streams() returns the full list).
            stream_name: Nonempty identifier, unique among active streams
                (used by show_streams() / cancel_stream()). If None (default),
                a name "stream_{date}_{time}" is generated, e.g.
                "stream_26_07_07_143052".
            acqdir: Acquisition directory (same conventions as
                write_files()). If None (default), stream_name is used, so the
                acquisition lands at {nfs_root}/{stream_name}/.
            fpga_seq_start: Start of the fpga-seq range (inclusive);
                0 (default) means "start asap".
            fpga_seq_end: End of the fpga-seq range (exclusive). None
                (default) means "run indefinitely" (sent as 2**63 - 1 on
                the wire).

        Returns:
            (stream_name, acqdir): the resolved values that were sent
            (useful to the caller when either defaulted from None).

        Raises grpc.RpcError on validation failure (empty/duplicate stream_name,
        unknown beam_id, bad acqdir, range entirely in the past, or server
        not yet initialized).
        """
        if stream_name is None:
            # "stream_{date}_{time}", e.g. stream_26_07_07_143052.
            stream_name = 'stream_' + datetime.datetime.now().strftime('%y_%m_%d_%H%M%S')

        if acqdir is None:
            acqdir = stream_name

        if fpga_seq_end is None:
            fpga_seq_end = 2**63 - 1   # "run indefinitely"

        request = frb_search_pb2.StartStreamRequest(
            protocol_version=_PROTOCOL_VERSION,
            stream_name=stream_name,
            acqdir=acqdir,
            beam_ids=beam_ids,
            fpga_seq_start=fpga_seq_start,
            fpga_seq_end=fpga_seq_end
        )
        self.stub.StartStream(request)
        return stream_name, acqdir

    def show_streams(self):
        """Query the server for its streams (active + recent history), by
        sending a ShowStreams RPC.

        Returns:
            ShowStreamsResponse protobuf message with fields:
            - current_fpga_seq: the server's current position as an fpga seq,
              derived from rb_processed (all data before this fpga seq has
              been fully processed).
            - beam_ids: ALL beams processed by this server (not just those
              with active streams).
            - streams: list of StreamInfo -- active streams first, then
              recently-deactivated (expired/cancelled) streams in
              deactivation order. Each has 'args' (the original
              StartStreamRequest, echoed back), 'status' (a StreamStatus:
              ACTIVE, DRAINING = deactivated with writes still in flight,
              or INACTIVE = deactivated and fully drained), 'cancelled'
              (true = CancelStream, false = expired; meaningful when not
              ACTIVE), wall-clock timestamps 'started_at_unix_ns' /
              'deactivated_at_unix_ns' (0 while active), and counters
              'num_files_queued' / 'num_files_written' /
              'num_files_errored' (written + errored <= queued always;
              equality on a deactivated stream = fully drained).
            - num_deactivated_streams: total streams ever deactivated; if
              it exceeds the non-ACTIVE entries listed, older history was
              dropped (only the last few are retained).
        """
        request = frb_search_pb2.ShowStreamsRequest(protocol_version=_PROTOCOL_VERSION)
        return self.stub.ShowStreams(request)

    def cancel_stream(self, stream_name: str = None, cancel_all: bool = False) -> int:
        """Cancel one active stream (by stream_name), or all of them, by
        sending a CancelStream RPC.

        File writes already queued still complete (and still notify
        subscribe_files() subscribers); cancellation only stops future
        matching. Cancelled streams remain visible in show_streams()'s
        bounded history.

        Args:
            stream_name: Stream to cancel (ignored if cancel_all=True).
                An unknown or already-inactive stream_name raises
                grpc.RpcError.
            cancel_all: If True, cancel all active streams (however many;
                returns the full count even if the display history retains
                only the most recent few).

        Returns:
            Number of streams cancelled.
        """
        request = frb_search_pb2.CancelStreamRequest(
            protocol_version=_PROTOCOL_VERSION,
            cancel_all=cancel_all,
            stream_name=("" if stream_name is None else stream_name)
        )
        response = self.stub.CancelStream(request)
        return response.num_cancelled

    def subscribe_files(self, subscribe_streams: bool = False):
        """Open a file-write-notification subscription (a SubscribeFiles
        server-streaming RPC).

        Returns a FileSubscriber whose constructor has already opened
        the stream and consumed the server's ready sentinel, so any
        WriteFiles calls issued AFTER this method returns are
        guaranteed to have their notifications delivered through the
        returned object's iterator.

        Args:
            subscribe_streams: If True, also receive notifications for
                files written by streams (start_stream()). Default False:
                WriteFiles-triggered notifications only.

        See FileSubscriber for usage examples (context-manager and
        sloppy forms) and lifetime semantics.
        """
        return FileSubscriber(self.stub, subscribe_streams)

    def monitor_ringbuf(self):
        """Subscribe to a server push stream of rb_processed updates, by opening
        a MonitorRingbuf server-streaming RPC.

        SPECIAL-PURPOSE: this RPC exists for the FakeXEngine "pacing"
        feature, which gates the sender's chunk rate against the
        server's GPU processing rate. Don't use it from new code
        without a similar push-based use case in mind -- for general
        status polling, use get_status() instead.

        Yields int64 rb_processed values, one per change. The stream
        starts with the current value, sent by the server as soon as
        the ring buffer is initialized. Iteration ends when the
        server closes the stream (e.g. FrbServer::stop() was called).

        Raises grpc.RpcError on transport failure. To end the stream
        cleanly, break out of the for-loop and let the generator go
        out of scope -- gRPC cancels the underlying call on GC.
        """
        request = frb_search_pb2.MonitorRingbufRequest(protocol_version=_PROTOCOL_VERSION)
        for response in self.stub.MonitorRingbuf(request):
            yield response.rb_processed

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"FrbSearchClient({self.server_address!r})"
