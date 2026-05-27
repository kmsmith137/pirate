"""FrbClient - Python client for FrbServer gRPC service."""

import grpc
from .grpc import frb_search_pb2
from .grpc import frb_search_pb2_grpc


class FileSubscriber:
    """A live subscription to FrbServer file-write notifications.

    Constructed via FrbClient.subscribe_files(). The constructor opens
    the gRPC stream AND blocks until the server confirms that the
    subscriber has been registered, so any WriteFiles calls issued
    AFTER the constructor returns are guaranteed to have their
    notifications delivered through this object's iterator.

    Iteration yields (filename, error_message) tuples. An empty
    error_message indicates success; non-empty indicates an error.

    Lifetime: the underlying gRPC stream stays open until either
    close() is called (explicitly or via __exit__/__del__), the
    server cancels the stream (e.g. on shutdown), or an RPC error
    occurs. Use as a context manager for deterministic teardown;
    "sloppy" use (let __del__ clean up at GC time) usually works
    but relies on CPython's reference-counting timeliness and
    gRPC's interpreter-shutdown behavior, neither of which is
    formally guaranteed.

    Not thread-safe: a single FileSubscriber must be iterated from
    one thread at a time. Multiple FileSubscriber objects from the
    same FrbClient on different threads are fine (each gets its own
    stream).

    Example -- context manager (recommended)::

        with client.subscribe_files() as sub:
            filenames = client.write_files(
                beams=[100, 101],
                min_time_chunk_index=0, max_time_chunk_index=4,
                filename_pattern="x_(BEAM)_(CHUNK).asdf",
            )
            remaining = set(filenames)
            for filename, error in sub:
                if error:
                    raise RuntimeError(f"{filename}: {error}")
                remaining.discard(filename)
                if not remaining:
                    break
        # On exit the stream is cancelled and any in-flight
        # notifications buffered by gRPC are dropped.

    Example -- "sloppy" (let __del__ clean up)::

        sub = client.subscribe_files()
        client.write_files(...)
        for filename, error in sub:
            print(filename, error)
            if some_condition:
                break
        # When `sub` goes out of scope, __del__ runs and cancels
        # the stream. Works in practice but not guaranteed to run
        # before interpreter shutdown -- the context-manager form
        # above is preferred.
    """

    def __init__(self, stub):
        # Set _closed and _call FIRST so __del__ -> close() is safe
        # even if __init__ raises partway through. CPython runs
        # __del__ on any object whose __new__ succeeded, including
        # partially-constructed ones, so we cannot rely on
        # attributes set later in __init__.
        self._closed = False
        self._call = None
        self._call = stub.SubscribeFiles(frb_search_pb2.SubscribeFilesRequest())
        try:
            first = next(self._call)
        except StopIteration:
            # Server closed the stream without emitting the sentinel
            # (shouldn't happen unless the server stopped between
            # add_subscriber() and Write(ready)).
            raise RuntimeError(
                "FileSubscriber: server closed stream before sending "
                "the ready sentinel"
            )
        kind = first.WhichOneof("kind")
        if kind != "ready":
            raise RuntimeError(
                f"FileSubscriber: first stream response had kind={kind!r}, "
                "expected 'ready' -- protocol version mismatch?"
            )

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            r = next(self._call)
        except StopIteration:
            self._closed = True
            raise
        except grpc.RpcError:
            # If our own close() cancelled the call (this thread or
            # another), surface as StopIteration so iteration ends
            # cleanly. Other RpcErrors (server crash, network failure,
            # etc.) propagate unchanged.
            if self._closed:
                raise StopIteration
            raise
        kind = r.WhichOneof("kind")
        if kind == "notification":
            return (r.notification.filename, r.notification.error_message)
        # Defense-in-depth: the server should never send a second
        # ready sentinel, nor any unknown oneof case. (Forward-
        # compat: a future server emitting a new oneof variant
        # would land here too; we fail loud rather than silently
        # dropping.)
        raise RuntimeError(
            f"FileSubscriber: unexpected stream message kind={kind!r} "
            "mid-stream (protocol violation or version mismatch)"
        )

    def close(self):
        """Cancel the stream. Idempotent; safe to call multiple times.

        After close(), iteration raises StopIteration on any thread
        (including one that was blocked in next() at the moment
        close() ran -- __next__'s grpc.RpcError handler converts
        our-own-cancel into StopIteration). Any in-flight
        notifications buffered by gRPC but not yet consumed are
        dropped.
        """
        if self._closed:
            return
        self._closed = True
        if self._call is not None:
            try:
                self._call.cancel()
            except Exception:
                # cancel() on an already-cancelled or finished call
                # may raise depending on grpc-python version;
                # swallow.
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __del__(self):
        # Best-effort cleanup for the "sloppy" pattern. May run
        # during interpreter shutdown when grpc internals are
        # already torn down -- the try/except in close() swallows
        # any resulting exception, and Python silently ignores
        # exceptions from __del__ anyway.
        try:
            self.close()
        except Exception:
            pass


class FrbClient:
    """Client for querying FrbServer via gRPC.
    
    Usage:
        with FrbClient("localhost:50051") as client:
            status = client.get_status()
            print(f"Connections: {status.num_connections}, Bytes: {status.num_bytes}")
    
    Or without context manager:
        client = FrbClient("localhost:50051")
        status = client.get_status()
        client.close()
    """
    
    def __init__(self, server_address: str = "localhost:50051"):
        """Create a client connected to the given server address.
        
        Args:
            server_address: gRPC server address (e.g. "localhost:50051")
        """
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = frb_search_pb2_grpc.FrbSearchStub(self.channel)
    
    def get_status(self):
        """Query the server for current status.
        
        Returns:
            GetStatusResponse protobuf message with the following int fields:
            - num_connections: Total number of active TCP connections (summed over receivers)
            - num_bytes: Total bytes received (summed over receivers)
            - rb_start: First frame_id in ring buffer
            - rb_reaped: (Last reaped frame_id) + 1
            - rb_processed: (Last GPU-processed frame_id) + 1; rpc-writeable upper bound
            - rb_assembled: (Last fully-assembled frame_id) + 1
            - rb_end: (Last frame_id in ring buffer) + 1
            - num_free_frames: Number of available frames in AssembledFrameAllocator
        """
        request = frb_search_pb2.GetStatusRequest()
        return self.stub.GetStatus(request)

    def get_config(self):
        """Query the server for its run-time configuration.

        Returns:
            GetConfigResponse protobuf message with fields:
            - rpc_ip_addr: "ip:port" this server's RPC is bound to
            - data_ip_addrs: list of "ip:port" strings, one per Receiver
            - time_samples_per_chunk: int
            - ringbuf_nchunks: logical ring buffer length (in time chunks)
            - ssd_dir: SSD cache directory
            - nfs_dir: NFS output directory (already interpolated for {user}/{date})
            - ssd_threads: number of FileWriter SSD threads
            - nfs_threads: number of FileWriter NFS threads
            - tree_rank: from config_prefilled
            - beams_per_batch: from config_prefilled
            - min_data_mtu: minimum data-NIC MTU expected on the sender side
            - fake_zone_nfreq: list[int] from config_prefilled.zone_nfreq (pre-metadata)
            - fake_zone_freq_edges: list[float] from config_prefilled.zone_freq_edges (pre-metadata)
            - fake_time_sample_ms: float from config_prefilled.time_sample_ms (pre-metadata)
            - fake_nbeams: int from config_prefilled.beams_per_gpu (pre-metadata)

        The 'fake_*' fields are what a fake X-engine sender should mimic;
        they're the pre-metadata values the receiver was started with, not
        what a real X-engine subsequently sent.
        """
        request = frb_search_pb2.GetConfigRequest()
        return self.stub.GetConfig(request)

    def get_xengine_metadata(self, verbose: bool = False) -> str:
        """Query the server for XEngine metadata as a YAML string.

        Args:
            verbose: If True, include comments explaining each field.

        Returns:
            YAML string representation of XEngine metadata, or empty string
            if metadata is not yet available.
        """
        request = frb_search_pb2.GetXEngineMetadataRequest(verbose=verbose)
        response = self.stub.GetXEngineMetadata(request)
        return response.yaml_string

    def write_files(
        self,
        beams: list[int],
        min_time_chunk_index: int,
        max_time_chunk_index: int,
        filename_pattern: str
    ) -> list[str]:
        """Request the server to write files to disk.

        Args:
            beams: List of beam IDs to write.
            min_time_chunk_index: First time chunk index (inclusive).
            max_time_chunk_index: Last time chunk index (inclusive).
            filename_pattern: Pattern with (BEAM) and (CHUNK) placeholders,
                e.g. "dir1/dir2/file_(BEAM)_(CHUNK).asdf"

        Returns:
            List of filenames that will be written.
        """
        request = frb_search_pb2.WriteFilesRequest(
            beams=beams,
            min_time_chunk_index=min_time_chunk_index,
            max_time_chunk_index=max_time_chunk_index,
            filename_pattern=filename_pattern
        )
        response = self.stub.WriteFiles(request)
        return list(response.filename_list)

    def subscribe_files(self):
        """Open a file-write-notification subscription.

        Returns a FileSubscriber whose constructor has already opened
        the stream and consumed the server's ready sentinel, so any
        WriteFiles calls issued AFTER this method returns are
        guaranteed to have their notifications delivered through the
        returned object's iterator.

        See FileSubscriber for usage examples (context-manager and
        sloppy forms) and lifetime semantics.
        """
        return FileSubscriber(self.stub)

    def monitor_ringbuf(self):
        """Subscribe to a server push stream of rb_processed updates.

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
        request = frb_search_pb2.MonitorRingbufRequest()
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
        return f"FrbClient({self.server_address!r})"
