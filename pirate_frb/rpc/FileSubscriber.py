"""FileSubscriber - a live subscription to FrbServer file-write notifications."""

import grpc
from .grpc import frb_search_pb2


class FileSubscriber:
    """A live subscription to FrbServer file-write notifications.

    Constructed via FrbSearchClient.subscribe_files(). The constructor opens
    the gRPC stream AND blocks until the server confirms that the
    subscriber has been registered, so any WriteFiles calls issued
    AFTER the constructor returns are guaranteed to have their
    notifications delivered through this object's iterator.

    Iteration yields (filename, error_message, acq_name) tuples. An
    empty error_message indicates success; non-empty indicates an
    error. acq_name is "" for WriteFiles-triggered files and the
    stream's acq_name for stream-triggered files (only delivered when
    the subscription was opened with subscribe_streams=True).

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
    same FrbSearchClient on different threads are fine (each gets its own
    stream).

    Example -- context manager (recommended)::

        with client.subscribe_files() as sub:
            filenames = client.write_files(
                beams=[100, 101],
                fpga_seq_start=0, fpga_seq_end=4 * seq_per_chunk,
                acqdir="my_acquisition",
            )
            remaining = set(filenames)
            for filename, error, acq_name in sub:
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
        for filename, error, acq_name in sub:
            print(filename, error)
            if some_condition:
                break
        # When `sub` goes out of scope, __del__ runs and cancels
        # the stream. Works in practice but not guaranteed to run
        # before interpreter shutdown -- the context-manager form
        # above is preferred.
    """

    def __init__(self, stub, subscribe_streams=False):
        # Set _closed and _call FIRST so __del__ -> close() is safe
        # even if __init__ raises partway through. CPython runs
        # __del__ on any object whose __new__ succeeded, including
        # partially-constructed ones, so we cannot rely on
        # attributes set later in __init__.
        self._closed = False
        self._call = None
        # protocol_version stamps the stream-opening request (see notes/grpc.md);
        # the server rejects a version mismatch when opening the stream.
        request = frb_search_pb2.SubscribeFilesRequest(
            protocol_version=frb_search_pb2.PROTOCOL_VERSION_CURRENT,
            subscribe_streams=subscribe_streams)
        self._call = stub.SubscribeFiles(request)
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
            return (r.notification.filename, r.notification.error_message,
                    r.notification.acq_name)
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
