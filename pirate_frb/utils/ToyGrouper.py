"""Toy CUDA-IPC consumer that pairs with pirate_frb.utils.ToyIPC.

See plans/grouper.md for the full design. Summary:

- Opens a gRPC server on a user-supplied address.
- The first ToyIpcStream RPC invocation is the "session". A reader
  sub-thread iterates the request_iterator, processes the initial
  IpcHandle (-> cudaIpcOpenMemHandle -> cupy view), then handles
  PRODUCED(N) notifications by bumping rb_end. The bidi handler's
  own generator body yields CONSUMED(N) responses off a queue.
- A second concurrent Stream invocation is rejected with
  FAILED_PRECONDITION ("toy assumes a single client").

The one user-facing entry point is receive(), which blocks until the
IPC handshake has completed AND a slot has been produced, then prints
the slot and enqueues a CONSUMED(N) notification.
"""

import queue
import threading
import time
from concurrent import futures

import cupy
import cupy.cuda
import cupy.cuda.runtime as cuda_runtime
import grpc

from ..rpc.grpc import frb_grouper_pb2 as pb
from ..rpc.grpc import frb_grouper_pb2_grpc as pb_grpc


# Shape (5, 2) float32: 5 slots * 2 floats * 4 bytes/float = 40 bytes.
_RINGBUF_NBYTES = 5 * 2 * 4


class _ToyIpcServicer(pb_grpc.ToyIpcStreamServicer):
    def __init__(self, owner):
        self.owner = owner

    def Stream(self, request_iterator, context):
        # Reject duplicate clients. The toy assumes exactly one
        # concurrent client; a second Stream would silently corrupt
        # shared state.
        with self.owner.lock:
            if self.owner._client_connected:
                # abort() raises -- unreachable past this line.
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "ToyGrouper: another client is already connected")
            self.owner._client_connected = True

        # Python sync gRPC handlers are generators (they yield
        # responses), so we cannot also iterate request_iterator
        # inline. Spawn a sub-thread to handle ingress while this
        # generator yields egress.
        reader = threading.Thread(
            target=self.owner._reader_main,
            args=(request_iterator, context),
            name="ToyGrouper.reader",
            daemon=True,
        )
        reader.start()

        try:
            while True:
                # Sentinel = None -> stop yielding (handler shutting down).
                consumed_n = self.owner._consumed_q.get()
                if consumed_n is None:
                    return
                yield pb.ConsumerMessage(consumed=pb.Consumed(slot=consumed_n))
        finally:
            # Cancelling the context unblocks the request_iterator
            # in the reader thread, which then exits cleanly.
            context.cancel()
            reader.join()
            with self.owner.lock:
                self.owner._client_connected = False


class ToyGrouper:
    """gRPC server + cupy-view consumer of a CUDA-IPC ring buffer.

    Pairs with the C++ class pirate_frb.utils.ToyIPC. Use as a context
    manager for deterministic teardown.

    Args:
        listen_address: 'ip:port' to bind (e.g. '127.0.0.1:6817').
        cuda_device_id: CUDA device to use. Must match the producer's,
            or cudaIpcOpenMemHandle will refuse the handle.
    """

    def __init__(self, listen_address, cuda_device_id=0):
        self.cuda_device_id = int(cuda_device_id)
        # Make sure this process is on the right device before any cupy
        # calls. cudaIpcOpenMemHandle is per-thread, so the reader thread
        # will repeat this on entry to _init_ringbuf().
        cupy.cuda.Device(self.cuda_device_id).use()

        # Mutex-protected state.
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.is_stopped = False
        self.error = None
        self.rb_start = 0
        self.rb_end = 0
        self.ringbuf = None          # cupy.ndarray, set in _init_ringbuf()
        self._ipc_dev_ptr = None     # int (device pointer), for ipcCloseMemHandle
        self._handle_ready = threading.Event()
        self._client_connected = False
        self._handle_seen = False    # exactly-one IpcHandle, before any PRODUCED

        # Queue of CONSUMED(N) values awaiting the handler's yield.
        # None is a poison-pill sentinel meaning "shutdown".
        self._consumed_q = queue.Queue()

        # max_workers=2: a single Stream handler + a little headroom.
        # We also rely on max_workers >= 2 so that the duplicate-client
        # rejector itself can be served without deadlock (gRPC dispatches
        # each incoming Stream RPC on a fresh worker, and we reject from
        # the worker's body).
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        pb_grpc.add_ToyIpcStreamServicer_to_server(_ToyIpcServicer(self), self._server)
        self._server.add_insecure_port(listen_address)
        self._server.start()
        print(f"ToyGrouper: gRPC server listening on {listen_address}, "
              "waiting for a client to connect...")

    # ---- Public API ----

    def receive(self):
        """Consume one produced ring-buffer slot. Blocks if necessary."""
        # 1. Wait for the IPC handshake to complete. _handle_ready is
        # also set by stop() / by _reader_main on error, so we always
        # re-check is_stopped after wake-up. Time spent here counts
        # toward 'wait for peer' on the first receive() call (when the
        # producer hasn't connected yet); on subsequent calls it's ~0.
        wait_start = time.monotonic()
        self._handle_ready.wait()

        # 2. Wait for at least one produced slot. Verify invariants.
        with self.cv:
            self._raise_if_stopped("receive")
            while self.rb_end <= self.rb_start:
                self.cv.wait()
                self._raise_if_stopped("receive")
            assert 0 <= self.rb_start < self.rb_end <= self.rb_start + 5, \
                f"ToyGrouper: bad invariants " \
                f"(rb_start={self.rb_start}, rb_end={self.rb_end})"
            n = self.rb_start
            slot = n % 5
            wait_sec = time.monotonic() - wait_start

        # 3. Print the slot. cupy auto-syncs on copy-to-host (the
        # ndarray.__repr__ path calls asnumpy under the hood).
        print(f"ToyGrouper.receive: slot={slot} value={self.ringbuf[slot]} "
              f"(waited {wait_sec:.3f} sec)")

        # 4. Enqueue CONSUMED(n) for the handler thread to yield.
        # Do this BEFORE bumping rb_start so that on a queue.put
        # failure the slot is not lost from the consumer's view.
        self._consumed_q.put(n)

        # 5. Bump rb_start.
        with self.cv:
            assert self.rb_start == n, \
                f"ToyGrouper: concurrent receive()? " \
                f"rb_start={self.rb_start} expected {n}"
            self.rb_start = n + 1
            assert self.rb_start <= self.rb_end
            self.cv.notify_all()

    def stop(self):
        """Stop the gRPC server and release the IPC mapping. Idempotent."""
        with self.cv:
            if self.is_stopped:
                return
            self.is_stopped = True
            self.cv.notify_all()

        # Unblock the handler thread + any receive() that's waiting on
        # the handshake.
        self._consumed_q.put(None)
        self._handle_ready.set()

        # Shut down the gRPC server (with a short grace period so the
        # handler can drain and the reader sub-thread can exit).
        self._server.stop(grace=1.0)

        # Release the IPC mapping (if we ever got one). Idempotent.
        with self.lock:
            if self._ipc_dev_ptr is not None:
                try:
                    cuda_runtime.ipcCloseMemHandle(self._ipc_dev_ptr)
                except Exception:
                    pass
                self._ipc_dev_ptr = None
                self.ringbuf = None

    # ---- Internals ----

    def _reader_main(self, request_iterator, context):
        try:
            for msg in request_iterator:
                kind = msg.WhichOneof("kind")
                if kind == "ipc_handle":
                    # Invariant: exactly one IpcHandle, and it must be
                    # the first message.
                    with self.lock:
                        if self._handle_seen:
                            raise RuntimeError(
                                "ToyGrouper: producer sent a second IpcHandle")
                        self._handle_seen = True
                    self._init_ringbuf(msg.ipc_handle)
                elif kind == "produced":
                    n = msg.produced.slot
                    with self.cv:
                        if self.is_stopped:
                            return
                        if not self._handle_seen:
                            raise RuntimeError(
                                "ToyGrouper: PRODUCED arrived before IpcHandle")
                        if n != self.rb_end:
                            raise RuntimeError(
                                f"ToyGrouper: out-of-order PRODUCED slot={n}, "
                                f"expected {self.rb_end}")
                        if n + 1 > self.rb_start + 5:
                            raise RuntimeError(
                                f"ToyGrouper: PRODUCED slot={n} would exceed "
                                f"capacity (rb_start={self.rb_start}, "
                                f"rb_end={self.rb_end})")
                        self.rb_end = n + 1
                        self.cv.notify_all()
                else:
                    raise RuntimeError(
                        f"ToyGrouper: unexpected ProducerMessage kind={kind!r}")
        except grpc.RpcError:
            # Stream cancelled (our own cancel during stop, or remote
            # disconnect). Exit quietly.
            return
        except Exception as e:
            # Invariant violation or other bug. Latch the error so that
            # the next receive() re-raises, then wake everyone blocked
            # on cv / _handle_ready / _consumed_q.
            with self.cv:
                if self.error is None:
                    self.error = e
                self.is_stopped = True
                self.cv.notify_all()
            self._handle_ready.set()
            self._consumed_q.put(None)
            # Print to stderr to match the C++ side's worker-thread
            # convention; don't re-raise (the gRPC handler would log
            # it again, and we already latched the error).
            import sys
            print(f"ToyGrouper: reader thread terminated with exception: {e}",
                  file=sys.stderr)

    def _init_ringbuf(self, ipc_handle_msg):
        # The reader thread might not have run cupy on this thread
        # before -- set the CUDA device explicitly. cudaIpcOpenMemHandle
        # is per-thread, so this matters.
        cupy.cuda.Device(self.cuda_device_id).use()

        handle_bytes = ipc_handle_msg.handle
        if len(handle_bytes) != 64:
            raise RuntimeError(
                f"ToyGrouper: bad IPC handle length {len(handle_bytes)} "
                f"(expected 64)")
        if ipc_handle_msg.device_id != self.cuda_device_id:
            raise RuntimeError(
                f"ToyGrouper: device mismatch "
                f"(producer={ipc_handle_msg.device_id}, "
                f"consumer={self.cuda_device_id})")

        # cudaIpcOpenMemHandle -> device pointer (int). Wrap as cupy
        # UnownedMemory: cupy will not try to free the allocation
        # (the C++ producer owns it); we call ipcCloseMemHandle in
        # stop() to release our mapping.
        device_ptr = cuda_runtime.ipcOpenMemHandle(handle_bytes)
        mem = cupy.cuda.UnownedMemory(device_ptr, _RINGBUF_NBYTES, owner=self)
        memptr = cupy.cuda.MemoryPointer(mem, 0)
        ringbuf = cupy.ndarray(shape=(5, 2), dtype=cupy.float32, memptr=memptr)

        with self.cv:
            self._ipc_dev_ptr = device_ptr
            self.ringbuf = ringbuf
            self._handle_ready.set()
            self.cv.notify_all()

    def _raise_if_stopped(self, method_name):
        # Caller must hold self.lock (or be inside `with self.cv`).
        if self.is_stopped:
            if self.error is not None:
                raise self.error
            raise RuntimeError(
                f"ToyGrouper.{method_name}() called on stopped instance")

    # ---- Context-manager glue ----

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
