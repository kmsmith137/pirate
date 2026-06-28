"""
FrbGrouper method injections (context-manager usage + get_output).

Split out from pirate_frb/pybind11_injections.py and kept here, alongside the
RPC clients, because FrbGrouper is the consumer side of the RPC interface.
Applied as a side effect of importing pirate_frb.rpc.
"""

from contextlib import contextmanager, ExitStack

import ksgpu
from .. import pirate_pybind11


@ksgpu.inject_methods(pirate_pybind11.FrbGrouper)
class FrbGrouperInjections:
    """Python extensions for FrbGrouper (context-manager usage + get_output)."""

    def __enter__(self):
        import cupy as cp
        from ..Hardware import Hardware
        from ..utils import ThreadAffinity

        # Blocks until the client connects and the handshake is processed.
        self.open()

        # YAML::Node is not pybind-wrapped; convert the wire string to a Python
        # object and attach it (py::dynamic_attr() on the C++ class enables this).
        import yaml
        self.dedispersion_plan_yaml = yaml.safe_load(self.dedispersion_plan_yaml_string)

        # The IPC-mapped output arrays live on cuda_device_id (known after the
        # handshake). For the duration of the 'with' block, pin this thread to
        # the vcpus local to that GPU and select the device, so the consumer's
        # cupy work runs on the right device with good CPU locality. Both are
        # entered via an ExitStack and undone in __exit__.
        vcpu_list = Hardware().vcpu_list_from_gpu(self.cuda_device_id)
        print(f"FrbGrouper: pinning thread to vcpu_list={vcpu_list} and selecting "
              f"cuda_device_id={self.cuda_device_id}", flush=True)
        self._exit_stack = ExitStack()
        self._exit_stack.enter_context(ThreadAffinity(vcpu_list))
        self._exit_stack.enter_context(cp.cuda.Device(self.cuda_device_id))
        return self

    def __exit__(self, *exc):
        # Undo the ThreadAffinity / cuda.Device contexts entered in __enter__,
        # then close the grouper. try/finally so close() always runs.
        try:
            es = getattr(self, "_exit_stack", None)
            if es is not None:
                self._exit_stack = None
                es.close()
        finally:
            self.close()
        return False

    @contextmanager
    def get_output(self, seq_id):
        """Acquire one batch's outputs; on exit synchronize the GPU, then release.

        On exit this calls ``cupy.cuda.get_current_stream().synchronize()``
        BEFORE ``release_output(seq_id)``, so all GPU reads the body queued on
        the current cupy stream complete before CONSUMED is sent to the
        producer. This is required because there is no IPC-event fence: once
        CONSUMED is sent, the producer may overwrite the ring-buffer slot (see
        plans/grouper_server.md). The body must therefore do its GPU work on the
        current cupy stream (the default; FrbGrouper's __enter__ has already
        selected the right device).

        Yields
        ------
        _GpuDedisperserOutputs
            Per-batch slice with .out_max / .out_argmax (lists of ksgpu Arrays,
            convertible to cupy via DLPack).
        """
        import cupy as cp
        outputs = self.acquire_output(seq_id)
        try:
            yield outputs
        finally:
            cp.cuda.get_current_stream().synchronize()
            self.release_output(seq_id)


