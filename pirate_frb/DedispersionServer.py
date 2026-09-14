"""Native dedispersion server for the four-terminal live workflow."""
from pathlib import Path
import time


class DedispersionServer:
    """One real FrbServer with finite, separately sized raw and dedispersion pools."""

    def __init__(self, bundle, data_address, rpc_address,
                 grouper_address, cuda_device_id):
        self.server = None
        self.file_writer = None
        self.allocator = None
        self.grouper_client = None
        self.receivers = []
        self.objects = []
        try:
            self._build(bundle, data_address, rpc_address,
                        grouper_address, cuda_device_id)
        except BaseException as exc:
            # Construction may have started writer threads before a subsequent
            # allocation or Receiver/FrbServer constructor failed.
            errors = self._stop_resources()
            if errors:
                exc.add_note("Partial server construction cleanup: " + "; ".join(errors))
                exc.cleanup_errors = errors
            raise

    def _build(self, bundle, data_address, rpc_address,
               grouper_address, cuda_device_id):
        import cupy as cp
        import ksgpu
        from .core import (AssembledFrameAllocator, BumpAllocator, CudaStreamPool,
                           FileWriter, Receiver, SlabAllocator)
        from .pirate_pybind11 import (DedispersionConfig, DedispersionPlan,
                                     FrbGrouperClient, FrbServer, GpuDedisperser)
        config = DedispersionConfig.from_yaml_string(bundle["dedispersion_config_yaml"])
        md = bundle["metadata"]
        config.zone_nfreq = list(md["zone_nfreq"])
        config.zone_freq_edges = list(md["zone_freq_edges"])
        config.time_sample_ms = bundle["time_sample_ms"]
        config.beams_per_gpu = len(bundle["beam_ids"])
        config.validate()
        nbeam, nfreq, ntime = len(bundle["beam_ids"]), sum(md["zone_nfreq"]), bundle["samples_per_chunk"]
        nstream, batch = config.num_active_batches, config.beams_per_batch
        chunk_seconds = ntime * bundle["time_sample_ms"] * 1e-3
        ring_chunks = 7
        with cp.cuda.Device(cuda_device_id):
            plan = DedispersionPlan(config)
            streams = CudaStreamPool(nstream)
            probe = GpuDedisperser(plan, streams, cuda_device_id=cuda_device_id,
                                   num_consumers=1, nbatches_out=2*nstream, nbatches_wt=nstream)
            gpu_dedispersion_bytes = probe.resource_tracker.get_gmem_footprint()
            host_dedispersion_bytes = probe.resource_tracker.get_hmem_footprint()
            scratch = nstream * batch * nfreq * (ntime // 2 + (ntime // 256) * 4)
            gpu_bytes = int(gpu_dedispersion_bytes + scratch + (1 << 16))
            host_bytes = int(host_dedispersion_bytes + (1 << 16))
            slab_bytes = int(AssembledFrameAllocator.slab_nbytes(nfreq, ntime))
            # Separate pools make dedispersion allocation independent of raw
            # retention. Extra slabs cover assembly/allocator/writer queues.
            raw_bytes = slab_bytes * nbeam * (ring_chunks + 8) + (1 << 16)
            raw_bump = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, raw_bytes,
                                      cuda_device=cuda_device_id)
            self.objects.append(raw_bump)
            raw_slabs = SlabAllocator(raw_bump)
            self.objects.append(raw_slabs)
            self.allocator = AssembledFrameAllocator(raw_slabs, num_consumers=1,
                                                       time_samples_per_chunk=ntime)
            host_bump = BumpAllocator(ksgpu.af_rhost | ksgpu.af_zero, host_bytes,
                                       cuda_device=cuda_device_id)
            self.objects.append(host_bump)
            gpu_bump = BumpAllocator(ksgpu.af_gpu | ksgpu.af_zero, gpu_bytes,
                                      cuda_device=cuda_device_id)
            self.objects.append(gpu_bump)
            # FrbServer requires an idle writer even for diagnostics. Existing
            # roots avoid creating directories; live mode never requests writes.
            ssd_root = Path.home()
            nfs_root = Path.home()
            self.file_writer = FileWriter(str(ssd_root), str(nfs_root),
                                          num_ssd_threads=1, num_nfs_threads=1)
            self.receivers = [Receiver(address=data_address, allocator=self.allocator)]
            self.grouper_client = FrbGrouperClient(grouper_address)
            self.server = FrbServer(config, self.receivers, self.file_writer,
                                    rpc_address, ring_chunks, min_data_mtu=1500,
                                    host_allocator=host_bump, gpu_allocator=gpu_bump,
                                    cuda_device_id=cuda_device_id,
                                    grouper_client=self.grouper_client,
                                    nbatches_wt=nstream, quiet=True,
                                    max_unprocessed_chunks=5)
        self.memory = dict(raw_pool_bytes=raw_bytes, raw_slab_bytes=slab_bytes,
                           host_dedispersion_pool_bytes=host_bytes, gpu_pool_bytes=gpu_bytes,
                           ringbuf_nchunks=ring_chunks,
                           nominal_retention_seconds=ring_chunks * chunk_seconds,
                           raw_headroom_chunks=8, shared_host_allocator=False)

    def start(self, timeout=600):
        self.grouper_client.ping(timeout_ms=int(timeout * 1000))
        self.server.start()
        deadline = time.monotonic() + timeout
        for receiver in self.receivers:
            while not receiver.wait_until_listening(timeout_sec=0.25):
                self.server.poll_from_python(timeout_ms=0)
                if time.monotonic() > deadline:
                    raise TimeoutError("receiver did not begin listening")

    def _stop_resources(self):
        errors = []
        actions = [("server.stop", self.server)]
        actions.extend((f"receiver[{i}].stop", receiver)
                       for i, receiver in enumerate(self.receivers))
        actions.extend((("allocator.stop", self.allocator),
                        ("file_writer.stop", self.file_writer)))
        for name, resource in actions:
            if resource is None:
                continue
            try:
                resource.stop()
            except BaseException as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return errors

    def stop(self):
        errors = self._stop_resources()
        if errors:
            exc = RuntimeError("server resource cleanup failed: " + "; ".join(errors))
            exc.cleanup_errors = errors
            raise exc
