"""FrbSearchClient - Python client for FrbServer gRPC service."""

import os
import datetime

import grpc
from .grpc import frb_search_pb2
from .grpc import frb_search_pb2_grpc
from .FileSubscriber import FileSubscriber


class FrbSearchClient:
    """Client for querying FrbServer via gRPC.
    
    Usage:
        with FrbSearchClient("localhost:50051") as client:
            status = client.get_status()
            print(f"Connections: {status.num_connections}, Bytes: {status.num_bytes}")
    
    Or without context manager:
        client = FrbSearchClient("localhost:50051")
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
            - frequency_subband_counts: list[int] from config_prefilled (a real
              search-config value, not a fake_ default)
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
        fpga_seq_start: int,
        fpga_seq_end: int,
        filename_pattern: str
    ) -> list[str]:
        """Request the server to write files to disk.

        Args:
            beams: List of beam IDs to write.
            fpga_seq_start: Start of the fpga-seq range (inclusive).
            fpga_seq_end: End of the fpga-seq range (exclusive). Files are
                written for all chunks overlapping
                fpga_seq_start <= f < fpga_seq_end.
            filename_pattern: Pattern with (BEAM) and (CHUNK) placeholders,
                e.g. "dir1/dir2/file_(BEAM)_(CHUNK).asdf"

        Returns:
            List of filenames that will be written.
        """
        request = frb_search_pb2.WriteFilesRequest(
            beams=beams,
            fpga_seq_start=fpga_seq_start,
            fpga_seq_end=fpga_seq_end,
            filename_pattern=filename_pattern
        )
        response = self.stub.WriteFiles(request)
        return list(response.filename_list)

    def start_stream(
        self,
        beam_ids: list[int],
        acq_name: str = None,
        filename_pattern: str = None,
        fpga_seq_start: int = 0,
        fpga_seq_end: int = None
    ) -> tuple[str, str]:
        """Register a "stream": data matching (beam_ids x fpga-seq range) is
        queued for disk writing automatically as it flows through the server.

        Complements write_files() (one-shot, retroactive within the ring
        buffer): a stream captures each frame at the moment it is processed,
        so chunks that were already processed when StartStream arrives are
        NOT captured retroactively, even if fpga_seq_start is in the past.

        Args:
            beam_ids: Nonempty list of beam IDs (no all-beams convention;
                list beams explicitly -- show_streams() returns the full list).
            acq_name: Nonempty identifier, unique among active streams
                (used by show_streams() / cancel_stream()). If None (default),
                a name "{username}_{date}_{time}" is generated, e.g.
                "kmsmith_26_07_05_143052".
            filename_pattern: Pattern with (BEAM) and (CHUNK) placeholders,
                same format as write_files(). If None (default),
                "streams/{acq_name}/frame_b(BEAM)_t(CHUNK).asdf" is used.
            fpga_seq_start: Start of the fpga-seq range (inclusive);
                0 (default) means "start asap".
            fpga_seq_end: End of the fpga-seq range (exclusive). None
                (default) means "run indefinitely" (sent as 2**63 - 1 on
                the wire).

        Returns:
            (acq_name, filename_pattern): the resolved values that were sent
            (useful to the caller when either defaulted from None).

        Raises grpc.RpcError on validation failure (empty/duplicate acq_name,
        unknown beam_id, bad pattern, range entirely in the past, or server
        not yet initialized).
        """
        if acq_name is None:
            # "{username}_{date}_{time}", e.g. kmsmith_26_07_05_143052.
            # ($USER mirrors run_server's nfs_dir {user} interpolation.)
            user = os.environ.get('USER', 'unknown')
            acq_name = user + '_' + datetime.datetime.now().strftime('%y_%m_%d_%H%M%S')

        if filename_pattern is None:
            filename_pattern = f"streams/{acq_name}/frame_b(BEAM)_t(CHUNK).asdf"

        if fpga_seq_end is None:
            fpga_seq_end = 2**63 - 1   # "run indefinitely"

        request = frb_search_pb2.StartStreamRequest(
            acq_name=acq_name,
            filename_pattern=filename_pattern,
            beam_ids=beam_ids,
            fpga_seq_start=fpga_seq_start,
            fpga_seq_end=fpga_seq_end
        )
        self.stub.StartStream(request)
        return acq_name, filename_pattern

    def show_streams(self):
        """Query the server for its active streams.

        Returns:
            ShowStreamsResponse protobuf message with fields:
            - current_fpga_seq: the server's current position as an fpga seq,
              derived from rb_processed (all data before this fpga seq has
              been fully processed).
            - beam_ids: ALL beams processed by this server (not just those
              with active streams).
            - streams: list of StreamInfo, each with 'args' (the original
              StartStreamRequest, echoed back) and queued-so-far counters
              'num_files_queued' / 'num_bytes_queued' (bytes are logical:
              data arrays only, headers neglected, hardlinks counted at
              full size).
        """
        request = frb_search_pb2.ShowStreamsRequest()
        return self.stub.ShowStreams(request)

    def cancel_stream(self, acq_name: str = None, cancel_all: bool = False) -> int:
        """Cancel one active stream (by acq_name), or all of them.

        File writes already queued still complete (and still notify
        subscribe_files() subscribers); cancellation only stops future
        matching.

        Args:
            acq_name: Stream to cancel (ignored if cancel_all=True).
                Unknown acq_name raises grpc.RpcError.
            cancel_all: If True, cancel all active streams.

        Returns:
            Number of streams cancelled.
        """
        request = frb_search_pb2.CancelStreamRequest(
            cancel_all=cancel_all,
            acq_name=("" if acq_name is None else acq_name)
        )
        response = self.stub.CancelStream(request)
        return response.num_cancelled

    def subscribe_files(self, subscribe_streams: bool = False):
        """Open a file-write-notification subscription.

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
        return f"FrbSearchClient({self.server_address!r})"
