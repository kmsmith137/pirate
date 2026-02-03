"""FrbClient - Python client for FrbServer gRPC service."""

import grpc
from .grpc import frb_search_pb2
from .grpc import frb_search_pb2_grpc


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
            - rb_finalized: (Last finalized frame_id) + 1
            - rb_end: (Last frame_id in ring buffer) + 1
            - num_free_frames: Number of available frames in AssembledFrameAllocator
        """
        request = frb_search_pb2.GetStatusRequest()
        return self.stub.GetStatus(request)
    
    def get_metadata(self, verbose: bool = False) -> str:
        """Query the server for metadata as a YAML string.

        Args:
            verbose: If True, include comments explaining each field.

        Returns:
            YAML string representation of metadata, or empty string if
            metadata is not yet available.
        """
        request = frb_search_pb2.GetMetadataRequest(verbose=verbose)
        response = self.stub.GetMetadata(request)
        return response.yaml_string

    def write_chunks(
        self,
        beams: list[int],
        min_time_chunk_index: int,
        max_time_chunk_index: int,
        filename_pattern: str
    ) -> list[str]:
        """Request the server to write chunks to disk.

        Args:
            beams: List of beam IDs to write.
            min_time_chunk_index: First time chunk index (inclusive).
            max_time_chunk_index: Last time chunk index (inclusive).
            filename_pattern: Pattern with (BEAM) and (CHUNK) placeholders,
                e.g. "dir1/dir2/file_(BEAM)_(CHUNK).asdf"

        Returns:
            List of filenames that will be written.
        """
        request = frb_search_pb2.WriteChunksRequest(
            beams=beams,
            min_time_chunk_index=min_time_chunk_index,
            max_time_chunk_index=max_time_chunk_index,
            filename_pattern=filename_pattern
        )
        response = self.stub.WriteChunks(request)
        return list(response.filename_list)

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"FrbClient({self.server_address!r})"
