"""FrbClient - Python client for FrbServer gRPC service."""

import grpc
from .grpc import frb_search_pb2
from .grpc import frb_search_pb2_grpc


class FrbClient:
    """Client for querying FrbServer via gRPC.
    
    Usage:
        with FrbClient("localhost:50051") as client:
            num_conn, num_bytes = client.get_status()
            print(f"Connections: {num_conn}, Bytes: {num_bytes}")
    
    Or without context manager:
        client = FrbClient("localhost:50051")
        num_conn, num_bytes = client.get_status()
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
    
    def get_status(self) -> tuple[int, int]:
        """Query the server for current status.
        
        Returns:
            Tuple of (num_connections, num_bytes) summed over all receivers.
        """
        request = frb_search_pb2.GetStatusRequest()
        response = self.stub.GetStatus(request)
        return (response.num_connections, response.num_bytes)
    
    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"FrbClient({self.server_address!r})"
