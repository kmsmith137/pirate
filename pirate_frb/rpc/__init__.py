# pirate_frb.rpc - RPC client/server classes

from .FrbSearchClient import FrbSearchClient
from ..pirate_pybind11 import FrbGrouper   # injections applied at pirate_frb import

__all__ = ["FrbSearchClient", "FrbGrouper"]
