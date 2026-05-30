# pirate_frb.rpc - RPC client/server classes

from .FrbClient import FrbClient
from ..pirate_pybind11 import FrbGrouper   # injections applied at pirate_frb import

__all__ = ["FrbClient", "FrbGrouper"]
