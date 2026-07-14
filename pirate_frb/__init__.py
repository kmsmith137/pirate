# Note: 'import ksgpu' must precede 'import pirate_pybind11'.
# (This is because 'import ksgpu' pulls in the libraries ksgpu.so and ksgpu_pybind11...so,
# using the "ctypes trick" to make their symbols globally visible.)
import ksgpu

# Fail loudly if 'ksgpu' was shadowed by a same-named directory on sys.path
# instead of the installed package. This happens when Python runs from a
# directory that contains a 'ksgpu/' subdir (e.g. the workspace root, where the
# ksgpu repo is checked out): 'import ksgpu' then picks up that directory as an
# empty PEP 420 namespace package, so ksgpu's __init__.py -- and its ctypes
# RTLD_GLOBAL trick that publishes ksgpu's C++ symbols -- never runs, and the
# 'pirate_pybind11' import below would otherwise fail with a cryptic
# "undefined symbol: ksgpu::...".
if getattr(ksgpu, "__file__", None) is None or not hasattr(ksgpu, "Dtype"):
    raise ImportError(
        "'ksgpu' was imported as an empty namespace package "
        f"(ksgpu.__file__ = {getattr(ksgpu, '__file__', None)!r}), not the "
        "installed package. Python most likely picked up a 'ksgpu/' directory "
        "from the current working directory (sys.path[0]) -- this happens when "
        "you run from the workspace root. Run from a different directory, or "
        "set PYTHONSAFEPATH=1 in the environment."
    )

from . import pirate_pybind11
from . import cuda_generator
from . import casm
from . import kernels
from . import loose_ends
from . import core
from . import slow_avar
from . import rpc  # noqa: F401  (RPC client/server classes; also applies FrbGrouper injections)

# Method injections for pybind11 classes live in per-class modules (e.g.
# core/BumpAllocator.py, casm/CasmBeamformer.py, kernels/GpuDequantizationKernel.py,
# rpc/FrbGrouper.py, and DedispersionConfig/GpuDedisperser below); importing each
# such module applies its injections. The subpackage imports above cover most of
# them; see notes/pybind11.md.

from .HwtestSender import HwtestSender
from .Hwtest import Hwtest
from .Hardware import Hardware
from .run_server import run_server
from .run_toy_grouper import run_toy_grouper
from .Acquisition import Acquisition
from .OfflineDedisperser import OfflineDedisperser
from .run_offline_dedisperser import run_offline_dedisperser

# DedispersionConfig and GpuDedisperser carry method injections, so they are
# imported from their per-class modules (which apply the injections and re-export
# the class); the rest are plain re-exports of the pybind11 classes.
from .DedispersionConfig import DedispersionConfig
from .GpuDedisperser import GpuDedisperser
from .pirate_pybind11 import (
    DedispersionPlan,
    FrbServer,
    FrbGrouperClient,
    constants,
)
