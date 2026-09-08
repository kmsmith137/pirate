# Unit tests for pirate_frb

import importlib
import inspect

from .test_assembled_frame_allocator import test_assembled_frame_allocator
from .test_assembled_frame_asdf import test_assembled_frame_asdf
from .test_atomic_out import test_atomic_out
from .test_decode_argmax import test_decode_argmax
from .test_network import test_network, test_slow_subscriber
from .test_packaging import test_offline_peak_modules_packaged
from .test_pulse_injection import test_multi_burst_cli, test_pulse_injection, test_pulse_invariants
from .test_server import test_server


def test_offline_peak_milestone(cuda_device_id=0):
    """Run the pytest-free offline extraction/grouping/catalog regressions.

    PIRATE's historical ``python -m pirate_frb test`` runner invokes ordinary
    assertion-based functions directly.  Discovering the focused functions here
    keeps that runner in sync with the same modules collected by pytest, without
    making pytest a runtime dependency of an installed wheel.
    """
    # Import submodules by fully qualified name so discovery is unambiguous.
    decoder_tests = importlib.import_module(
        f"{__name__}.test_gpu_argmax_decoder"
    )
    offline_tests = importlib.import_module(
        f"{__name__}.test_offline_grouper"
    )
    streaming_tests = importlib.import_module(
        f"{__name__}.test_offline_grouper_streaming"
    )
    config_tests = importlib.import_module(
        f"{__name__}.test_offline_grouper_config"
    )
    grouping_tests = importlib.import_module(
        f"{__name__}.test_offline_candidate_grouper"
    )
    peakfinder_tests = importlib.import_module(
        f"{__name__}.test_peakfinders_stream"
    )
    catalog_tests = importlib.import_module(
        f"{__name__}.test_trigger_catalog"
    )

    # Packaging is a host-only contract check; the modules below contain the
    # focused GPU and offline integration cases. Sorting by name
    # gives reproducible output/failure order in the custom runner.
    test_offline_peak_modules_packaged()
    for module in (
            config_tests,
            peakfinder_tests,
            decoder_tests,
            grouping_tests,
            offline_tests,
            streaming_tests,
            catalog_tests):
        for name in sorted(vars(module)):
            function = getattr(module, name)
            if not name.startswith("test_") or not callable(function):
                continue
            signature = inspect.signature(function)
            required = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if required:
                raise TypeError(
                    f"{module.__name__}.{name} requires pytest-style arguments; "
                    "focused milestone tests must remain directly callable"
                )
            if "cuda_device_id" in signature.parameters:
                function(cuda_device_id=cuda_device_id)
            else:
                function()
