# Unit tests for pirate_frb

# Not a test -- a diagnostic, reached as 'pirate_frb dev coverage'. See its module docstring.
from .coverage import report_coverage

from .test_assembled_frame_allocator import test_assembled_frame_allocator
from .test_assembled_frame_asdf import test_assembled_frame_asdf
from .test_atomic_out import test_atomic_out
from .test_decode_argmax import test_decode_argmax
from .test_dedispersion_config import test_max_width_monotone, test_random_args_flags
from .test_network import test_network, test_slow_subscriber
from .test_pulse_injection import test_pulse_injection, test_pulse_invariants
from .test_server import test_server
from .test_subbands import test_frequency_subbands_parity, test_subband_property

