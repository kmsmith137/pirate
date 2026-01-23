# Unit tests for pirate_frb

from .test_assembled_frame_allocator import test_assembled_frame_allocator
from ..core import AssembledFrame

def test_assembled_frame_asdf():
    """Test AssembledFrame ASDF file I/O."""
    AssembledFrame.test_asdf()
