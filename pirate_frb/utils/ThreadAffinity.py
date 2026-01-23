"""ThreadAffinity context manager for temporarily setting thread CPU affinity."""

from ..pirate_pybind11 import get_thread_affinity, set_thread_affinity


class ThreadAffinity:
    """Context manager for temporarily setting thread CPU affinity.
    
    On entry, sets the calling thread's affinity to the specified vCPUs.
    On exit, restores the original affinity.
    
    Example:
        with ThreadAffinity([2, 3]):
            # Thread is pinned to vCPUs 2 and 3
            do_work()
        # Original affinity is restored
    
    Args:
        vcpu_list: List of vCPU indices to pin the thread to.
    """
    
    def __init__(self, vcpu_list):
        self.vcpu_list = vcpu_list
        self.saved_affinity = None
    
    def __enter__(self):
        self.saved_affinity = get_thread_affinity()
        set_thread_affinity(self.vcpu_list)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_thread_affinity(self.saved_affinity)
        return False  # Don't suppress exceptions
