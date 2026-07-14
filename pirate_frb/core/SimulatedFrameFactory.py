"""SimulatedFrameFactory method injections (+ re-export of the pybind11 class)."""

import numpy as np
import ksgpu
from ..pirate_pybind11 import SimulatedFrameFactory


@ksgpu.inject_methods(SimulatedFrameFactory)
class SimulatedFrameFactoryInjections:
    # No class docstring here: SimulatedFrameFactory's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector only adds pop_events().

    def pop_events(self, chunk_fpga_start, chunk_fpga_end):
        """Return the FRB-injection events recorded since the last call, as an FrbSifterEvents.

        Wraps the C++ _pop_events() (which returns a list of SimulatedFrameFactoryEvent and clears
        the internal list, so each event is returned exactly once), converting it to a
        pirate_frb.rpc.FrbSifterEvents with rfi_probs = 0 and the given FPGA-counter window.

        Parameters
        ----------
        chunk_fpga_start, chunk_fpga_end : int
            Absolute FPGA-counter window (start/end of the time chunk) for this batch of events.

        Returns
        -------
        pirate_frb.rpc.FrbSifterEvents
            One entry per injected FRB (rfi_prob = 0). Empty (length-0 arrays) if no FRBs were
            injected since the last call.
        """

        # Lazy import to avoid an import cycle (rpc -> ... at module-load time).
        from ..rpc import FrbSifterEvents

        # One awkward aspect of the current code: we have two very similar ways to represent an event
        # list, either a python FrbSifterEvents object, or a C++ vector<SimulatedFrameFactory::Event>.
        # This function converts between the two, at the "python/C++ boundary".
        #
        # This design is awkward but I think it's the least bad option. (The only real alternative seems
        # to be rewriting the FrbSifterClient in C++, and I prefer the current python implementation.)

        events = self._pop_events()
        n = len(events)
        return FrbSifterEvents(
            beam_ids             = np.array([e.beam_id             for e in events], dtype=np.int32),
            fpga_timestamps      = np.array([e.fpga_timestamp      for e in events], dtype=np.int64),
            dms                  = np.array([e.dm                  for e in events], dtype=np.float32),
            snrs                 = np.array([e.snr                 for e in events], dtype=np.float32),
            rfi_probs            = np.zeros(n, dtype=np.float32),
            widths_ms            = np.array([e.width_ms            for e in events], dtype=np.float32),
            subband_freqs_lo_MHz = np.array([e.subband_freq_lo_MHz for e in events], dtype=np.float32),
            subband_freqs_hi_MHz = np.array([e.subband_freq_hi_MHz for e in events], dtype=np.float32),
            chunk_fpga_start     = chunk_fpga_start,
            chunk_fpga_end       = chunk_fpga_end)
