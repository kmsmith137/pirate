"""
Toy FrbGrouper consumer(s): per-chunk peak SNR + argmax, optionally reported to a sifter.

Note that a streamlined version of _run_toy_grouper() is cut-and-pasted into the sphinx
docs (in notes/grouper_interface.md), so changes made here should be reflected there.
"""

import contextlib
import itertools
import time


def _run_toy_grouper(grouper, sifter=None, delay=0.0, do_histogram=False):
    """
    Main grouper loop (factored out of run_toy_grouper to reduce nesting).

    In the toy grouper, we don't do peak-finding or thresholding. We just send one
    event per time chunk, corresponding to the (beam,dm,time) triple with highest
    snr (even if the snr doesn't correspond to a statistically significant event!)

    If 'delay' > 0, sleep that many seconds at the end of each chunk -- an
    artificial slowdown for testing how the producer behaves when the consumer
    lags.
    """
    import cupy as cp

    nbeams_tot = grouper.total_beams
    nbatches = grouper.nbatches
    beams_per_batch = nbeams_tot // nbatches

    if sifter is not None:
        # Send the ConfigMessage to the sifter (this is only done once).
        beam_set_id = grouper.xengine_yaml['beamset']

        sifter.send_configuration(
            pirate_yaml = grouper.dedispersion_config_yaml_string,
            xengine_yaml = grouper.xengine_metadata_yaml_string,
            dedispersion_plan_yaml = grouper.dedispersion_plan_yaml_string,
            grouper_yaml = {'toy_grouper': True},  # placeholder for future expansion
            search_ip_addr = grouper.search_ip_addr)
        
        print(f'{grouper.grouper_ip_addr}: connected to sifter at {sifter.server_address}, '
              f'sent ConfigMessage', flush=True)

    if do_histogram:
        lo,hi = -10, +100
        nbins = int((hi - lo)/0.1)
        grouper.g_histogram = cp.zeros(nbins, int)
        grouper.g_histogram_bins = cp.linspace(lo, hi, nbins+1)
        grouper.g_histogram_bins[0]  = -1e6
        grouper.g_histogram_bins[-1] = +1e6

    # The grouper receives dedispersion outputs as an outer loop over time chunks,
    # followed by an inner loop over beam batches. Dedispersion outputs are arrays
    # (beams_per_batch, coarse_ndm, coarse_ntime). (One array for each dedispersion tree.)

    for ichunk in itertools.count():  # outer loop over time chunks
        # GPU-resident accumulators, updated entirely on the GPU below.
        gmax    = cp.full((), -cp.inf, dtype=cp.float32)   # max(snr)
        g_tree  = cp.full((), -1, dtype=cp.int64)          # argmax(tree index)
        g_ibeam = cp.full((), -1, dtype=cp.int64)          # argmax(beam index)
        g_idm   = cp.full((), -1, dtype=cp.int64)          # argmax(dm index)
        g_itime = cp.full((), -1, dtype=cp.int64)          # argmax(time index)
        per_beam_max = cp.full((nbeams_tot,), -cp.inf, dtype=cp.float32)

        for ibatch in range(nbatches):          # inner loop over beam batches
            beam0 = ibatch * beams_per_batch    # global index of this batch's first beam

            # Inside the inner context manager, you can use the 'output' object
            # (of type GpuDedisperserOutputs) to get the (out_max, out_argmax) arrays.
            #
            # WARNING: these arrays are only valid inside the context manager!
            # (They are "raw" pointers to GPU memory, which will be reused
            # when the context manager exits, even if you keep references.)

            with grouper.get_output(ichunk, ibatch) as outputs:
                # Loop over dedispersion trees.
                # 'tree_out' has shape (beams_per_batch, coarse_ndm, coarse_ntime).
                for itree, tree_out in enumerate(outputs.out_max):
                    # Per-beam max (over dm,time) for this batch's beams.
                    sl = per_beam_max[beam0 : beam0 + beams_per_batch]
                    cp.maximum(sl, tree_out.max(axis=(1, 2)), out=sl)

                    # Global peak + argmax candidate from this tree.
                    # Done entirely on GPU; no gpu<->host copies in sight!
                    m = tree_out.max()
                    lb, idm, itime = cp.unravel_index(tree_out.argmax(), tree_out.shape)
                    upd = m > gmax
                    gmax    = cp.where(upd, m, gmax)
                    g_tree  = cp.where(upd, itree, g_tree)
                    g_ibeam = cp.where(upd, beam0 + lb, g_ibeam)
                    g_idm   = cp.where(upd, idm, g_idm)
                    g_itime = cp.where(upd, itime, g_itime)

                    if do_histogram:
                        # SNR histogram
                        h,_ = cp.histogram(tree_out.ravel(), bins=grouper.g_histogram_bins)
                        grouper.g_histogram += h

        # Now we have the (snr, itree, ibeam, idm, itime) of a single "event".
        # We copy this data from the GPU to the host, and do a little math to convert
        # to "physical" quantities such as DM and arrival time. The math is explained
        # in notes/tree_dedispersion.tex, and is implemented in a helper method
        # grouper.create_events(), which returns an FrbSifterEvents object.

        events = grouper.create_events(ichunk, g_tree, g_ibeam, g_idm, g_itime, gmax)

        print(f'{grouper.grouper_ip_addr}: {ichunk=}: max snr={float(events.snrs):.3g}', flush=True)

        if sifter is not None:
            # Send the FrbEventsMessage to the sifter.
            sifter.send_events(
                beam_set_id = beam_set_id,
                events = events,
                coarsegrain_snr = per_beam_max)

        if delay > 0:
            time.sleep(delay)


def run_toy_grouper(grouper_addr, sifter_addr=None, delay=0.0, histogram=None):
    """Run a toy FrbGrouper consumer at 'grouper_addr' (e.g. '127.0.0.1:7000').

    Acts as the downstream consumer of an FrbServer producer over CUDA IPC.
    Blocks (in FrbGrouper.open(), via __enter__) until the producer connects, then
    per chunk finds the peak SNR event (and the per-beam max) and prints it; if
    'sifter_addr' is given, also reports to that FrbSifter. Runs until the producer
    disconnects or Ctrl-C.

    'sifter_addr' is an 'ip:port' string, or None to run without a sifter.
    'delay' (seconds) inserts an artificial per-chunk slowdown into the loop;
    see _run_toy_grouper.
    """
    # Imported here (not at module top) so 'import pirate_frb' stays light.
    from .rpc import FrbGrouper, FrbSifterClient

    # Construct the sifter client (opens a gRPC channel; the RPCs themselves are
    # issued in _run_toy_grouper, which has the grouper metadata). It's used as a
    # context manager so it's closed on every exit path; with no sifter,
    # nullcontext() yields None (which _run_toy_grouper accepts).
    sifter_cm = FrbSifterClient(sifter_addr) if (sifter_addr is not None) else contextlib.nullcontext()

    # FrbGrouper.__enter__ blocks until the producer connects, then pins this thread
    # to the GPU's vcpus and selects the CUDA device (printing a message); __exit__
    # restores them and closes the grouper. The sifter (if any) is closed when its
    # 'with' exits (after the grouper's).
    with sifter_cm as sifter, FrbGrouper(grouper_addr) as grouper:
        try:
            _run_toy_grouper(grouper, sifter, delay, do_histogram=(histogram is not None))
        except KeyboardInterrupt:
            print(f'{grouper_addr}: interrupted; shutting down', flush=True)
        except RuntimeError as e:
            # Most likely the producer disconnected (the grouper stops and
            # _acquire_output rethrows). Report cleanly; re-raise anything that is
            # not a stop (e.g. a genuine usage/assert bug).
            if grouper.is_stopped:
                print(f'{grouper_addr}: producer disconnected ({e}); exiting', flush=True)
            else:
                raise
        finally:
            if histogram:
                print('Grouper finally: writing histogram file' + histogram)
                with open(histogram, 'wb') as f:
                    import pickle
                    print('Writing', histogram)
                    pickle.dump(dict(histogram=grouper.g_histogram.get(),
                                     histogram_bins=grouper.g_histogram_bins.get()), f)
        # FrbGrouper.__exit__ restores affinity/device + closes on every path.
