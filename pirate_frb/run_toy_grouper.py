"""
Toy FrbGrouper consumer(s): per-chunk peak SNR + argmax, optionally reported to a sifter.

Note that a streamlined version of _run_toy_grouper() is cut-and-pasted into the sphinx
docs (in notes/grouper_interface.md), so changes made here should be reflected there.
Deliberate exception: the mirror captures "essentials" only, so optional bells and
whistles (e.g. the --histogram feature) are NOT mirrored, and changes to them don't
need to be reflected in the sphinx docs.
"""

import contextlib
import itertools
import time


def _run_toy_grouper(grouper, sifter=None, delay=0.0, snr_threshold=10.0, do_histogram=False):
    """
    Main grouper loop (factored out of run_toy_grouper to reduce nesting).

    The toy grouper does a minimal thresholding: for each time chunk it emits one
    event per beam whose peak SNR (over the tree/dm/time it searched) exceeds
    snr_threshold -- so a chunk produces between 0 and nbeams events. Even when a
    chunk has no beam above threshold, the EventMessage is still sent (it carries
    coarsegrain_snr). coarsegrain_snr is unaffected by the threshold: it is always
    the per-beam max SNR over the whole chunk.

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
        # Per-beam accumulators over (tree, dm, time): the peak SNR, its argmax, and the
        # winning out_argmax token. All GPU-resident, updated entirely on the GPU below.
        per_beam_max   = cp.full((nbeams_tot,), -cp.inf, dtype=cp.float32)  # peak snr
        per_beam_tree  = cp.full((nbeams_tot,), -1, dtype=cp.int64)         # argmax(tree index)
        per_beam_idm   = cp.full((nbeams_tot,), -1, dtype=cp.int64)         # argmax(dm index)
        per_beam_itime = cp.full((nbeams_tot,), -1, dtype=cp.int64)         # argmax(time index)
        per_beam_token = cp.zeros((nbeams_tot,), dtype=cp.uint32)           # out_argmax token at the argmax

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
                    # Per-beam max SNR + its (dm, time) argmax for this batch's beams.
                    # Done entirely on GPU; no gpu<->host copies in sight!
                    bpb, ndm, nt = tree_out.shape
                    flat = tree_out.reshape(bpb, ndm * nt)
                    beam_max = flat.max(axis=1)
                    beam_arg = flat.argmax(axis=1)
                    beam_idm, beam_itime = beam_arg // nt, beam_arg % nt

                    # Winning out_argmax token, gathered at the same argmax position.
                    # (Must happen here, inside the context manager, while out_argmax
                    # is valid -- create_events() decodes the tokens later.)
                    flat_arg = outputs.out_argmax[itree].reshape(bpb, ndm * nt)
                    beam_tok = flat_arg[cp.arange(bpb), beam_arg]

                    if do_histogram:
                        # SNR histogram
                        h,_ = cp.histogram(tree_out.ravel(), bins=grouper.g_histogram_bins)
                        grouper.g_histogram += h

                    sl = slice(beam0, beam0 + bpb)
                    upd = beam_max > per_beam_max[sl]
                    per_beam_max[sl]   = cp.where(upd, beam_max,   per_beam_max[sl])
                    per_beam_tree[sl]  = cp.where(upd, itree,      per_beam_tree[sl])
                    per_beam_idm[sl]   = cp.where(upd, beam_idm,   per_beam_idm[sl])
                    per_beam_itime[sl] = cp.where(upd, beam_itime, per_beam_itime[sl])
                    per_beam_token[sl] = cp.where(upd, beam_tok,   per_beam_token[sl])

        # Now we have one event per beam whose peak SNR exceeds threshold (0 <= nevents <= nbeams).
        # Events are identified by (snr, itree, ibeam, idm, itime, token) on the GPU. We copy this
        # data from the GPU to the CPU, and convert to "physical" quantities (DM, fpga_timestamp,
        # width, frequency subband) by decoding the out_argmax tokens. The math is explained in
        # notes/tree_dedispersion.tex, and is implemented in a helper method
        # grouper.create_events(), which returns an FrbSifterEvents.

        ibeam = cp.nonzero(per_beam_max > snr_threshold)[0]   # global indices of above-threshold beams
        events = grouper.create_events(ichunk, per_beam_tree[ibeam], ibeam,
                                       per_beam_idm[ibeam], per_beam_itime[ibeam],
                                       per_beam_max[ibeam], per_beam_token[ibeam])

        coarse_snr_max = float(per_beam_max.max())
        print(f'toy_grouper: beamset={grouper.xengine_yaml["beamset"]}, ichunk={ichunk}, '
              f'fpga=[{events.chunk_fpga_start}:{events.chunk_fpga_end}], '
              f'coarse_snr_max={coarse_snr_max:.4g}, nevents={len(events)}', flush=True)

        if sifter is not None:
            # Send the FrbEventsMessage (even if nevents==0).
            sifter.send_events(
                beam_set_id = beam_set_id,
                events = events,
                coarsegrain_snr = per_beam_max)

        if delay > 0:
            time.sleep(delay)


def run_toy_grouper(grouper_addr, sifter_addr=None, delay=0.0, snr_threshold=10.0, histogram=None):
    """Run a toy FrbGrouper consumer at 'grouper_addr' (e.g. '127.0.0.1:7000').

    Acts as the downstream consumer of an FrbServer producer over CUDA IPC.
    Blocks (in FrbGrouper.open(), via __enter__) until the producer connects, then
    per chunk emits one event per beam whose peak SNR exceeds 'snr_threshold' (and
    reports the per-beam max); if 'sifter_addr' is given, also reports to that
    FrbSifter. Runs until the producer disconnects or Ctrl-C.

    'sifter_addr' is an 'ip:port' string, or None to run without a sifter.
    'delay' (seconds) inserts an artificial per-chunk slowdown into the loop.
    'snr_threshold' (default 10) is the per-beam event threshold; see
    _run_toy_grouper.
    'histogram' is a filename (or None): on termination, pickle a histogram of
    all out_max SNR values (accumulated over all trees/beams/chunks) to it.
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
            _run_toy_grouper(grouper, sifter, delay, snr_threshold, do_histogram=(histogram is not None))
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
