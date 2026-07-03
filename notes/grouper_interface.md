# Writing a grouper

The next step after the FRB search (`pirate`) is the "grouper".
The grouper is a python program which receives dedispersion
outputs from `pirate`.
(The "dedispersion outputs" consist of a pair of 2-d arrays
`out_max`, `out_argmax` in the dm/time plane, for each dedispersion
"tree" and for each beam. See below.)

The purpose of the grouper is to identify peaks in the dm/time plane,
run a classifier on each peak, and send a top-k list of peaks ("events")
to the "sifter". The sifter is a process that runs on a central node,
that receives events from groupers on all search nodes.

We run one grouper per GPU, i.e. two groupers per search node.
The grouper exchanges data arrays with `pirate` via a shared
GPU memory ring buffer. This avoids the overhead of copying data
between GPU and CPU. The grouper exchanges **metadata** with `pirate`
via `grpc` over the loopback network (`127.0.0.1`).
For performance reasons, the grouper should leave arrays on the GPU
if possible, and process them with `cupy`.

 - {py:class}`~pirate_frb.pirate_pybind11.FrbGrouper`: for managing communication with pirate.
 - {py:class}`~pirate_frb.rpc.FrbSifterClient`: for managing communication with the sifter.
 - {py:class}`~pirate_frb.rpc.FrbSifterEvents`: helper class for sending events to the sifter.
 - {py:class}`~pirate_frb.core.GpuDedisperserOutputs`: helper class for storing dedispersion output arrays.
 - Protocol for pirate-grouper communication: [`grpc/frb_grouper.proto`](../grpc/frb_grouper.proto).
 - Protocol for grouper-sifter communication: [`grpc/frb_sifter.proto`](../grpc/frb_sifter.proto).

## Example code

From `pirate_frb/run_toy_grouper.py` (slightly streamlined):
```py
# In the toy grouper, we don't do peak-finding or thresholding. We just send one
# event per time chunk, corresponding to the (beam,dm,time) triple with highest
# snr (even if the snr doesn't correspond to a statistically significant event!)

# The FrbGrouper context manager blocks until 'pirate' connects and sends metadata.
# WARNING: don't touch the gpu (e.g. by allocating memory with cupy) until you enter
# the context manager. See 'FrbGrouper context manager' below

with FrbGrouper(grouper_addr) as grouper:
    sifter = FrbSifterClient(sifter_addr)
    
    # Send the ConfigMessage to the sifter (this is only done once).
    sifter.send_configuration(
        pirate_yaml = grouper.dedispersion_config_yaml_string,
        xengine_yaml = grouper.xengine_metadata_yaml_string,
        dedispersion_plan_yaml = grouper.dedispersion_plan_yaml_string,
        grouper_yaml = {'toy_grouper': True},  # placeholder for future expansion
        search_ip_addr = grouper.search_ip_addr)
        
    beam_set_id = grouper.xengine_yaml['beamset']
    nbeams_tot = grouper.total_beams
    nbatches = grouper.nbatches
    beams_per_batch = nbeams_tot // nbatches
    
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
```
  
## FrbGrouper context manager

When the `FrbGrouper` context manager is entered, a lot happens under the hood:

 - The grouper starts a grpc service, and blocks until `pirate` connects.
   Note that the grouper is the grpc server, and `pirate` is the client.
   
 - The grouper receives metadata (three yaml objects, see below) from `pirate`.

 - The grouper learns which GPU it is running on (by receiving a cuda `device_id`),
   and *changes the current cupy device* to the appropriate GPU. This change will
   be un-done when the context manager exits. Within the `FrbGrouper` context manager,
   cupy functions will use the correct GPU, but may revert to GPU 0 after the context
   manager exits.

   *Important consequence:* the grouper code must not touch the GPU outside
   (either before or after) the `FrbGrouper` context manager. Otherwise, you
   may use the wrong GPU (which will lead to either a crash, or a slowdown if
   data is copied between GPUs).

   (Similarly, the `FrbGrouper` context manager sets the CPU affinity of the
   python caller, to whichever CPU is appropriate.)

 - Based on received metadata (yaml files + a `cudaIpcMemHandle_t`), the grouper
   configures a GPU memory ring buffer which will be shared with `pirate`.

 - The grouper spawns background threads for sending/receiving short rpcs with
   `pirate`. These short rpcs are used to synchronize access to the shared ring
   buffer in GPU memory. (When pirate produces new data, it sends a message
   to the grouper, and vice versa when the grouper is finished consuming data.)
   For GIL reasons, these are C++ threads, not python threads.

The grouper receives three metadata objects from `pirate`:

 1. `xengine_metadata_yaml` -- the X-engine metadata
    ([example](../configs/xengine_metadata.yml)).

 2. `dedispersion_config_yaml` -- same format as on-disk dedispersion configs
    ([example](../configs/dedispersion/chord_sb2_et.yml)).

 3. `dedispersion_plan_yaml` -- more detailed dedispersion metadata, including array shapes
    ([example](../configs/example_dedispersion_plan.yml)).

Note: a `dedispersion_config` file is specified on the `pirate` command line
when `pirate` is started, and passed through to the grouper mostly unmodified
(item 2 above). However, the following members may be modified, if the real-time
values in the X-engine are different from the "static" values in the config file:
`zone_nfreq`, `zone_freq_edges`, `time_sample_ms`, `beams_per_gpu`.

## Data arrays and GpuDedisperserOutputs context manager

Dedispersion outputs are processed in the following loop structure
(see example code above):

 - Outer loop over time "chunks"
 
 - Middle loop over beam "batches". Call `FrbGrouper.get_output()` once per
   iteration of the middle loop, to get a {py:class}`~pirate_frb.core.GpuDedisperserOutputs`
   object.
   
 - Inner loop over dedispersion trees. For each tree, the dedispersion outputs are
   an array of shape `(beams_per_batch, coarse_ndm, coarse_ntime)`, i.e. a 2-d array
   in the dm-time plane, with a short beam spectator axis.

Here are some details that are not obvious from the example code:

 - Each batch must be fully processed by the grouper, before moving on to the
   next batch. This is because the dedispersion output array is only valid inside
   its `GpuDedisperserOutputs` context manager.

   If you must process multiple batches in parallel, or save a previous batch,
   then you'll need to copy data. (Don't save a reference to the original array,
   it will be overwritten as soon as you exit the context manager!)

   This is not a fundamental limitation -- it's just the easiest interface to implement.
   If it creates problems then let me know, and we can figure out an alternative.
 
 - Each tree searches a different DM range, and can have different levels of
   coarse-graining in the DM and time axes. These details are partially "hidden"
   by `FrbGrouper.create_events()`, which converts `(tree_index, dm_index, time_index)`
   tuples to DMs and arrival times.

   However, when implementing peak-finding logic in a new grouper, you should
   consider how it might be affected by having different coarse-graining in
   different trees.

   For trees with early triggers, there is an extra complication.
   There are two notions of arrival time: pulse arrival time at the trigger
   frequency, and pulse arrival time at the lower edge of the full band.
   Currently, we send the latter arrival time (lower edge of full band)
   to the sifter, even though this arrival time will often be in the future!
   (This detail is also "hidden" in `FrbGrouper.create_events()`)

   Another thing that happens in pirate (but not its predecessor bonsai):
   for some trees, only the upper half of the DM range is passed to the grouper,
   i.e.\ the lower edge of the dm-axis doesn't correspond to dm=0.
   (This happens for downsampled trees, since the lower half of the dm-range
   has been searched by a previous `ds_level`. You can test for this by either
   checking `ds_level > 0` or `dm_min > 0` in the per-tree metadata.)
   This may affect details of your peak-finding logic.

   For a lot more info on these details, see the tex notes.

 - The dedisperser passes `out_argmax` arrays to the grouper, to indicate which
   fine-grained (dm, time, frequency subband, peak-finding width) is responsible
   for each coarse-grained maximum SNR. Currently, these `out_argmax` arrays are
   ignored by `FrbGrouper.create_events()`, which just uses the central (dm,time)
   in each coarse-grained pixel, and sends placeholder values for the per-event
   `width_ms` (one time sample) and frequency subband (`subband_freq_{lo,hi}_MHz`,
   set to the full band).

   In the future, I plan to add code to recover fine-grained information from
   the `out_argmax` array. (This is not as trivial as it sounds, since the `out_argmax`
   array uses an "encoding" that's convenient in the GPU kernel, and I need to write code
   to decode it.)

The `FrbGrouper` and `FrbSifter` classes aren't well-optimized at all.
However, I find empirically that the "toy" grouper does not slow down a CHORD-scale search.
If this changes in the future, then here are some optimization ideas (just want to write
these down somewhere so I don't forget):

 - The biggest open risk is that cupy will not be fast enough for efficient peak-finding.
   If this turns out to be an issue, then a custom cuda kernel or two might help.
   It may also help to use a larger batch size, or to switch to an interface which
   allows processing multiple batches in parallel (see above).
 
 - Currently, the pirate-grouper intercommunication synchronizes cuda streams
   with host grpc threads on both sides. It would be more efficient to serialize and
   exchange cuda events, to directly "couple" pirate's cuda streams to cupy's streams).

 - Currently, grouper-sifter intercommunication is synchronous: the grouper blocks
   until a tcp connection to the sifter is opened and closed. It would be better to
   use asynchronous communication, with the tcp connection managed by another thread.
