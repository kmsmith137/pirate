# Grouper interface

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

We run one one grouper per GPU, i.e. two groupers per search node.
The grouper exchanges data arrays with `pirate` via a shared
GPU memory ring buffer. This avoids the overhead of copying data
between GPU and CPU. The grouper exhanges **metadata** with `pirate`
via `grpc` over the loopback network (`127.0.0.1`).
For performance reasons, the grouper should leave arrays on the GPU
if possible, and process them with `cupy`.

 - Python class for the grouper endpoint: {py:class}`~pirate_frb.pirate_pybind11.FrbGrouper`.
 - Python class for the sifter client: {py:class}`~pirate_frb.rpc.FrbSifterClient`.
 - Protocol for pirate-grouper communication: [`grpc/frb_grouper.proto`](../grpc/frb_grouper.proto).
 - Protocol for grouper-sifter communication: [`grpc/frb_sifter.proto`](../grpc/frb_sifter.proto).

## Example code (placeholder)

Still converging on final interface:

  - Refactor `chord_grouper` code into `FrbServer` methods?
  
  - I may rethink the details of the `FrbServer` methods.

  - Code should note that cupy is assumed, and that you must not
    touch the GPU before entering the context manager.
  
## FrbServer context manager

When the `FrbServer` context manager is entered, a lot happens under the hood:

 - The grouper starts a grpc service, and blocks until `pirate` connects.
   Note that the grouper is the grpc server, and `pirate` is the client.
   
 - The grouper receives metadata (three yaml objects, see below) from `pirate`.

 - The grouper learns which GPU it is running on (by receiving a cuda `device_id`),
   and *changes the current cupy stream* to the appropriate GPU. This change will
   be un-done when the context manager exits. Within the `FrbServer` context manager,
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
 
## Metadata

The grouper receives three metadata objects from `pirate`:

 1. `xengine_metadata_yaml` -- the X-engine metadata
    ([example](../configs/xengine_metadata.yml)).

 2. `dedispersion_config_yaml` -- same format as on-disk dedispersion configs (XXX example).

 3. `dedispersion_plan_yaml` -- more detailed dedispersion metadata, including array shapes
    ([example](../configs/example_dedispersion_plan.yml)).
    
XXX FrbGrouper member names?

Note: a `dedispersion_config` file is specified on the `pirate` command line
when `pirate` is started, and passed through to the grouper mostly unmodified
(item 2 above). However, the following members may be modified, if the real-time
values in the X-engine are different from the "static" values in the config file:
`zone_nfreq`, `zone_freq_edges`, `time_sample_ms`, `beams_per_gpu`.

## Data arrays (placeholder)

 - Outer loop over time chunk, inner loop over beam "batches"
 
 - 3-d arrays indexed by (beam, dm, time)

 - Meaning of time coordinate is different for early triggers.
