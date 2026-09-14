# FRB maps for classifier development

Open [RetrieveEvent.ipynb](RetrieveEvent.ipynb) to follow a Grouper event
back to its source pixel and extract a small DM-time window with a bowtie mask.
The notebook uses PIRATE's production map loader, GPU argmax decoder, and
peakfinder geometry. Its maps, bowtie mask, and classifier window are CuPy
arrays on the GPU. ASDF reading and plotting use the CPU.

Use the same built PIRATE environment as the offline and online examples,
with ksgpu, CuPy, and an NVIDIA GPU (see [installation](../notes/install.md)).
Install the notebook extras and launch from the repository root:

```bash
python -m pip install -r AIclassifier/requirements.txt
python -m jupyterlab AIclassifier/RetrieveEvent.ipynb
```

Choose that same environment as the notebook kernel, then select
**Restart Kernel and Run All Cells**. Set `CUDA_DEVICE_ID` in the first cell
if needed. A copied folder requires an installed PIRATE package in that
environment. The notebook locates `data/` from either this folder or the
repository root; it does not depend on `/tmp` or the generator's original paths.

## Included data

- `data/frame_b1_t19_snrmap.asdf`: one unmodified source chunk containing the
  FRB. The S/N map is `(4096, 128)` in `(DM row, time column)` order. Its matching
  argmax map, producer plan, timing, and beam metadata are retained.
- `data/events.asdf`: the small original Grouper catalog. It contains one event
  plus the metadata and coverage of the full 32-chunk search. The event was
  found using that full observation, not by regrouping the isolated chunk.

Both files are direct copies from the [simple offline example](../examples/simple_frb/README.md).
Together they occupy about 3 MiB. The map's `source.filename` records the original
intensity-frame path as provenance; the notebook does not open that file.

The injected burst was on beam 1, DM 500 pc cm^-3, Gaussian sigma 1 ms, S/N 50,
with flat-spectrum emission across 300-1500 MHz, arriving at 40 s at 300 MHz.
The recovered representative is beam 1, tree 0, source chunk 19, `idm=1385`,
`itime=72`, DM 499.997795, S/N 46.0625. Its recovered 2.9952 ms boxcar width is
not the injected Gaussian sigma.

## What the notebook explains

1. Load an event and choose its beam, tree, and actual source chunk.
2. Load maps with `FrbOfflineGrouper.load_beam_chunk`, select the beam axis,
   and check the exact `(idm, itime)` pixel against the event.
3. Decode that pixel with `GpuArgmaxDecoder.decode` and verify DM, arrival time,
   width, and band against the catalog.
4. View the chunk and the burst neighborhood.
5. Extract a fixed-size window, padding unavailable context with NaN and
   keeping a separate validity mask.
6. Use `PeakFinderGeometry.full_band_bowtie` from the recorded settings.
7. Return CuPy arrays and metadata that a classifier can consume.

The default bowtie window is `(17, 19)` and fits entirely inside the saved
chunk. Only chunk 19 is included. If a different window crosses its boundary,
the missing columns stay invalid: neighboring chunks are needed to fill them.
This is missing *sample context*, not evidence of a physical acquisition edge.
The bowtie is the peakfinder's comparison footprint, not a classifier model.

The notebook calls `PeakFinderGeometry.from_plan` directly, including its
one-chunk time-radius crop. Offline and online grouping use this same class
and decoder. The only classifier-specific helper slices and pads a window;
it does not duplicate the decoding or bowtie calculation. Source pointers are
collected at the end of the notebook.

## Regenerate the sample

Run all three offline steps in `examples/simple_frb/README.md`. Then copy
`frames/frame_b1_t19_snrmap.asdf` and `events.asdf` from that run into `data/`.
If parameters, software, or the detected source chunk change, choose the map
named by the new catalog's `beam_id` and `source_chunk_index` instead. Keep the
catalog and map from the same run; the notebook verifies their producer metadata
and representative pixel before making a window.
