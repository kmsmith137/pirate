"""
Toy offline dedispersion driver + rudimentary peak-finding over an acquisition directory.

Reads a directory of acquired AssembledFrame ".asdf" files (one file per
(beam, time chunk) -- as written by 'pirate_frb rpc_start_stream'). Beams and their
time chunks are enumerated with pirate_frb.utils.list_acqdir(). Each beam is
processed independently: a fresh single-beam OfflineDedisperser (nbeams == 1) is
built from scratch, and for each of that beam's time chunks the driver:

  1. uploads the quantized (int4) data to the GPU,
  2. dequantizes it to float16,
  3. tree-dedisperses it on the GPU,
  4. does rudimentary peak finding (per-chunk max SNR over the DM-vs-time plane of
     every dedispersion tree),

and prints the peak SNR of each chunk (one beam+chunk pair per line).

Run via: python -m pirate_frb run_toy_offline_dd ACQDIR CONFIG [--max-chunks N]
"""


DEFAULT_ACQDIR = "/mnt/cs00/data/kmsmith/2026-07-05/streams/kmsmith_26_07_05_202822"
DEFAULT_CONFIG = "configs/dedispersion/chord_sb2.yml"


def _process_beam(beam_id, files, config):
    """Dedisperse all of one beam's chunks and print each chunk's peak SNR.

    Builds a fresh OfflineDedisperser for this beam (which lazily initializes its
    GpuDedisperser + analytic weights from the first frame's metadata), then
    processes the frames in time order (one line per beam+chunk). Rudimentary peak
    finding: the printed SNR is the max out_max over the DM-vs-time plane of every
    tree, done on the GPU.
    """
    import cupy as cp
    from .core import AssembledFrame
    from .OfflineDedisperser import OfflineDedisperser

    # FIXME: there is currently no way to reset an OfflineDedisperser, so we
    # construct a new one from scratch for every beam.
    od = OfflineDedisperser(config)
    print(f"\nBeam {beam_id}: {len(files)} chunks")

    for ichunk, path in enumerate(files):
        frame = AssembledFrame.from_asdf(path)
        assert frame.beam_id == beam_id, (frame.beam_id, beam_id)

        # The first run() lazily builds the pipeline from the frame's metadata; the
        # dedisperser geometry (nfreq / nt_in / ntrees) is known only afterwards.
        outputs = od.run(frame)
        if ichunk == 0:
            print(f"  nfreq={od.nfreq}, nt_in={od.nt_in}, ntrees={od.ntrees}, dtype={od.dtype}, "
                  f"time_sample_ms={od.time_sample_ms:.4f}")

        # out_max[t] has shape (beams_per_batch=1, ndm_out, nt_out); only the
        # scalar peak is copied back to the host.
        snr = max(float(cp.asarray(outputs.out_max[t]).max()) for t in range(od.ntrees))
        print(f"  beam {beam_id}  chunk {ichunk:3d} (tci={frame.time_chunk_index:3d}):  "
              f"snr_max={snr:7.3f}")

    od.close()


def run_toy_offline_dd(acqdir=DEFAULT_ACQDIR, config_filename=DEFAULT_CONFIG, max_chunks=None):
    """Offline single-beam dedispersion + peak-SNR print over an acqdir (see module docstring).

    Parameters
    ----------
    acqdir : str
        Directory of 'frame_b(BEAM)_t(CHUNK).asdf' files (a 'start_stream' acqdir).
    config_filename : str
        Dedispersion config YAML (loaded with DedispersionConfig.from_yaml).
    max_chunks : int or None
        If given, only process the first 'max_chunks' time chunks of each beam.
    """
    from . import DedispersionConfig
    from .utils import list_acqdir

    beams = list_acqdir(acqdir)   # [(beam_id, [filename, ...]), ...], sorted
    print(f"Found {len(beams)} beam(s) in {acqdir}: {[b for b, _ in beams]}")

    config = DedispersionConfig.from_yaml(config_filename)

    for beam_id, files in beams:
        if max_chunks is not None:
            files = files[:max_chunks]
        _process_beam(beam_id, files, config)
