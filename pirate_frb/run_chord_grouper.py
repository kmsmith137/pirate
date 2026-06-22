"""Prototype CHORD FrbGrouper consumer(s).  Sends coarse-grained maxes to FRB Sifter."""

from .run_toy_grouper import run_groupers

import itertools
import sys
import time

def _run_chord_grouper(grouper_addr, sifter_addr, grouper, delay=0.0):
    """Main loop (factored out of run_chord_grouper to reduce nesting).

    This is a prototype/placeholder for the CHORD FRB grouper.

    It uses gRPC calls (defined in grpc/frb_sifter.proto) to communicate
    with the sifter.

    Currenly only two types of messages are implemented:

    - the initial message that sends all the configuration (yaml)
      strings for the xengine, pirate, dedispersion details, as well as
      (this) grouper.

    - a per-time-period and per-beam maximum SNR.  This can be used as a
      "liveness" indicator or to replace the "ultra-coarse-grained
      viewer" in CHIME/FRB.

    The key RPC that still needs to be implemented is:

    - the peaks detected in the DM-vs-t planes produced by Pirate!
      These are candidate FRB events.

    Other missing bits include:

    - Pirate needs to send its gRPC endpoint address so that the Sifter
      can actually do callbacks!

    - There is a decision about how to coarse-grain the time in the
      beam-max-snr data flow.  It would be good for the multiple Pirate
      instances (ie, this code) to send their beam-max-snr values at the
      same time.  The time system they share is the FPGA counter, so the
      time chunking should probably be aligned based on that.

    If 'delay' > 0, sleep that many seconds at the end of each chunk -- an
    artificial slowdown for testing how the producer behaves when the consumer
    lags.
    """
    import cupy as cp
    import numpy as np

    from .rpc.grpc.frb_sifter_pb2_grpc import FrbSifterStub
    from .rpc.grpc.frb_sifter_pb2 import ConfigMessage, FrbEventsMessage, FrbEvent
    import grpc
    import yaml
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper
    from datetime import datetime

    print('Starting CHORD grouper: sifter address is', sifter_addr)
    print('xengine meta:')
    print('-------------------------------------------------')
    print(grouper.xengine_metadata_yaml_string)
    print('-------------------------------------------------')
    print('dedisp meta:')
    print('-------------------------------------------------')
    print(grouper.dedispersion_config_yaml_string)
    print('-------------------------------------------------')
    print('dedisp plan meta:')
    print('-------------------------------------------------')
    print(grouper.dedispersion_plan_yaml_string)
    print('-------------------------------------------------')

    # this is beams_per_gpu
    nbeams = grouper.total_beams
    xengine = yaml.load(grouper.xengine_metadata_yaml_string, Loader=Loader)
    beamset = xengine['beamset']
    # timing parameters
    seq_per_frb_time_sample = xengine['seq_per_frb_time_sample']
    time_samples_per_chunk = grouper.nt_in
    fpga0_nano = xengine['unix_ns_at_seq_0']
    nano_per_fpga = xengine['dt_ns_per_seq']
    fpga_per_chunk = seq_per_frb_time_sample * time_samples_per_chunk
    print('Nbeams:', nbeams)
    print('Beamset:', beamset)
    print('Time samples per chunk:', time_samples_per_chunk)
    print('FPGA seq per time sample:', seq_per_frb_time_sample)
    print('nanoseconds per FPGA seq:', nano_per_fpga)
    print('FPGA per chunk:', fpga_per_chunk)

    # Add a fake yaml configuration string for this Grouper.
    my_config = dict(the_answer=42)
    my_config_yaml = yaml.dump(my_config)

    # Connect to the Sifter gRPC endpoint.
    ch1 = grpc.insecure_channel(sifter_addr)
    sifter = FrbSifterStub(ch1)
    # Make the first Sifter gRPC call -- send yaml configuration strings.
    msg = ConfigMessage(xengine_yaml=grouper.xengine_metadata_yaml_string,
                        pirate_yaml=grouper.dedispersion_config_yaml_string,
                        dedispersion_plan_yaml=grouper.dedispersion_plan_yaml_string,
                        grouper_yaml=my_config_yaml)
    print('Sending first gRPC to Sifter...')
    r1 = sifter.CheckConfiguration(msg)
    print('Got Sifter result:', r1.ok)

    # Approximately how often to send the coarse-grained beam SNRs to the
    # Sifter, in seconds
    beam_snr_period = 2.0
    # convert to number of chunks (otherwise, we might include a varying number of
    # chunks, and then the S/N statistics will be different)
    beam_snr_chunks = int(np.round(beam_snr_period / (fpga_per_chunk * nano_per_fpga * 1e-9)))

    # For each beam, the maximum SNR we have seen in this time period (start
    # uninitialized)
    per_beam_max = None

    # Loop over time chunks (forever)
    for ichunk in itertools.count():

        if per_beam_max is None:
            per_beam_max = cp.full((nbeams,), -cp.inf, dtype=cp.float32)

        # Pirate provides DM-vs-t arrays in "beam batches";
        # these are defined in the dedispersion metadata; if
        #   beams_per_gpu: 16
        #   beams_per_batch: 2
        # Then there will be 8 beam-batches, each with two beams.
        # The shapes of the DM-vs-t arrays vary by tree, as defined in the
        # dedispersion-plan metadata.  An example set of shapes would be:
        #
        #   Chunk 180 batch 0
        #   Tree_out: <class 'cupy.ndarray'> (2, 128, 16)    (nbeams, ndm, nt)
        #   Tree_out: <class 'cupy.ndarray'> (2, 64, 8)
        #   Tree_out: <class 'cupy.ndarray'> (2, 64, 8)
        #   Tree_out: <class 'cupy.ndarray'> (2, 32, 4)
        #   Tree_out: <class 'cupy.ndarray'> (2, 64, 4)
        #   Tree_out: <class 'cupy.ndarray'> (2, 64, 4)
        #   Chunk 180 batch 1
        #   .... same
        #   up to batch 7.

        # which beam index are we looking at now?  We're maybe supposed to use
        # outputs.ibeam instead?
        beam_index = 0
        for ibatch in range(grouper.nbatches):
            seq_id = ichunk * grouper.nbatches + ibatch
            with grouper.get_output(seq_id) as outputs:
                # outputs.out_max: list (length ntrees) of cupy arrays
                # (views into GPU memory)
                # data type is (float16 or float32)
                #   each out_max has shape (beam_per_batch, DM, T)
                #
                # outputs.out_argmax (uint32)
                #   is the same shape

                if ibatch == 0:
                    print('output: ibeam', outputs.ibeam,
                          'ichunk_fpga:', outputs.ichunk_fpga_based,
                          'ichunk_zero_based', outputs.ichunk_zero_based)
                    # we want the fpga-based one -- ichunk_fpga_based
                    fpga_chunk_start = outputs.ichunk_fpga_based * fpga_per_chunk
                    fpga_chunk_end = (outputs.ichunk_fpga_based + 1) * fpga_per_chunk
                    print('chunk FPGA seq:', fpga_chunk_start, 'to', fpga_chunk_end)

                    # We don't need to convert to wall time, but for reference,
                    # here's how to do so:
                    # unix_time_nano_start = fpga0_nano + fpga_chunk_start * nano_per_fpga
                    # nano = 1_000_000_000
                    # date_start = datetime.fromtimestamp(unix_time_nano_start // nano +
                    #                                     1e-9 * (unix_time_nano_start % nano))

                    # If we are starting a new beam-max-snr time period, record
                    # the start time.
                    if ichunk % beam_snr_chunks == 0:
                        beam_fpga_start = fpga_chunk_start

                # outputs.out_max is a list of length ntrees of ksgpu.Array objects.
                for tree_out in outputs.out_max:
                    # We don't really need to know the shape, but for reference:
                    (nbeam, ndm, nt) = tree_out.shape
                    # Update the per-beam max SNR seen
                    cp.maximum(per_beam_max[beam_index:beam_index+nbeam],
                               tree_out.max(axis=(1,2)),
                               out=per_beam_max[beam_index:beam_index+nbeam])
                beam_index += nbeam

        # .get() does the gpu device-to-host copy.
        bmax = per_beam_max.get()
        print(f'{grouper_addr}: ichunk={ichunk}: '
              f'per-beam max =', '[ ' + ', '.join(['%.1f' % b for b in bmax]) + ' ]',
              flush=True)

        # If this is the final chunk in our beam-max-snr chunk, record the
        # final time and send it to the Sifter!
        if (ichunk+1) % beam_snr_chunks == 0:
            beam_fpga_end = fpga_chunk_end

            # Send to Sifter
            msg = FrbEventsMessage(has_injections=0,
                                   beam_set_id=beamset,
                                   chunk_fpga_count=fpga_chunk_start,
                                   events=[],
                                   coarsegrain_start_fpga_count=beam_fpga_start,
                                   coarsegrain_end_fpga_count=beam_fpga_end,
                                   coarsegrain_snr=bmax)
            reply = sifter.FrbEvents(msg)
            print('Got Sifter result:', reply.ok, 'message', reply.message)

        if delay > 0:
            time.sleep(delay)

def run_chord_grouper(grouper_addr, sifter_addr, delay=0.0):
    """Run a toy FrbGrouper consumer at 'grouper_addr' (e.g. '127.0.0.1:7000').

    Acts as the downstream consumer of an FrbServer producer over CUDA
    IPC.  Blocks (in FrbGrouper.open(), via __enter__) until the
    producer connects, then collects per-beam maximum SNR in multi-chunk
    time periods, and sends them to the Sifter; until the producer
    disconnects or Ctrl-C.

    'delay' (seconds) inserts an artificial per-chunk slowdown into the loop;
    see _run_chord_grouper.
    """
    # Imported here (not at module top) so 'import pirate_frb' stays light.
    from .rpc import FrbGrouper

    # FrbGrouper.__enter__ blocks until the producer connects, then pins this
    # thread to the GPU's vcpus and selects the CUDA device (printing a message);
    # __exit__ restores them and closes the grouper.
    with FrbGrouper(grouper_addr) as grouper:
        try:
            _run_chord_grouper(grouper_addr, sifter_addr, grouper, delay)
        except KeyboardInterrupt:
            print(f'{grouper_addr}: interrupted; shutting down', flush=True)
        except RuntimeError as e:
            # Most likely the producer disconnected (the grouper stops and
            # acquire_output rethrows). Report cleanly; re-raise anything that is
            # not a stop (e.g. a genuine usage/assert bug).
            if grouper.is_stopped:
                print(f'{grouper_addr}: producer disconnected ({e}); exiting', flush=True)
            else:
                raise
        # FrbGrouper.__exit__ restores affinity/device + closes on every path.

def run_chord_groupers(grouper_addrs, sifter_addr, delay=0.0):
    run_groupers(run_chord_grouper, grouper_addrs, (sifter_addr,), dict(delay=delay),
                 [sys.executable, '-m', 'pirate_frb', 'run_chord_grouper',
                  '--sifter', sifter_addr,
                  '--delay', str(delay)])
