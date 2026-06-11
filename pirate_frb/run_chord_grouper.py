"""Prototype CHORD FrbGrouper consumer(s).  Sends coarse-grained maxes to FRB Sifter."""

from .run_toy_grouper import run_groupers

import itertools
import sys
import time

def _run_chord_grouper(grouper_addr, sifter_addr, grouper, delay=0.0):
    """Main loop (factored out of run_chord_grouper to reduce nesting).

    For each time chunk, accumulate the global max over all beam-batches, trees,
    DMs and time samples of 'out_max', and print one line per chunk. The cupy
    work runs on the current/default stream (FrbGrouper.__enter__ already
    selected the right device); get_output's __exit__ synchronizes that stream
    before releasing each batch.

    If 'delay' > 0, sleep that many seconds at the end of each chunk -- an
    artificial slowdown for testing how the producer behaves when the consumer
    lags.
    """
    import cupy as cp

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
    print('Nbeams:', nbeams)
    # time_samples_per_chunk
    #grouper.nt_in

    xengine = yaml.load(grouper.xengine_metadata_yaml_string, Loader=Loader)
    beamset = xengine['beamset']
    print('Beamset:', beamset)

    # timing
    seq_per_frb_time_sample = xengine['seq_per_frb_time_sample']
    time_samples_per_chunk = grouper.nt_in
    fpga0_nano = xengine['unix_ns_at_seq_0']
    nano_per_fpga = xengine['dt_ns_per_seq']
    fpga_per_chunk = seq_per_frb_time_sample * time_samples_per_chunk
    print('Time samples per chunk:', time_samples_per_chunk)
    print('FPGA seq per time sample:', seq_per_frb_time_sample)
    print('nanoseconds per FPGA seq:', nano_per_fpga)
    print('FPGA per chunk:', fpga_per_chunk)

    my_config = dict(the_answer=42)
    my_config_yaml = yaml.dump(my_config)

    ch1 = grpc.insecure_channel(sifter_addr)
    sifter = FrbSifterStub(ch1)
    msg = ConfigMessage(xengine_yaml=grouper.xengine_metadata_yaml_string,
                        pirate_yaml=grouper.dedispersion_config_yaml_string,
                        dedispersion_plan_yaml=grouper.dedispersion_plan_yaml_string,
                        grouper_yaml=my_config_yaml)
    print('Sending first gRPC to Sifter...')
    r1 = sifter.CheckConfiguration(msg)
    print('Got Sifter result:', r1.ok)

    for ichunk in itertools.count():            # loop over time chunks
        running_max = cp.full((1,), -cp.inf, dtype=cp.float32)

        per_beam_max = cp.full((nbeams,), -cp.inf, dtype=cp.float32)

        beam_index = 0
        for ibatch in range(grouper.nbatches):  # loop over beam batches
            seq_id = ichunk * grouper.nbatches + ibatch
            with grouper.get_output(seq_id) as outputs:
                #print('Chunk', ichunk, 'batch', ibatch)
                # outputs.out_max: list (length ntrees) of cupy arrays (views
                # into the IPC-mapped memory via DLPack). get_output's __exit__
                # synchronizes the current stream before releasing the batch.

                # outputs.out_argmax (uint32)
                # outputs.out_max (float16/float32)
                #   each out_max has shape (beam_per_batch, DM, T)

                #print('outputs:', dir(outputs))
                #'ibeam', 'ichunk_fpga_based', 'ichunk_zero_based'
                if ibatch == 0:
                    print('output: ibeam', outputs.ibeam,
                          'ichunk_fpga:', outputs.ichunk_fpga_based,
                          'ichunk_zero_based', outputs.ichunk_zero_based)
                    # we want the fpga-based one -- ichunk_fpga_based
                    fpga_chunk_start = outputs.ichunk_fpga_based * fpga_per_chunk
                    fpga_chunk_end = (outputs.ichunk_fpga_based + 1) * fpga_per_chunk
                    print('chunk FPGA seq:', fpga_chunk_start, 'to', fpga_chunk_end)
                    # unix_time_nano_start = fpga0_nano + fpga_chunk_start * nano_per_fpga
                    # unix_time_nano_end   = fpga0_nano + fpga_chunk_end   * nano_per_fpga
                    # print('unix nano:', unix_time_nano_start, unix_time_nano_end)
                    # nano = 1_000_000_000
                    # date_start = datetime.fromtimestamp(unix_time_nano_start // nano +
                    #                                     1e-9 * (unix_time_nano_start % nano))
                    # date_end = datetime.fromtimestamp(unix_time_nano_end // nano +
                    #                                     1e-9 * (unix_time_nano_end % nano))
                    # print('chunk:', date_start, 'to', date_end)

                # out_max is a list of ntrees ksgpu.Array objects.
                for tree_out in outputs.out_max:        # loop over trees
                    #print('Tree_out:', type(tree_out), tree_out.shape)
                    (nbeam, ndm, nt) = tree_out.shape
                    cp.maximum(running_max, tree_out.max(), out=running_max)
                    #print('beam-wise max:', tree_out.max(axis=(1,2)).get())
                    cp.maximum(per_beam_max[beam_index:beam_index+nbeam],
                               tree_out.max(axis=(1,2)),
                               out=per_beam_max[beam_index:beam_index+nbeam])
                beam_index += nbeam

        # float() does a D2H copy (+ sync); one print per chunk.
        print(f'{grouper_addr}: ichunk={ichunk}: '
              f'global out_max = {float(running_max[0])}', flush=True)

        bmax = per_beam_max.get()
        print(f'{grouper_addr}: ichunk={ichunk}: '
              f'per-beam max =', '[ ' + ', '.join(['%.1f' % b for b in bmax]) + ' ]',
              flush=True)

        # Send to Sifter
        # (we could coarse-grain the SNR values over more than one chunk...)
        msg = FrbEventsMessage(has_injections=0,
                               beam_set_id=beamset,
                               chunk_fpga_count=fpga_chunk_start,
                               events=[],
                               coarsegrain_start_fpga_count=fpga_chunk_start,
                               coarsegrain_end_fpga_count=fpga_chunk_end,
                               coarsegrain_snr=bmax)
        reply = sifter.FrbEvents(msg)
        print('Got Sifter result:', reply.ok, 'message', reply.message)

        if delay > 0:
            time.sleep(delay)

                    
'''
dedisp meta:
beams_per_gpu: 16
beams_per_batch: 2

Chunk 180 batch 0
Tree_out: <class 'cupy.ndarray'> (2, 128, 16)
Tree_out: <class 'cupy.ndarray'> (2, 64, 8)
Tree_out: <class 'cupy.ndarray'> (2, 64, 8)
Tree_out: <class 'cupy.ndarray'> (2, 32, 4)
Tree_out: <class 'cupy.ndarray'> (2, 64, 4)
Tree_out: <class 'cupy.ndarray'> (2, 64, 4)
... batch 7

Dedisp plan:

trees:
- tree_index: 0
  ndm_out: 128
  nt_out: 16
  dm_min: 0
  dm_max: 52.570234149182113
  trigger_frequency: 400
  ds_level: 0
  delta_et: 0
  max_width: 16
  dm_downsampling: 8
  time_downsampling: 16
  wt_dm_downsampling: 64
  wt_time_downsampling: 64
  frequency_subband_counts: [0, 3, 2, 1]
        '''



def run_chord_grouper(grouper_addr, sifter_addr, delay=0.0):
    """Run a toy FrbGrouper consumer at 'grouper_addr' (e.g. '127.0.0.1:7000').

    Acts as the downstream consumer of an FrbServer producer over CUDA IPC.
    Blocks (in FrbGrouper.open(), via __enter__) until the producer connects,
    then prints the per-chunk global 'out_max' until the producer disconnects or
    Ctrl-C.

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
