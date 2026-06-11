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

    my_config = dict(the_answer=42)
    my_config_yaml = yaml.dump(my_config)

    ch1 = grpc.insecure_channel(sifter_addr)
    stub1 = FrbSifterStub(ch1)
    msg = ConfigMessage(xengine_yaml=grouper.xengine_metadata_yaml_string,
                        pirate_yaml=grouper.dedispersion_config_yaml_string,
                        dedispersion_plan_yaml=grouper.dedispersion_plan_yaml_string,
                        grouper_yaml=my_config_yaml)
    print('Sending first gRPC to Sifter...')
    r1 = stub1.CheckConfiguration(msg)
    print('Got Sifter result:', r1.ok)

    for ichunk in itertools.count():            # loop over time chunks
        running_max = cp.full((1,), -cp.inf, dtype=cp.float32)

        for ibatch in range(grouper.nbatches):  # loop over beam batches
            seq_id = ichunk * grouper.nbatches + ibatch
            with grouper.get_output(seq_id) as outputs:
                # outputs.out_max: list (length ntrees) of cupy arrays (views
                # into the IPC-mapped memory via DLPack). get_output's __exit__
                # synchronizes the current stream before releasing the batch.
                for tree_out in outputs.out_max:        # loop over trees
                    cp.maximum(running_max, tree_out.max(), out=running_max)

        # float() does a D2H copy (+ sync); one print per chunk.
        print(f'{grouper_addr}: ichunk={ichunk}: '
              f'global out_max = {float(running_max[0])}', flush=True)

        if delay > 0:
            time.sleep(delay)


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
