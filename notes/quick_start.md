# Quick start

## Run unit tests

```
# Run all unit tests 5 times (default is 100!)
# See 'pirate_frb test --help' for many more flags.
pirate_frb test -n 5
```

## Running a toy search

The "toy" search is a subscale example which starts quickly, runs over the loopback network on
a single node, and uses a small fraction of a single GPU.
To run the full sequence (fake X-engine) -> (FRB search) -> (grouper) -> (sifter),
run the following commands in separate terminal windows:
```
# Window 1: start the sifter (waits for grouper to connect)
pirate_frb run_toy_sifter 127.0.0.1:7500

# Window 2: start the grouper (waits for search to connect)
pirate_frb run_toy_grouper -s 127.0.0.1:7500 127.0.0.1:7000

# Window 3: start the search (waits for fake X-engine to connect)
pirate_frb run_server configs/frb_server/toy.yml configs/dedispersion/toy.yml

# Window 4: start the fake X-engine. Data will start streaming through all 4 processes.
pirate_frb run_fake_xengine 127.0.0.1:6000

# Optional: in window 5, send RPC "status" requests to the server.
# This will monitor connections, bytes received, files written, ring buffer state.
pirate_frb rpc_status 127.0.0.1:6000

# Optional: in window 6, send RPC "write_files" requests to the server, for randomly
# chosen beams/times. Filenames will be printed in the 'rpc_status' process (window 5)
# as they are written. Files appear in /dev/shm/pirate_nfs, and will be deleted when
# the server exits.
pirate_frb rpc_write 127.0.0.1:6000
```
Note that you don't need to run this entire sequence every time!
The programs above have command-line args to "short-circuit" the downstream programs.
(For example, `pirate_frb run_server --no-grouper` or `pirate_frb run_toy_grouper --no-sifter`.)

## Running a production search (cf00/cf05)

The "production" search uses an entire node (cf05), with full CHORD parameters,
and many beams. We use a different node (cf00) as the fake X-engine, and send data
over the real network (not the loopback network). In this example, we run the
sifter on cf05, but it could be run on cf00 (or a third node). Note that there
are two grouper processes (one per GPU), each of which independently connects
to the sifter.
```
# Window cf05/1: start the sifter (waits for grouper to connect)
pirate_frb run_toy_sifter 127.0.0.1:7500

# Window cf05/2: start the grouper (waits for search to connect)
pirate_frb run_toy_grouper -s 127.0.0.1:7500 127.0.0.1:7000 127.0.0.1:7001

# Window cf05/3: start the search (waits for fake X-engine to connect)
pirate_frb run_server configs/frb_server/cf05_production.yml configs/dedispersion/chord_sb2_et.yml

# Window cf00/1: start the fake X-engine. Data will start streaming through all 4 processes.
pirate_frb run_fake_xengine 10.222.3.5:6000 10.222.3.5:6001

# Optional: on either cf00 or cf05, send RPC "status" requests to the server.
pirate_frb rpc_status 10.222.3.5:6000 10.222.3.5:6001

# Optional: in window 6, send RPC "write_files" requests to the server, for randomly
# chosen beams/times. Filenames will be printed in the 'rpc_status' process as they
# are written. Files appear on the real NFS server: /mnt/cs00/data/{user}/{date},
# and persist after the server exits.
pirate_frb rpc_write 10.222.3.5:6000 10.222.3.5:6001
```
