# Hardware

## Nodes

  - The CHORD FRB nodes have:
  
     - Two 16-core Intel CPUs
     - 2x1 TB DDR5 memory (8 channels/CPU)
     - 2xL40S GPUs
     - 2x3TB SSDs (one of these is the root filesystem)
     - 2x2x25 GbE NICs (plus additional NICs for admin)
     - 2x1 GbE NICs (for admin and RPC)

  - The nodes are "balanced", in the sense that each CPU is physically connected
    to half of the hardware (i.e. 1 TB memory, one GPU, 2x25 GbE networking, one SSD).
    
    (Exception: the 1GbE NICs are both on the first CPU. These are low-bandwith
    enough that there shouldn't be performance implications, but it may affect
    details of our core-pinning logic.)
  
  - To avoid cross-numa IO, each CPU runs an independent `FrbServer` instance
    which processes a separate set of beams. All threads should be pinned to the
    appropriate CPU, and should only "touch" memory that was originally allocated
    on the same CPU (with minor exceptions as needed).
    
    Currently, I see no need for these two FrbServer instances to interact
    which each other. It may be helpful to think of the FRB node as two independent
    "half-nodes" running on one physical machine.

## Full CHORD network (future)

  - In full CHORD, we'll have 64 X-engine nodes and 14 FRB nodes.
    We're currently planning to use TCP for communication, but we
    might switch to RDMA in the future, if we run into performance issues.
  
  - As described above, each FRB node can be viewed as two half-nodes
    which processes an independent set of beams. Similarly, each X-engine
    node can be viewed as two half-nodes nodes which process an independent
    set of frequencies. Thus, we'll have 128 x 28 TCP connections (one connection
    for each X-engine half-node and FRB half-node).

  - Small complication: instead of having one big switch, we'll have two
    smaller switches to reduce cost.

    On the X-engine side, each switch handles half of the X-engine half-nodes.
    (I'm not sure whether the two half-nodes in a given X-engine node will be
    connected to the same switch, or different switches -- I don't think it matters.)

    On the FRB side, each half-node has 2x25 GbE NICs (see above).
    Each of these two NICs will be connected to a different switch.
    Since each FRB half-node is connected to both switches, each FRB half-node
    "sees" all 128 X-engine half-nodes, and no cross-numa IO is needed
    in the FRB node.

    In this scheme, each FRB beam receives half of its frequency channels
    from different NICs. This complicates the code a little, but is not really
    a big deal. We end up with two `struct Receiver` objects per `struct FrbServer`,
    and a little extra synchronization logic.

  - We'll want to decide how to divide traffic between the 2x2x25 GbE and 2x1 GbE NICs
    in the FRB node. Is this scheme best?
    
      - 4x25 GbE for X-engine -> FRB traffic only.
      - 1 GbE NIC for admin + ssh + "lightweight" RPCs.
      - 1 GbE NIC for "heavyweight" RPCs (i.e. RPCs with data payloads) and NFS.

    Context: most RPCs are tiny messages, but low latency can be important. A few RPCs
    do have data payloads (e.g. pulse injections: the RPC caller simulates the pulse and
    sends a data array).

## DRAO backend

We currently have a few FRB nodes in the DRAO block house, and an ad hoc network.
Let me know if you don't have access.

  - I'm using `cf05` as the test server, and  `cf00` to send test data
    (the "fake X-engine"). The other nodes are currently unused and unconfigured.
    Remaining bullet points apply only to `cf00` and `cf05`.

  - These nodes are configured with 75% of their host memory in 2MB hugepages:
    ```
    # In /etc/tmpfiles.d/hugepages-numa.conf
    w /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages - - - - 393216
    w /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages - - - - 393216
    ```
    
  - The directories `/scratch` and `/scratch2` are intended to test IO on
    the two SSDs. (Since the first SSD is the root filesystem, `/scratch` is
    an ordinary directory, whereas `/scratch2` is a mount point.)

    Note that on `cf05`, both SSDs are fast (6.7 GB/s) Kioxia drives.
    On `cf00`, one of the SSDs is a slower (3.5 GB/s) Samsung drive.

  - We don't have a 25 GbE switch yet, but The 2x2x25 GbE NICs in cf00 and cf05 are
    directly connected as follows:
    ```
    # cf00 -> cf05
    10.0.0.1 -> 10.0.0.2
    10.0.1.1 -> 10.0.1.2
    10.0.2.1 -> 10.0.2.2
    10.0.3.1 -> 10.0.3.2
    ```
    IP addresses are assigned in netplan (no ansible) and MTUs are set to 9000.

  - NFS server available at `/mnt/cs00/data`.
  