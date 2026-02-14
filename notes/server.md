## FRB server

Here are some initial thoughts on writing the real-time server frontend.

### Hardware

  - The CHORD FRB nodes have:
  
     - Two 16-core Intel CPUs
     - 2x1 TB DDR5 memory (8 channels/CPU)
     - 2xL40S GPUs
     - 2x2x25 GbE NICs (plus additional NICs for admin)

  - Each thread in the real-time system should be "pinned" to the cores on one
    of the CPUs. It should only interact with the GPU/NIC on its CPU, and only
    "touch" memory that was originally allocated on its CPU (with minor exceptions
    as needed).

    To make this possible, I suggest that each CPU should process an independent
    set of beams (like in CHIME).
  
  - I hacked up some terrible python code to query hardware in real time, and
    work out which devices are associated with each CPU. For example, it will
    tell you which IP addresses are associated with each CPU, or the list of
    "vCPUs" associated with each CPU. (You need this list in order to pin threads
    to cores.) You can run this code with `python -m pirate_frb show_hardware`.

    I'd like to use this code (or something similar) to communicate hardware
    details to the real-time server, so that it behaves robustly if details
    change, such as NUMA/hyperthread configuration, or IP address assignment.
    Hopefully we can avoid rewriting this code in C++!

  - You may also find it useful to play with `Hwtest`:
    code intended for testing hardware, which times a synthetic load
    that consists of several things in parallel: network IO, disk IO, GPU-host
    PCIe transfers, GPU compute kernels, CPU compute kernels, host memory bandwidth.

    If you'd like to run the Hwtest, here's an example networking-only run
    (see `python -m pirate_frb hwtest --help` for more info):

    ```
    # On cx67. The test will pause after "listening for TCP connections".
    python -m pirate_frb hwtest config.yml

    # On cx68. Send to all 4 IP addresses on cx67.
    python -m pirate_frb hwtest -s config.yml
    ```
    I don't expect that the Hwtest will have much long-term usefulness,
    but browsing the code may help a little with getting started.
  
### X-engine to FRB networking

Each FRB node CPU (i.e. one "half" of a node, see above) has 2x25 GbE NICs,
and receives data for a certain per-CPU set of beams.

  - We decided to use TCP in CHORD (unlike CHIME).

  - I think it will make sense to have 128 x 28 TCP connections. Each of the 64
    X-engine nodes processes an independent set of frequencies on each of its two
    CPUs, and each of the 14 FRB nodes processes an independent set of beams on
    each of its two CPUs.

  - Complication: each FRB CPU (i.e. "half" of a node) has 2x25 GbE NICs.
    Each NIC receives **half of the frequencies for all beams**.
    This is less convenient than receiving all frequencies for half the beams,
    but reduces network hardware cost. It complicates the code a little (we
    end up with two `struct Receiver` objects per `struct FrbServer`).

  - I like the design principle that the metadata should contain all relevant
    configuration info from the X-engine, and that this info should not appear
    redundantly in an FRB server configuration file. For example, the number
    of frequency channels, upchannelization scheme, and beam configuration
    should all be received from the X-engine.

    However, this may complicate the FRB server, since some configuration info
    will be received in real time. So far, implementing this "dynamic configuration"
    feature hasn't been too painful, but let's see how it goes.

  - We currently crash if different X-engine nodes send inconsistent metadata,
    or if there is a big problem, such as failing to parse the network protocol.
    I think this is a better choice than "failing gracefully" -- we want to know
    if there's a misconfiguration/bug, so that a human can get involved.
  
### Design decisions

  - Where should we put the "python/C++ boundary"? For example, should the
    metadata parsing happen in python or C++? I don't have anything planned
    here -- let's just dive in and see how it goes.
    
    Sadly, I think we should avoid using free-threaded python -- I spent
    some time looking at it, and I just don't think it's ready for production
    yet. This is such a big constraint that I suspect we'll need to write
    almost all code in C++. Perhaps in a few years, we could revisit this
    decision. It would be fantastic if we could write most of the server code
    in python, and just call C++ for low-level compute kernels.

    Note that I'm using pybind11 to export C++ code to python. (Currently the
    python bindings are not really systematic -- I've just been adding ad hoc
    bindings whenever they're needed.)

  - How does the server start up? One option is to define
    `python -m pirate_frb server [config.yaml]`. Another option is to
    not use the yaml file, and write short python scripts which construct
    and run a Server object.

  - How does the server shut down? Long-term I like the idea of a "shutdown"
    RPC, but in the short term we might want to implement a simpler alternative.
