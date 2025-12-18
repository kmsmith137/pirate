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
  
### Protocol

We need a network protocol, for sending data from the X-engine to the FRB server.

  - We decided to use TCP in CHORD (unlike CHIME).

  - I think it will make sense to have 128 x 28 TCP connections. Each of the 64
    X-engine nodes processes an independent set of frequencies on each of its CPUs,
    and each of the 14 FRB nodes processes an independent set of beams on each
    of its CPUs.

  - Over each TCP connection, metadata should be sent first (yaml? json?)
    followed by data in regular chunks. The chunked data consists of some
    small float16 arrays (offsets/scales) and larger int4 arrays (intensities).
    (At least, this is what's currently planned in the FRB quantization kernel
    that runs on the X-engine.)

  - The details of the metadata format, and "chunked" data format are TBD.
    Please feel free to dive in and propose something.

  - I like the design principle that the metadata should contain all relevant
    configuration info from the X-engine, and that this info should not appear
    redundantly in an FRB server configuration file. For example, the number
    of frequency channels, upchannelization scheme, and beam configuration
    should all be received from the X-engine.

    However, this may complicate the FRB server, since some configuration info
    will be received in real time. I don't see any reason offhand why this would
    be a problem, but let's see how it goes.

  - What should we do if different X-engine nodes send inconsistent metadata?
    Crash?

### Low-level network code

The repo currently includes a little bit of low-level code that may be helpful:

  - `Socket`: C++ wrapper class for unix socket, provides RAII semantics
     and translates unix error codes to exceptions.

  - `Epoll`: similar C++ wrapper class for linux epoll file descriptor.
     (Allows one thread to read from multiple TCP sockets efficiently, see `man epoll`).

  - To see `class Socket` and `class Epoll` in action, check out `FakeServer.cu`,
    which accepts connections from a fixed number of TCP sockets, and then reads
    data in parallel from them.

  - For context (and note that this is a bit of a tangent), the FakeServer
    is some code intended for testing hardware, which times a synthetic load
    that consists of several things in parallel: network IO, disk IO, GPU-host
    PCIe transfers, GPU compute kernels, CPU compute kernels, host memory bandwidth.

    If you'd like to run the FakeServer, here's an example networking-only run
    (see `python -m pirate_frb test_node --help` for more info):

    ```
    # On cx67. The test will pause after "listening for TCP connections".
    python -m pirate_frb test_node -n --toronto

    # On cx68. Send to all 4 IP addresses on cx67.
    python -m pirate_frb send 10.50.0.1 10.50.1.1 10.50.2.1 10.50.3.1
    ```
    I don't expect that the FakeServer will have much long-term usefulness,
    but browsing the code may help a little with getting started.

  - Note that the FakeServer network receive code assumes that the number of
    senders is known in advance. The real networking code should allow senders
    to dynamically open/close connections (e.g. if an X-engine node goes down).
    This will make the real code more complicated than the FakeServer.

  - I think the "listener" thread which accepts new connections on a listening
    socket (and parses metadata) should be different from the "receiver" thread
    that receives chunked data. Do we also need a third "assembler" thread,
    which rearranges chunked data into an internal data structure?
    (`assembled_chunk` in CHIME, but in CHORD I'd prefer `AssembledFrame`
    for consistency with naming conventions in the rest of the code).
  
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

  - To start out, we should just read data and throw it away, and verify that
    we can keep up with the data rate. Later, we can add the `AssembledFrame`
    data structure, triggered ring buffer, and copy to the GPU.

  - Feel free to change the way that things are done in the code, add new
    dependencies (e.g. grpc) etc. I'm a big believer in software development
    by iterative improvement -- it's hard to get things right on the first try.
