# Introduction

Some notes/caveats:

  - This repo contains real-time server code (network receive, ring buffering,
    file-writing RPCs) and GPU dedispersion code, but currently the **real-time
    server does not call the GPU dedisperser**! Working on this is my top
    priority.

  - Currently, this code can only be compiled on a recent ubuntu
    linux machine with **a physical GPU**, and the cuda toolkit installed.
    I hope to improve this in the future!

  - Uses the [`ksgpu`](https://github.com/kmsmith137/ksgpu) helper library,
    but the **chord branch**, not the main branch. (I'm currently struggling
    with branch divergence between unrelated projects -- will fix this some
    day.)
    
Pirate is a real-time FRB search written in C++ / cuda / python.

  - Receives data via TCP from the "upstream" X-engine. The network protocol
    is [here](network_protocol.md). The protocol includes metadata -- the
    format is defined by [`configs/xengine/xengine_metadata_v1.yml`](../configs/xengine/xengine_metadata_v1.yml).
    Details of the protocol and metadata will change in the future.

  - Dynamic configuration: the X-engine metadata
    (see [`configs/xengine/xengine_metadata_v1.yml`](../configs/xengine/xengine_metadata_v1.yml))
    includes important parameters such as the frequency upchannelization and beam layout.
    The FRB search "dynamically" configures itself when this data is
    received from the X-engine.

    (However, not all configuration is dynamic -- the FRB search will
    have its own config file, which contains parameters that don't affect
    the X-engine.)

  - Most code is written in C++/cuda, but exported to python via pybind11.
    (The python module is named `pirate_frb`, since `pirate` was already
    taken on pypi.)

    Technical comment: we're not using free-threaded python yet. I spent 
    some time looking at it, and I just don't think it's ready for production
    yet. This is such a big constraint that we need to write most code in C++.
    As a result, the scope of the python interface is limited (mostly high-level
    configuration). Perhaps in a few years, when free-threaded python is more
    mature, we could refactor so that most of the FRB server code is in python,
    and we just call C++ for low-level compute kernels.

  - Command line interface (`python -m pirate_frb ...`).
    For more info, see `python -m pirate_frb --help` or the HTML docs.
  
  - The FRB server defines [gRPC service(s)](../grpc/frb_search.proto) for things like control, monitoring,
    and triggered data writes. All interaction with a running server is via gRPC.
    There is a python client (`pirate_frb.RpcClient`) but not a C++ client
    (not sure if we'll need one).
    
  - The FRB server stores data sent by the X-engine in a ring buffer, and defines
    RPCs for saving data to disk, via a two-stage process where data is written
    to a local high-bandwidth SSD to relieve short-term memory pressure,
    then "trickled" to an NFS server for long-term storage. I'm currently
    using Erik's `asdf-cxx` library to write files.
  
  - The FRB server does real-time RFI masking (not implemented yet but it will
    be top priority soon!) followed by dedispersion. Both of these steps are
    done on the GPU, with custom GPU kernels.

  - Simple GPU kernels are written in cuda, but complicated GPU kernels 
    are emitted by a python code generator (`pirate_frb.cuda_generator`).
    Some kernels ended up getting so complicated and coalesced that a
    hacked-up code generator was easier than C++ templates!

  - The output of the FRB search is a stream of arrays ("coarse-grained
    dedispersion outputs", to be described in future documentation).
    The plan for these arrays is as follows.
    
    Arrays will be handed off to a "downstream" process (the "grouper"),
    using a shared ring buffer in GPU memory. This could be implemented
    with `cudaIpc`, using grpc to synchronize the producer/consumer, by
    streaming integer frame offsets back and forth. None of this is implemented
    yet, but it will be soon!
