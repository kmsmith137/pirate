## pirate - Perimeter Institute RAdio Transient Engine

An experimental GPU-based fast transient search, intended for use in CHORD.

Some notes/caveats:

  - This repo contains real-time server code (network receive, ring buffering,
    file-writing RPCs) and GPU dedispersion code, but currently the **real-time
    server does not call the GPU dedisperser**! Working on this is my top
    priority.

  - Currently, this code can only be compiled on a recent ubuntu
    linux machine with **a physical GPU**, and the cuda toolkit installed.
    I hope to improve this in the future!

  - Uses the [`ksgpu`](https://github.com/kmsmith137/ksgpu/tree/chord) helper library,
    but the **chord branch**, not the main branch. (I'm currently struggling
    with branch divergence between unrelated projects -- will fix this some
    day.)

Documentation: HTML docs are at [kmsmith137.github.io/pirate](https://kmsmith137.github.io/pirate/)
but may not be completely up-to-date. You can generate up-to-date docs with `make docs` and `make docs-serve`,
or browse the `.md` files below:

- [Installation](notes/install.md)
- [Introduction](notes/intro.md)
- [To do list](notes/todo.md)
- [Developer notes](notes/developer.md)
- [Build system](notes/build.md)
- [Hardware](notes/hardware.md)
- [X->FRB network protocol (v1)](notes/network_protocol.md)
- [X->FRB metadata (v1)](configs/xengine/xengine_metadata_v1.yml)
- [gRPC protocol definitions](grpc/)

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
