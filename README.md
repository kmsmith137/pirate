## pirate - Perimeter Institute RAdio Transient Engine

An experimental GPU-based fast transient search, intended for use in CHORD.

**NOTE.** Currently, this code can only be compiled on a recent ubuntu
linux machine with **a physical GPU**, and the cuda toolkit installed.
I hope to improve this in the future!

Documentation:

- [Installation](notes/install.md)
- [Introduction](notes/intro.md)
- [To do list](notes/todo.md)
- [Developer notes](notes/developer.md)
- [Build system](notes/build.md)
- [Hardware](notes/hardware.md)
- [X->FRB network protocol (v1)](notes/network_protocol.md)
- [X->FRB metadata (v1)](configs/xengine/xengine_metadata_v1.yml)
- [gRPC protocol definitions](grpc/)
- [C++/cuda guidelines](notes/cpp.md)
- [Pybind11](notes/pybind11.md)
- [Thread-backed class pattern](notes/thread_backed_class.md)

Contact: Kendrick Smith <kmsmith@perimeterinstitute.ca>
