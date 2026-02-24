# To do list

## Version 0.1

 - Include offsets and scales in [network protocol](network_protocol.md).
   (Necessary modifications to the FRB server code are not hard, but not trivial.)

 - Decide what metadata the X-engine should send to the FRB server.
   I'm currently using a [placeholder](../configs/xengine/xengine_metadata_v1.yml).
   (Note that the metadata is part of the network protocol.)

 - In particular, decide how timestamps should be represented in the
   metadata. Modify the server code to account for timestamps and reconnections.
 
## Version 0.2

Goal here is to develop an end-to-end system where we can send simulated
pulses from a `FakeXEngine`, and watch the pulses arrive in a web viewer.
No RFI for now!

There are a lot of design decisions to be made -- here are some high-level
bullet points:

  - An interesting `FakeXEngine`. i.e. one that can send simulated pulses.
    (A secondary goal here is to test the FRB server's file-writing path,
    which currently has no unit tests!)

  - Finish GPU dedispersion in the RFI-free case. (Currently 95% done.)
  
  - Design interface between GPU dedisperser and downstream grouping
    code. (Dustin and I have had some blackboard brainstorms here, and
    and we have a pretty concrete design in mind.)

  - Write placeholder grouping code. Since there is no RFI for now, this
    can be pretty trivial.

  - Design the gRPC interface between grouping code and thw downstream
    sifting code. (Or did Dustin do this already?)

  - There are a lot of todo items related to the sifting code, database,
    and web viewer. Dustin is keeping track of those.

## Subsequent versions

Everything needed for the CHORD pathfinder: dedisperser should handle an
RFI mask, library of RFI transforms, non-placeholder grouping code, lots
of work on sifter/databse/web.

