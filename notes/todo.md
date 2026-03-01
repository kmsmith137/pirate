# To do list

## Immediate concrete tasks

 - Iterate on [network protocol](network_protocol.md).
   (At minimum, need to include offsets, scales, UT1 timestamps.)

 - Iterate on metadata sent by X-engine
   ([`configs/xengine/xengine_metadata_v1.yml`](../configs/xengine/xengine_metadata_v1.yml)).
   (Note that the metadata is part of the network protocol.)

 - In particular, decide how timestamps should be represented in the
   metadata.

 - Currently there is a placeholder `FakeXEngine`, which sends garbage
   data to the FRB server. It would be really valuable to have a more
   interesting FakeXEngine, for a few reasons:

     - The networking/file-writing code currently has no unit tests!
       One approach is an "end-to-end" test, where the `FakeXEngine`
       remembers 
       Even a minimal test would be

     - Simulated pulses (link to simpulse)
     
     - Some sort of simulated RFI (
     
     - Feel free to make big changes to (or even rewrite) the current
       `FakeXEngine`. I'm looking for someone to take "ownership" of
       this code.
     
 - Feel free to dive in and make miscellaneous improvements to the code
   (random example: use spdlog for logging)
 
## Goals by end of May

Goal here is to develop an end-to-end system where we can send simulated
pulses from a `FakeXEngine`, and watch the pulses arrive in a web viewer.
No RFI for now!

There are a lot of design decisions to be made -- here are some high-level
bullet points:

  - Finish GPU dedispersion in the RFI-free case. (Currently 95% done.)
  
  - Design interface between GPU dedisperser and downstream grouping
    code. (Dustin and I have had some blackboard brainstorms here, and
    and we have a pretty concrete design in mind.)

  - Write placeholder grouping code. Since there is no RFI for now, this
    can be pretty trivial.

  - Design the gRPC interface between grouping code and thw downstream
    sifting code. (Or did Dustin do this already?)

  - There are a lot of todo items related to the sifting code, database,
    and web viewer. Dustin is keeping track of those -- please ask Dustin
    if you're interested!

## Goals for a pathfinder search

Everything needed for the CHORD pathfinder: dedisperser should handle an
RFI mask, library of RFI transforms, non-placeholder grouping code, lots
of work on sifter/databse/web.

