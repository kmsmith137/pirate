    
### Chunks, batches, frames, and segments

Throughout the code:
- A "chunk" (or "time chunk") is a range of **time indices**. The chunk size (e.g. 1024 or 2048) is defined in `DedispersionConfig::time_samples_per_chunk`.
- A "minichunk" is 256 time samples. This is the cadence for sending data from the X-engine to the FRB backend, and only arises in a limited part of the code (`struct Receiver` and closely related).
- A "batch" (or "beam batch") is a range of **beam indices**. The batch size (e.g. 1,2,4) is defined in `DedispersionConfig::beams_per_batch`.
- A "frame" is a (chunk,beam) pair (not a (chunk,batch) pair!). Frames are used in `class MegaRingbuf`, and will also be used in the front-end server code and its intensity ring buffer.
- A "segment" refers to a 128-byte, memory-contiguous subset of any array in GPU memory. Segments are used in low-level GPU kernels, and data structures which are GPU kernel adjacent (e.g. `DedispersionPlan`, `MegaRingbuf`).

### File writing

When the FRB server receives data from the X-engine, it stores it in a ring buffer.
If an event is detected (this decision is made downstream by the "sifter"), the FRB
server receives an RPC, instructing it to save data to disk. There are some nontrivial
design decisions here, so I made some notes. (Most of this came out of some blackboard
brainstorming sessions with Dustin.)

  - Client sends a `write_files` RPC, and server responds immediately (without
    waiting to write to disk) with a list of filenames that are scheduled for
    writing.

  - There is a separate `subscribe_files` RPC, which establishes a persistent TCP
    connection to the server. Whenever the server writes a file, it sends the filename
    to all callers of `subscribe_files`.

  - Files are written in asdf, using Erik's `asdf-cxx` library.
  
  - The FRB server uses a two-stage write path: first, data is written
    to a local high-bandwidth SSD to relieve short-term memory pressure,
    then "trickled" to an NFS sever for long-term storage.

  - This two-stage process makes sense because the total throughput of the FRB
    server (in GB/s received from the X-engine) is less than SSD bandwidth, but
    greater than NFS bandwidth. By writing to SSD, we ensure that we never crash
    under heavy write requests (unless the SSD fills completely) since we can
    always save data quickly enough to make room for new data.

  - Idea for a future feature: use the SSDs in the FRB search nodes as a distributed
    cache for the NFS serer. Each node just has to keep the most recently written ~TB
    of data on its SSD.

  - Idea for a future feature: if the FRB node crashes, and some files have been
    written to SSD but not yet written to NFS, then write these files to NFS when
    the server is restarted.

We decided to deprioritze these future features, until we make more progress on
the downstream code, and have a better sense for which features are most useful.
  
### Hwtest and 'class Hardware'
  
  - I hacked up some python code (`class Hardware`) to query hardware
    and work out which devices are associated with each CPU.
    You can run this code with `python -m pirate_frb show_hardware`.
    This code is currently pretty terrible -- feel free to improve it.

    The `Hardware` class is used when starting the real-time server,
    to decide which hardware to associate with each of the two `FrbServer`
    instances. This is all currently done from python.

  - `hwtest`: this code was written early, when we were benchmarking test nodes.
    It runs and times parallel synthetic loads: network IO, disk IO, PCIe transfers
    between GPU host, GPU compute kernels, CPU compute kernels, host memory bandwidth.

    Here's an example networking-only run:
    ```
    # On cf05. The test will pause after "listening for TCP connections".
    python -m pirate_frb hwtest configs/hwtest/cf05_net64.yml

    # On cf00. Send to all four IP addresses on cf05
    python -m pirate_frb hwtest -s configs/hwtest/cf05_net64.yml
    ```
    See `configs/hwtest/*.yml` for more examples.

### Kendrick's unsolicited options on software engineering

  - The hardest thing to do as a programmer is to keep things simple.
    When software projects fail, it's usually a "soft failure" where overcomplexity
    starts to run away, and everyone loses motivation.
    There can be tension between avoiding this long-term failure mode,
    and short-term pressure to implement new features.
    
    It's a hard problem to solve and there's no easy answer!
    Two things that I've found helpful: (1) have blackboard discussions with other
    developers before implementing new features, and (2) expect to frequently refactor,
    when you find that an existing interface is overcomplicated/awkward.

  - Good low-level abstractions are very important (e.g. an N-dimensional Array class).
    I'm skeptical of high-level abstractions (e.g. any sort of Task virtual base class).
    It should always be possible to "opt out" of using an abstraction if it's getting in the way.

  - Avoid databases and unnecessary layers of software -- most of our
    operational problems come from unanticipated issues in these areas.

    For example, dynamic configuration (where the X-engine passes its metadata to the
    FRB server, the FRB server passes metadata to the sifter, etc.) helps avoid the use of
    central databases. Including a copy of the metadata in saved data files also helps.
  
  - Given the choice between crashing and failing gracefully, it's usually
    better to crash. Most errors are a result of misconfiguration, and in
    this case it's best to crash with a helpful error message, so that a
    human can get involved.

  - Time spent writing unit tests always pays off in the long run!

  - The most painful bugs are the ones that only happen a small fraction of the time.
    You should pay the most attention to bugs that are very unlikely (e.g. race conditions,
    corner cases), which can be a little counterintuitive.

  - I'm not a believer in engineering practices that impede "flow state", such as CI, code reviews,
    or pull requests that require waiting on others. (Needless to say, everyone should run
    tests frequently, and get feedback from others in situations where it makes sense.)

  - This one is controversial and it's okay if we disagree! I've recently become a huge convert
    to LLM-assisted programming, and I'd advise everyone to start using LLM agents heavily. I recommend
    Claude Code -- it's slower but more powerful than other tools, so it's the best choice for
    research-level work. A good way to start is by asking it to review your code for bugs and
    suggest improvements, and letting it write code as you get more comfortable. Let me know if
    you need help getting started!
