# X->FRB network protocol (v2)

This document describes the network protocol used in the CHORD radio telescope, to send beamformed
intensity data from the X-engine (a 64-node cluster) to the FRB search backend (a 14-node cluster).

The intensity data is a 3-d "data cube" with axes `(frequency channel, beam, time)`.
Each X-engine sends a subset of the frequency channels (for all beams and times).
Each FRB search node receives a subset of the beams (for all frequency channels and times).
The data is sent over persistent TCP connections (one connection per sender/receiver node pair).
The network protocol for each of these connections is as follows:

- Everything is little-endian. (You can assume that all code is running on a little-endian architecture.)

- A persistent TCP connection is opened. The first 4 bytes are `0xf4bf4b02` where the `02` is the protocol version number.

- The next 4 bytes are a 32-bit integer string length, including one or more bytes of zero padding.

- A zero-terminated ascii string follows, containing metadata in the format defined by [`configs/xengine/xengine_metadata_v1.yml`](../configs/xengine/xengine_metadata_v2.yml). There is a C++ class `XEngineMetadata` for parsing this string.

  Note that the metadata includes `freq_channels` and `nbeams`. Here, `nbeams` is the (receiver-dependent) number of 
  beams sent to the FRB search node, and `nfreq = len(freq_channels)` is the (sender-dependent) number of frequency 
  channels sent by the X-engine node.

- Next a sequence of "minichunks" is sent. Each minichunk represents 256 time samples of intensity data.
  It consists of the following data, sent "back-to-back" with no padding or alignment:

    - A `uint64` FPGA sequence number (seq) corresponding to the beginning of the minichunk,.

    - An `(nbeams, nfreq, 2)` float16 array, where the length-2 axis is `{scales,offsets}`.
    
    - An `(nbeams, nfreq, 256)` int4 array, containing intensity data.
      The value (-8) indicates "this sample is masked". 
      We pack two int4s into a byte as (`(x[1] << 4) | x[0]`).
