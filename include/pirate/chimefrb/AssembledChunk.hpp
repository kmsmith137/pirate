#ifndef _PIRATE_CHIMEFRB_ASSEMBLED_CHUNK_HPP
#define _PIRATE_CHIMEFRB_ASSEMBLED_CHUNK_HPP

#include <ksgpu/Array.hpp>

#include <string>
#include <memory>
#include <cstdint>

namespace pirate {
class SlabAllocator;   // defined in SlabAllocator.hpp

namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// AssembledChunk: one data file written by the OLD CHIME FRB search, in the format
// ch_frb_io calls "assembled_chunk in msgpack format".
//
// Read one with the static factory:
//
//   auto chunk = AssembledChunk::from_msgpack("chunk_01060317.msg");
//   Array<float> intensity({chunk->nfreq(), chunk->nt()}, af_uhost);
//   chunk->decode_intensity(intensity, /*apply_rfimask=*/false);
//
// The file holds uint8 data plus per-(coarse channel, time block) scales and offsets;
// decode_intensity() is what turns that back into physical units. See notes/chimefrb.md
// for the porting rules this class follows, and for how to spot-check it against the
// original ch_frb_io code.
//
// METADATA-ONLY READS. from_msgpack(filename, /*metadata_only=*/true) parses every scalar
// below, then stops before reading any array body: four small reads (~4 KB) instead of
// 17 MB, which is ~30x faster from local NVMe (0.33 ms vs 10 ms per file) and far more from
// the page cache, where a full read is essentially a 17 MB memcpy. Validation is
// unchanged -- the checks that catch a corrupt or truncated file all happen during the
// parse -- but the four arrays are left empty, and decode_intensity() / decode_weights()
// throw. This is the cheap way to survey an acquisition: which beams and FPGA ranges a
// directory covers, or whether a set of files has the uniform parameters that sharing one
// SlabAllocator requires.
//
// SCOPE: this is only the read path, and only for files that are uncompressed and
// msgpack format version 2. Anything else throws, rather than being silently mishandled.
// (Every file we have seen is version 2 and uncompressed.)

struct AssembledChunk
{
    // Where this chunk came from, and whether its arrays were read. The scalars below are
    // always present; the ARRAYS are empty if 'metadata_only' is true (class comment above).
    std::string filename;
    bool metadata_only = false;

    // Metadata, straight out of the msgpack header.
    int version = 0;
    int compression = 0;          // always 0; a nonzero value is rejected by from_msgpack()
    int beam_id = 0;
    int binning = 0;
    long nupfreq = 0;             // "upchannelization" factor: fine channels per coarse channel
    long nt_per_packet = 0;       // time samples sharing one (scale, offset) pair
    long fpga_counts_per_sample = 0;
    long nt_coarse = 0;           // == nt_per_chunk / nt_per_packet
    long nscales = 0;             // == nfreq_coarse * nt_coarse
    long ndata = 0;               // == nfreq_coarse * nupfreq * nt_per_chunk
    long nrfifreq = 0;            // frequency resolution of the RFI chain; see 'rfi_mask' below
    uint64_t fpga_begin = 0;
    uint64_t fpga_end = 0;
    uint64_t frame0_nano = 0;     // ctime in nanoseconds of FPGA count zero
    bool has_rfi_mask = false;

    // NOT stored in the file: ch_frb_io hardwires both as compile-time constants
    // (constants::nfreq_coarse_tot and constants::nt_per_assembled_chunk, both 1024).
    // We derive them from fields that ARE stored, and cross-check the result, so that
    // the reader reports what the file says rather than what ch_frb_io was compiled with.
    long nfreq_coarse = 0;        // == nscales / nt_coarse
    long nt_per_chunk = 0;        // == nt_coarse * nt_per_packet

    inline long nfreq() const { return nfreq_coarse * nupfreq; }   // fine channels
    inline long nt() const { return nt_per_chunk; }

    // The three raw arrays, all empty if metadata_only is true.
    //
    // Frequency ordering: index 0 is the HIGHEST radio frequency (800 MHz), decreasing to
    // 400 MHz. This matches pirate's own convention, so no axis flip is needed anywhere.
    //
    // A ZERO ENTRY IN 'scales' MEANS NO PACKET EVER ARRIVED for that (coarse channel, time
    // block): an assembled_chunk is zero-initialized, and the L0 encoder never writes a zero
    // scale (a fully-masked block gets scale 1, offset 0, data 0). So the fraction of zero
    // scales is packet loss, and it is the only thing that distinguishes packet loss from
    // RFI flagging. It is NOT an extra masking source: data is zero wherever scale is, so
    // decode_weights()'s sentinel test already covers those samples.
    ksgpu::Array<float> scales;      // shape (nfreq_coarse, nt_coarse)
    ksgpu::Array<float> offsets;     // shape (nfreq_coarse, nt_coarse)
    ksgpu::Array<uint8_t> data;      // shape (nfreq, nt)

    // Bit-packed RFI mask, shape (nrfifreq, nt/8) BYTES, LSB-first within each byte:
    // bit i of byte j is time sample 8*j+i. A SET bit means GOOD data -- it is a validity
    // mask, the opposite polarity from what the name suggests.
    //
    // WARNING: the frequency axis is 'nrfifreq', NOT 'nfreq'. The RFI chain ran at its own
    // resolution (nrfifreq == nfreq_coarse on every file we have, i.e. 16x coarser than
    // the data). Anything applying the mask to the data must broadcast over
    // (nfreq / nrfifreq) fine channels, computed rather than assumed.
    //
    // Empty array if has_rfi_mask is false, or if metadata_only is true. (The python
    // binding maps a maskless file to None, since an empty Array has no numpy
    // representation, and raises for a metadata-only chunk.)
    ksgpu::Array<uint8_t> rfi_mask;

    // The four arrays above are views into this buffer -- one allocation per file, with
    // each array 128-byte aligned within it. Each array's ksgpu Array::base holds a
    // reference too, so an individual array may safely outlive the AssembledChunk.
    // Empty if metadata_only is true: no buffer is allocated at all.
    std::shared_ptr<void> buffer;

    // Reads and parses one file. Throws (without allocating a buffer) if the file is
    // truncated, corrupt, compressed, or not msgpack format version 2.
    //
    // If 'metadata_only' is true, the array bodies are not read (see the class comment) and
    // 'allocator' goes unused -- in particular, a metadata-only read does not fix a fresh
    // SlabAllocator's slab size, so one can be used to CHOOSE that size.
    //
    // If 'allocator' is non-null, the buffer comes from it. Note SlabAllocator fixes its
    // slab size on first use, so a sequence of files with differing parameters cannot
    // share one allocator; the mismatch is reported as an exception naming the parameter.
    // The default (no allocator) allocates independently per call.
    static std::shared_ptr<AssembledChunk> from_msgpack(
        const std::string &filename,
        bool metadata_only = false,
        const std::shared_ptr<SlabAllocator> &allocator = std::shared_ptr<SlabAllocator>());

    // Decode to physical units, writing a full-resolution (nfreq, nt) float32 array:
    //
    //   intensity = scales[ifc,itc] * data + offsets[ifc,itc]
    //   weights   = 0 where data is 0 or 255 (the saturation sentinels), else 1
    //
    // These reproduce ch_frb_io's assembled_chunk::decode() when apply_rfimask is false,
    // which is what a comparison against the old pipeline needs.
    //
    // 'apply_rfimask' has NO DEFAULT on purpose: on real data it changes ~46% of samples,
    // so every call site should state its intent. When true, samples the rfi_mask marks
    // bad are zeroed, and both functions throw if the file carried no mask. Callers
    // should pass the SAME value to both: masking the intensity but not the weights
    // leaves a masked sample reading as a real measurement of zero.
    //
    // Requires nt_per_packet == 16 (true of every file we have); other values throw, as
    // does a metadata-only chunk, which has no data to decode.
    void decode_intensity(ksgpu::Array<float> &dst, bool apply_rfimask) const;
    void decode_weights(ksgpu::Array<float> &dst, bool apply_rfimask) const;
};


}}  // namespace pirate::chimefrb

#endif // _PIRATE_CHIMEFRB_ASSEMBLED_CHUNK_HPP
