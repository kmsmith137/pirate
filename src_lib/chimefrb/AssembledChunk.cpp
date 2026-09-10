#include "../../include/pirate/chimefrb/AssembledChunk.hpp"
#include "../../include/pirate/SlabAllocator.hpp"
#include "../../include/pirate/constants.hpp"   // bytes_per_gpu_cache_line
#include "../../include/pirate/inlines.hpp"     // align_up()

#include <ksgpu/xassert.hpp>
#include <ksgpu/mem_utils.hpp>

#include <immintrin.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace ksgpu;

namespace pirate {
namespace chimefrb {
#if 0
}}  // editor auto-indent
#endif


// Arrays are aligned to this within the buffer, matching SlabAllocator::nalign and the
// rest of pirate.
static constexpr long nalign = constants::bytes_per_gpu_cache_line;

// The msgpack header string that identifies the format, and the only version we accept.
static const char *format_magic = "assembled_chunk in msgpack format";
static constexpr int accepted_version = 2;
static constexpr int nitems = 21;   // version-2 files; version 1 had 17, and is rejected


// -------------------------------------------------------------------------------------------------
//
// Hand-rolled msgpack reader.
//
// We parse the 21 items by hand rather than linking libmsgpack: the format is fixed, will
// never change again, and parsing it ourselves is what lets each 'bin' body be read
// straight into its final aligned home instead of being copied out of a staging buffer.
//
// WATCH THE ENDIANNESS. msgpack's scalar integers are BIG-endian, but the 'bin' bodies are
// raw little-endian x86 memory (ch_frb_io memcpy's its float arrays straight out). So the
// header needs byte swaps and the bodies must not be touched.


// Cursor over a byte buffer. Every read is bounds-checked, because these bytes come from a
// file that nothing in this process wrote.
struct Cursor
{
    const uint8_t *p = nullptr;
    const uint8_t *end = nullptr;
    const string &filename;

    Cursor(const uint8_t *p_, long nbytes, const string &filename_) :
        p(p_), end(p_ + nbytes), filename(filename_) { }

    [[noreturn]] void fail(const string &msg) const
    {
        throw runtime_error("chimefrb: " + filename + ": " + msg);
    }

    void need(long n, const char *what) const
    {
        if (end - p < n)
            fail(string("truncated while reading ") + what);
    }

    uint8_t u8(const char *what)  { need(1, what); return *p++; }
    uint16_t be16(const char *what) { need(2, what); uint16_t v = (uint16_t(p[0]) << 8) | p[1]; p += 2; return v; }
    uint32_t be32(const char *what) { need(4, what); uint32_t v = 0; for (int i = 0; i < 4; i++) v = (v << 8) | p[i]; p += 4; return v; }
    uint64_t be64(const char *what) { need(8, what); uint64_t v = 0; for (int i = 0; i < 8; i++) v = (v << 8) | p[i]; p += 8; return v; }

    // Any msgpack integer, signed or unsigned. ch_frb_io packs C ints, so a small value
    // arrives as a fixint and a large one as uint16/uint32; negatives are possible in
    // principle and are accepted here so that a legal file is never rejected.
    long get_int(const char *what)
    {
        uint8_t tag = u8(what);
        if (tag < 0x80) return tag;                        // positive fixint
        if (tag >= 0xe0) return long(int8_t(tag));         // negative fixint
        switch (tag) {
            case 0xcc: return u8(what);                    // uint8
            case 0xcd: return be16(what);                  // uint16
            case 0xce: return be32(what);                  // uint32
            case 0xcf: return long(be64(what));            // uint64
            case 0xd0: return long(int8_t(u8(what)));      // int8
            case 0xd1: return long(int16_t(be16(what)));   // int16
            case 0xd2: return long(int32_t(be32(what)));   // int32
            case 0xd3: return long(int64_t(be64(what)));   // int64
        }
        stringstream ss;
        ss << "expected an integer for " << what << ", got msgpack tag 0x"
           << std::hex << int(tag);
        fail(ss.str());
    }

    uint64_t get_uint(const char *what)
    {
        long v = get_int(what);
        if (v < 0) fail(string("expected a nonnegative value for ") + what);
        return uint64_t(v);
    }

    bool get_bool(const char *what)
    {
        uint8_t tag = u8(what);
        if (tag == 0xc2) return false;
        if (tag == 0xc3) return true;
        stringstream ss;
        ss << "expected a bool for " << what << ", got msgpack tag 0x" << std::hex << int(tag);
        fail(ss.str());
    }

    string get_str(const char *what)
    {
        uint8_t tag = u8(what);
        long n;
        if ((tag & 0xe0) == 0xa0) n = tag & 0x1f;          // fixstr
        else if (tag == 0xd9) n = u8(what);                // str8
        else if (tag == 0xda) n = be16(what);              // str16
        else {
            stringstream ss;
            ss << "expected a string for " << what << ", got msgpack tag 0x" << std::hex << int(tag);
            fail(ss.str());
        }
        need(n, what);
        string ret((const char *)p, n);
        p += n;
        return ret;
    }

    // Reads a 'bin' header and returns the body length, leaving the cursor at the body.
    // Does NOT consume the body -- callers read it separately, into an aligned destination.
    long get_bin_header(const char *what)
    {
        uint8_t tag = u8(what);
        if (tag == 0xc4) return u8(what);        // bin8
        if (tag == 0xc5) return be16(what);      // bin16
        if (tag == 0xc6) return be32(what);      // bin32
        stringstream ss;
        ss << "expected a binary blob for " << what << ", got msgpack tag 0x" << std::hex << int(tag);
        fail(ss.str());
    }

    long get_array_header(const char *what)
    {
        uint8_t tag = u8(what);
        if ((tag & 0xf0) == 0x90) return tag & 0x0f;   // fixarray
        if (tag == 0xdc) return be16(what);            // array16
        if (tag == 0xdd) return be32(what);            // array32
        stringstream ss;
        ss << "expected an array for " << what << ", got msgpack tag 0x" << std::hex << int(tag);
        fail(ss.str());
    }
};


// -------------------------------------------------------------------------------------------------
//
// File reading.


// RAII file descriptor. (pirate::File in file_utils.hpp is currently hwtest-only and does
// not do positioned reads, so this local helper is simpler than adapting it.)
struct Fd
{
    int fd = -1;
    explicit Fd(int fd_) : fd(fd_) { }
    ~Fd() { if (fd >= 0) ::close(fd); }
    Fd(const Fd &) = delete;
    Fd &operator=(const Fd &) = delete;
};


// Positioned read of exactly 'nbytes'. Loops, since a short read is legal.
static void pread_exact(int fd, void *dst, long nbytes, long offset, const string &filename,
                        const char *what)
{
    long done = 0;
    while (done < nbytes) {
        ssize_t n = ::pread(fd, (char *)dst + done, nbytes - done, offset + done);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            throw runtime_error("chimefrb: " + filename + ": read failed while reading "
                                + what + ": " + strerror(errno));
        }
        if (n == 0)
            throw runtime_error("chimefrb: " + filename + ": unexpected end of file while reading "
                                + what);
        done += n;
    }
}


// -------------------------------------------------------------------------------------------------
//
// Parsing.
//
// The read sequence is not the obvious one, and the reason is worth stating before the
// code: the slab size depends on 'nrfifreq', which is item 18, i.e. AFTER the 16 MB data
// blob. And a small prefix read does NOT reach items 15 and 16 either -- each 'bin' header
// is immediately followed by its multi-hundred-kilobyte body, so on a real file the three
// bin headers sit at byte offsets 79, 262228 and 524377.
//
// So we read a prefix, COMPUTE the rest of the layout from fields in it (each body length
// is predicted by 'nscales' or by the data_size of item 3), verify the computed header
// positions, read the tail, and only then allocate. Every check happens before the
// allocation, so a corrupt file never takes a slab from the pool.


// File byte offsets of the four 'bin' bodies, plus the offset where items 17..20 begin.
// All computed from the prefix; all verified before use.
struct FileLayout
{
    long scales_body = 0;
    long offsets_body = 0;
    long data_body = 0;
    long rfi_mask_body = 0;
    long item15_hdr = 0;
    long item16_hdr = 0;
    long tail = 0;          // where item 17 (frame0_nano) begins
    long data_nbytes = 0;   // item 3, == ndata for an uncompressed file
    long scales_nbytes = 0; // 4 * nscales
};


// Parses items 0..13 plus the item-14 bin header out of a prefix buffer, filling in the
// scalar metadata and enough of 'layout' to locate everything else.
static void parse_prefix(AssembledChunk &c, FileLayout &layout, const uint8_t *prefix,
                         long prefix_nbytes, const string &filename)
{
    Cursor cur(prefix, prefix_nbytes, filename);

    long n = cur.get_array_header("the toplevel array");
    if (n != nitems) {
        stringstream ss;
        ss << "expected a " << nitems << "-item msgpack array (format version "
           << accepted_version << "), got " << n << " items";
        if (n == 17)
            ss << ". This is a format version 1 file, which is not supported";
        cur.fail(ss.str());
    }

    string magic = cur.get_str("the header string");
    if (magic != format_magic)
        cur.fail("not an assembled_chunk msgpack file (header string was \"" + magic + "\")");

    c.version = int(cur.get_int("the version"));
    if (c.version != accepted_version) {
        stringstream ss;
        ss << "msgpack format version " << c.version << " is not supported (only version "
           << accepted_version << " is)";
        cur.fail(ss.str());
    }

    c.compression = int(cur.get_int("the compression flag"));
    if (c.compression != 0) {
        stringstream ss;
        ss << "compressed msgpack files are not supported (compression=" << c.compression << ")";
        cur.fail(ss.str());
    }

    layout.data_nbytes          = cur.get_int("data_size");
    c.beam_id                   = int(cur.get_int("beam_id"));
    c.nupfreq                   = cur.get_int("nupfreq");
    c.nt_per_packet             = cur.get_int("nt_per_packet");
    c.fpga_counts_per_sample    = cur.get_int("fpga_counts_per_sample");
    c.nt_coarse                 = cur.get_int("nt_coarse");
    c.nscales                   = cur.get_int("nscales");
    c.ndata                     = cur.get_int("ndata");
    c.fpga_begin                = cur.get_uint("fpga_begin");
    uint64_t fpga_n             = cur.get_uint("the fpga count span");
    c.binning                   = int(cur.get_int("binning"));
    c.fpga_end = c.fpga_begin + fpga_n;

    // Sanity-check the shape parameters against each other before using any of them to
    // compute an offset or a size. Everything downstream assumes these hold.
    if ((c.nupfreq <= 0) || (c.nt_per_packet <= 0) || (c.nt_coarse <= 0) ||
        (c.nscales <= 0) || (c.ndata <= 0) || (c.binning <= 0)) {
        stringstream ss;
        ss << "nonpositive shape parameter: nupfreq=" << c.nupfreq
           << ", nt_per_packet=" << c.nt_per_packet << ", nt_coarse=" << c.nt_coarse
           << ", nscales=" << c.nscales << ", ndata=" << c.ndata << ", binning=" << c.binning;
        cur.fail(ss.str());
    }

    // Derive the two shape constants that ch_frb_io hardwires (see the header file).
    if ((c.nscales % c.nt_coarse) != 0) {
        stringstream ss;
        ss << "nscales (" << c.nscales << ") is not a multiple of nt_coarse (" << c.nt_coarse << ")";
        cur.fail(ss.str());
    }
    c.nfreq_coarse = c.nscales / c.nt_coarse;
    c.nt_per_chunk = c.nt_coarse * c.nt_per_packet;

    if (c.ndata != c.nfreq_coarse * c.nupfreq * c.nt_per_chunk) {
        stringstream ss;
        ss << "ndata (" << c.ndata << ") != nfreq_coarse*nupfreq*nt_per_chunk ("
           << c.nfreq_coarse << "*" << c.nupfreq << "*" << c.nt_per_chunk << " = "
           << c.nfreq_coarse * c.nupfreq * c.nt_per_chunk << ")";
        cur.fail(ss.str());
    }
    if (fpga_n != uint64_t(c.nt_per_chunk) * uint64_t(c.fpga_counts_per_sample) * uint64_t(c.binning)) {
        stringstream ss;
        ss << "fpga_end-fpga_begin (" << fpga_n << ") != nt_per_chunk*fpga_counts_per_sample*binning ("
           << c.nt_per_chunk << "*" << c.fpga_counts_per_sample << "*" << c.binning << ")";
        cur.fail(ss.str());
    }
    // Uncompressed, so item 3 must agree with item 10. (A compressed file was rejected above.)
    if (layout.data_nbytes != c.ndata) {
        stringstream ss;
        ss << "uncompressed file, but data_size (" << layout.data_nbytes
           << ") != ndata (" << c.ndata << ")";
        cur.fail(ss.str());
    }

    layout.scales_nbytes = 4 * c.nscales;

    // Item 14: the first bin header IS inside the prefix, so parse it normally.
    long got = cur.get_bin_header("the scales array");
    if (got != layout.scales_nbytes) {
        stringstream ss;
        ss << "scales blob is " << got << " bytes, expected 4*nscales = " << layout.scales_nbytes;
        cur.fail(ss.str());
    }
    layout.scales_body = cur.p - prefix;

    // The item-15 header sits immediately after the scales body, so we can locate it --
    // but not what follows it, because a bin header is 2, 3 or 5 bytes depending on how
    // the writer encoded the length. resolve_mid_headers() reads the two headers and
    // fills in the rest of the layout.
    layout.item15_hdr = layout.scales_body + layout.scales_nbytes;
}


// Reads the two 'bin' headers that the prefix could not reach, and fills in the file
// offsets that depend on them.
//
// Their WIDTH cannot be assumed: msgpack encodes a bin length in 1, 2 or 4 bytes
// (tags 0xc4/0xc5/0xc6), so a header is 2, 3 or 5 bytes. Real files use bin32 for both,
// since the bodies are hundreds of kilobytes, but a small synthetic chunk -- exactly what
// the unit tests generate -- does not. So read each header, take its actual width, and
// only then compute where the next one is.
static void resolve_mid_headers(int fd, FileLayout &layout, long file_nbytes,
                                const string &filename)
{
    struct Item { long hdr_off; long expect_nbytes; const char *what; };
    Item items[2] = {
        { layout.item15_hdr, layout.scales_nbytes, "the offsets array" },
        { 0,                 layout.data_nbytes,   "the data array" },   // hdr_off filled in below
    };
    long body_off[2] = { 0, 0 };

    for (int i = 0; i < 2; i++) {
        if ((items[i].hdr_off < 0) || (items[i].hdr_off >= file_nbytes)) {
            stringstream ss;
            ss << items[i].what << ": computed header offset " << items[i].hdr_off
               << " is outside the file (" << file_nbytes << " bytes)";
            throw runtime_error("chimefrb: " + filename + ": " + ss.str());
        }

        // A bin header is at most 5 bytes; read a few more so the cursor never runs dry.
        uint8_t buf[8];
        long navail = min(long(sizeof(buf)), file_nbytes - items[i].hdr_off);
        pread_exact(fd, buf, navail, items[i].hdr_off, filename, items[i].what);

        Cursor cur(buf, navail, filename);
        long got = cur.get_bin_header(items[i].what);
        if (got != items[i].expect_nbytes) {
            stringstream ss;
            ss << items[i].what << ": blob is " << got << " bytes at file offset "
               << items[i].hdr_off << ", expected " << items[i].expect_nbytes;
            cur.fail(ss.str());
        }
        body_off[i] = items[i].hdr_off + (cur.p - buf);

        if (i == 0)
            items[1].hdr_off = body_off[0] + layout.scales_nbytes;
    }

    layout.offsets_body = body_off[0];
    layout.item16_hdr = items[1].hdr_off;
    layout.data_body = body_off[1];
    layout.tail = layout.data_body + layout.data_nbytes;
}


// Reads and parses items 17..20 (which live after the data blob), and computes where the
// rfi_mask body is.
static void parse_tail(AssembledChunk &c, FileLayout &layout, int fd, long file_nbytes,
                       const string &filename)
{
    // Items 17..20 are: uint64 (<=9 bytes), int (<=5), bool (1), bin header (<=5).
    uint8_t buf[32];
    long avail = min(long(sizeof(buf)), file_nbytes - layout.tail);
    if (avail <= 0)
        throw runtime_error("chimefrb: " + filename + ": file is too short to contain items 17-20");
    pread_exact(fd, buf, avail, layout.tail, filename, "the trailing metadata");

    Cursor cur(buf, avail, filename);
    c.frame0_nano = cur.get_uint("frame0_nano");
    c.nrfifreq = cur.get_int("nrfifreq");
    c.has_rfi_mask = cur.get_bool("has_rfi_mask");
    long mask_nbytes = cur.get_bin_header("the rfi_mask array");
    layout.rfi_mask_body = layout.tail + (cur.p - buf);

    if (c.nrfifreq < 0)
        cur.fail("negative nrfifreq");

    // ch_frb_io writes a zero-length blob when the mask was never filled in.
    long expect = c.has_rfi_mask ? (c.nrfifreq * c.nt_per_chunk / 8) : 0;
    if (c.has_rfi_mask && ((c.nt_per_chunk % 8) != 0))
        cur.fail("has_rfi_mask is set, but nt_per_chunk is not a multiple of 8");
    if (mask_nbytes != expect) {
        stringstream ss;
        ss << "rfi_mask blob is " << mask_nbytes << " bytes, expected " << expect
           << " (has_rfi_mask=" << (c.has_rfi_mask ? "true" : "false")
           << ", nrfifreq=" << c.nrfifreq << ")";
        cur.fail(ss.str());
    }
    if (!c.has_rfi_mask)
        c.nrfifreq = 0;   // no mask, so the resolution is meaningless; keep it unambiguous

    // The strongest single check we have: the computed layout must account for the whole
    // file, exactly. This catches an offset slip anywhere upstream of it.
    long computed = layout.rfi_mask_body + mask_nbytes;
    if (computed != file_nbytes) {
        stringstream ss;
        ss << "computed file layout ends at byte " << computed << ", but the file is "
           << file_nbytes << " bytes";
        cur.fail(ss.str());
    }
}


// -------------------------------------------------------------------------------------------------
//
// Buffer layout and allocation.


// Byte offsets of the four arrays within the buffer, each aligned to 'nalign'.
struct SlabLayout
{
    long scales = 0;
    long offsets = 0;
    long data = 0;
    long rfi_mask = 0;
    long nbytes = 0;
};


static SlabLayout get_slab_layout(const AssembledChunk &c, long mask_nbytes)
{
    SlabLayout s;
    s.scales = 0;
    s.offsets = align_up(s.scales + 4*c.nscales, nalign);
    s.data = align_up(s.offsets + 4*c.nscales, nalign);
    s.rfi_mask = align_up(s.data + c.ndata, nalign);
    s.nbytes = align_up(s.rfi_mask + mask_nbytes, nalign);
    return s;
}


// Initializes one of the chunk's arrays as a 2-d view into the buffer at 'byte_offset'.
// (Array's default constructor leaves the caller to fill in every member; see the
// "non-standard situations" note in ksgpu/Array.hpp.)
template<typename T>
static void init_array(Array<T> &arr, const shared_ptr<void> &buffer, long byte_offset,
                       long m, long n, int aflags)
{
    arr.data = (T *)((char *)buffer.get() + byte_offset);
    arr.ndim = 2;
    arr.shape[0] = m;
    arr.shape[1] = n;
    arr.size = m * n;
    arr.strides[0] = n;
    arr.strides[1] = 1;
    arr.dtype = Dtype::native<T>();
    arr.aflags = aflags;
    arr.base = buffer;
    arr.check_invariants("pirate::chimefrb::AssembledChunk");
}


// -------------------------------------------------------------------------------------------------
//
// AssembledChunk::from_msgpack()


shared_ptr<AssembledChunk> AssembledChunk::from_msgpack(const string &filename,
                                                        const shared_ptr<SlabAllocator> &allocator)
{
    Fd f(::open(filename.c_str(), O_RDONLY));
    if (f.fd < 0)
        throw runtime_error("chimefrb: " + filename + ": open failed: " + strerror(errno));

    struct stat st;
    if (::fstat(f.fd, &st) < 0)
        throw runtime_error("chimefrb: " + filename + ": fstat failed: " + strerror(errno));
    long file_nbytes = st.st_size;

    auto chunk = make_shared<AssembledChunk> ();
    FileLayout layout;

    // Step 1: a fixed prefix, big enough for items 0..13 plus the item-14 bin header.
    // (Those total well under 100 bytes on any legal file; 4 KB is one page and leaves
    // room for the widest possible integer encodings.)
    uint8_t prefix[4096];
    long prefix_nbytes = min(long(sizeof(prefix)), file_nbytes);
    if (prefix_nbytes <= 0)
        throw runtime_error("chimefrb: " + filename + ": file is empty");
    pread_exact(f.fd, prefix, prefix_nbytes, 0, filename, "the msgpack header");
    parse_prefix(*chunk, layout, prefix, prefix_nbytes, filename);

    // Steps 2-4: verify the computed positions of the two bin headers the prefix could not
    // reach, then read the tail (which is what tells us nrfifreq, hence the buffer size),
    // then check the computed layout against the file size.
    resolve_mid_headers(f.fd, layout, file_nbytes, filename);
    parse_tail(*chunk, layout, f.fd, file_nbytes, filename);

    long mask_nbytes = chunk->has_rfi_mask ? (chunk->nrfifreq * chunk->nt_per_chunk / 8) : 0;
    SlabLayout slab = get_slab_layout(*chunk, mask_nbytes);

    // Step 5: allocate. Everything above this point is validation, so a corrupt file never
    // takes a slab from the pool.
    int aflags = af_uhost;
    if (allocator) {
        aflags = allocator->aflags;
        try {
            chunk->buffer = allocator->get_slab(slab.nbytes);
        }
        catch (const exception &e) {
            // SlabAllocator fixes its slab size on first use, so a file whose parameters
            // differ from the first one it saw cannot be served. Say which parameters we
            // asked for -- the raw size mismatch is hard to act on.
            stringstream ss;
            ss << "chimefrb: " << filename << ": could not get a " << slab.nbytes
               << "-byte slab (nupfreq=" << chunk->nupfreq
               << ", nt_per_packet=" << chunk->nt_per_packet
               << ", nrfifreq=" << chunk->nrfifreq
               << "). A SlabAllocator serves one slab size, so all files sharing one"
               << " allocator must have identical parameters. Underlying error: " << e.what();
            throw runtime_error(ss.str());
        }
        if (!chunk->buffer)
            throw runtime_error("chimefrb: " + filename + ": SlabAllocator returned an empty slab");
    }
    else {
        // Align by hand: af_alloc() makes no promise beyond malloc's alignment. The
        // aliasing shared_ptr keeps the real allocation alive while 'buffer' points at
        // the first aligned byte inside it.
        shared_ptr<void> raw = _af_alloc(Dtype::native<uint8_t>(), slab.nbytes + nalign, aflags);
        uintptr_t base = (uintptr_t)raw.get();
        uintptr_t aligned = (base + nalign - 1) & ~(uintptr_t)(nalign - 1);
        chunk->buffer = shared_ptr<void> (raw, (void *)aligned);
    }

    // Step 6: read the four bodies straight into their aligned slots. This is the whole
    // point of parsing by hand -- the 16 MB payload is never copied twice.
    //
    // (SlabAllocator's free list is intrusive and clobbers the first 16 bytes of a
    // returned slab. 'scales' sits at buffer offset 0, and the pread below overwrites it
    // in full, so nothing stale survives.)
    char *buf = (char *)chunk->buffer.get();
    pread_exact(f.fd, buf + slab.scales, layout.scales_nbytes, layout.scales_body,
                filename, "the scales array");
    pread_exact(f.fd, buf + slab.offsets, layout.scales_nbytes, layout.offsets_body,
                filename, "the offsets array");
    pread_exact(f.fd, buf + slab.data, chunk->ndata, layout.data_body,
                filename, "the data array");
    if (mask_nbytes > 0)
        pread_exact(f.fd, buf + slab.rfi_mask, mask_nbytes, layout.rfi_mask_body,
                    filename, "the rfi_mask array");

    init_array(chunk->scales, chunk->buffer, slab.scales, chunk->nfreq_coarse, chunk->nt_coarse, aflags);
    init_array(chunk->offsets, chunk->buffer, slab.offsets, chunk->nfreq_coarse, chunk->nt_coarse, aflags);
    init_array(chunk->data, chunk->buffer, slab.data, chunk->nfreq(), chunk->nt(), aflags);
    if (mask_nbytes > 0)
        init_array(chunk->rfi_mask, chunk->buffer, slab.rfi_mask, chunk->nrfifreq,
                   chunk->nt_per_chunk / 8, aflags);

    return chunk;
}


// -------------------------------------------------------------------------------------------------
//
// Decode kernels (AVX2).
//
// nt_per_packet == 16 means each (scale, offset) pair governs exactly 16 consecutive time
// samples, i.e. two __m256 vectors, so the inner loop is flat with no cross-block handling.
// One _mm_loadu_si128() fetches all 16 bytes and _mm256_cvtepu8_epi32() widens each half.
//
// These are AVX2 rather than AVX-512 deliberately. AVX2 is inside pirate's baseline
// (-march=x86-64-v3), so no target attribute or runtime CPU check is needed and the binary
// still runs anywhere pirate does. AVX-512 measured ~10% faster with bit-identical output
// (both are one FMA per sample), which does not buy back the portability.


// Shared argument checking for the two decode entry points.
static void check_decode_args(const AssembledChunk &c, const Array<float> &dst,
                              bool apply_rfimask, const char *where)
{
    if (c.nt_per_packet != 16) {
        stringstream ss;
        ss << where << ": the decode kernels require nt_per_packet == 16, got "
           << c.nt_per_packet << ". (The parser accepts any value; only the kernels are"
           << " specialized. Supporting a multiple of 16 is an extra inner loop.)";
        throw runtime_error(ss.str());
    }
    if (!c.data.data)
        throw runtime_error(string(where) + ": chunk has no data (was it default-constructed?)");

    xassert_shape_eq(dst, ({c.nfreq(), c.nt()}));
    xassert(dst.on_host());
    xassert_eq(dst.strides[1], 1);   // rows contiguous; row stride may be arbitrary

    if (apply_rfimask) {
        if (!c.has_rfi_mask)
            throw runtime_error(string(where) + ": apply_rfimask=true, but this file carries"
                                " no RFI mask (has_rfi_mask is false)");
        xassert_eq(c.nfreq() % c.nrfifreq, 0);
    }
}


// Expands 8 mask bits (LSB-first) into an 8-lane float select mask: all-ones where the bit
// is SET, i.e. where the sample is GOOD.
static inline __m256 rfimask_bits_to_ps(uint8_t bits)
{
    const __m256i sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256i v = _mm256_and_si256(_mm256_set1_epi32(bits), sel);
    return _mm256_castsi256_ps(_mm256_cmpeq_epi32(v, sel));
}


void AssembledChunk::decode_intensity(Array<float> &dst, bool apply_rfimask) const
{
    check_decode_args(*this, dst, apply_rfimask, "AssembledChunk::decode_intensity()");

    const long nt_ = nt();
    const long dstride = dst.strides[0];
    const long fdiv = apply_rfimask ? (nfreq() / nrfifreq) : 1;   // fine channels per mask row
    const long mask_stride = apply_rfimask ? rfi_mask.strides[0] : 0;

    for (long ifc = 0; ifc < nfreq_coarse; ifc++) {
        const float *sc = scales.data + ifc * nt_coarse;
        const float *of = offsets.data + ifc * nt_coarse;

        for (long iu = 0; iu < nupfreq; iu++) {
            const long ifine = ifc * nupfreq + iu;
            const uint8_t *src = data.data + ifine * nt_;
            float *dp = dst.data + ifine * dstride;
            const uint8_t *mrow = apply_rfimask ? (rfi_mask.data + (ifine / fdiv) * mask_stride)
                                                : nullptr;

            for (long itc = 0; itc < nt_coarse; itc++) {
                __m128i b = _mm_loadu_si128((const __m128i *)(src + 16*itc));
                __m256 vs = _mm256_set1_ps(sc[itc]);
                __m256 vo = _mm256_set1_ps(of[itc]);
                __m256 y0 = _mm256_fmadd_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b)), vo);
                __m256 y1 = _mm256_fmadd_ps(vs, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b,8))), vo);

                if (apply_rfimask) {
                    // 16 samples per block == exactly 2 mask bytes, since nt_per_packet==16.
                    y0 = _mm256_and_ps(y0, rfimask_bits_to_ps(mrow[2*itc]));
                    y1 = _mm256_and_ps(y1, rfimask_bits_to_ps(mrow[2*itc + 1]));
                }

                _mm256_storeu_ps(dp + 16*itc, y0);
                _mm256_storeu_ps(dp + 16*itc + 8, y1);
            }
        }
    }
}


void AssembledChunk::decode_weights(Array<float> &dst, bool apply_rfimask) const
{
    check_decode_args(*this, dst, apply_rfimask, "AssembledChunk::decode_weights()");

    const long nt_ = nt();
    const long dstride = dst.strides[0];
    const long fdiv = apply_rfimask ? (nfreq() / nrfifreq) : 1;
    const long mask_stride = apply_rfimask ? rfi_mask.strides[0] : 0;

    const __m256i zero = _mm256_setzero_si256();
    const __m256i c255 = _mm256_set1_epi32(255);
    const __m256 one = _mm256_set1_ps(1.0f);

    for (long ifine = 0; ifine < nfreq(); ifine++) {
        const uint8_t *src = data.data + ifine * nt_;
        float *dp = dst.data + ifine * dstride;
        const uint8_t *mrow = apply_rfimask ? (rfi_mask.data + (ifine / fdiv) * mask_stride)
                                            : nullptr;

        for (long itc = 0; itc < nt_coarse; itc++) {
            __m128i b = _mm_loadu_si128((const __m128i *)(src + 16*itc));
            __m256i x0 = _mm256_cvtepu8_epi32(b);
            __m256i x1 = _mm256_cvtepu8_epi32(_mm_srli_si128(b, 8));

            // 0 and 255 are the saturation sentinels; anything else is good data.
            __m256i bad0 = _mm256_or_si256(_mm256_cmpeq_epi32(x0, zero), _mm256_cmpeq_epi32(x0, c255));
            __m256i bad1 = _mm256_or_si256(_mm256_cmpeq_epi32(x1, zero), _mm256_cmpeq_epi32(x1, c255));
            __m256 w0 = _mm256_andnot_ps(_mm256_castsi256_ps(bad0), one);
            __m256 w1 = _mm256_andnot_ps(_mm256_castsi256_ps(bad1), one);

            if (apply_rfimask) {
                w0 = _mm256_and_ps(w0, rfimask_bits_to_ps(mrow[2*itc]));
                w1 = _mm256_and_ps(w1, rfimask_bits_to_ps(mrow[2*itc + 1]));
            }

            _mm256_storeu_ps(dp + 16*itc, w0);
            _mm256_storeu_ps(dp + 16*itc + 8, w1);
        }
    }
}


float AssembledChunk::fraction_missing() const
{
    if (!scales.data || (nscales <= 0))
        return 0.0f;

    long nzero = 0;
    for (long i = 0; i < nscales; i++)
        if (scales.data[i] == 0.0f)
            nzero++;

    return float(double(nzero) / double(nscales));
}


}}  // namespace pirate::chimefrb
