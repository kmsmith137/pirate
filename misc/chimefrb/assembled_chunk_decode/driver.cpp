// Old-side driver for the 'assembled_chunk_decode' spot test.
//
//    driver <input.npy> <output.npy> [force=reference|fast]
//
// LIBS: -lch_frb_io -lhdf5 -llz4 -lzmq -ljsoncpp -lcurl
//
// Input is a 1-d uint8 array holding a whole msgpack file.
// Output is a (2, nfreq, nt) float32 array: intensity, then weights.
//
// The bytes are unpacked with ch_frb_io's own msgpack adaptor and decoded with its own
// assembled_chunk::decode(), so this is the authority pirate's reader and decode kernels
// are being compared against.  Passing the file through memory rather than through a
// temp file means the adaptor -- convert<shared_ptr<assembled_chunk>> -- is exercised
// directly, which is where the format actually lives.
//
// 'force' selects which decode() ch_frb_io uses.  assembled_chunk::make() returns a
// fast_assembled_chunk (AVX2 kernels) when nt_per_packet==16 and nupfreq is even, and a
// plain assembled_chunk otherwise, and the two are separate implementations.  Production
// CHIME data has nt_per_packet=16 and nupfreq=16, so the FAST path is what actually
// decoded the real files.

#include <ch_frb_io.hpp>
#include <assembled_chunk_msgpack.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <exception>

static std::string get_param(int argc, char **argv, const char *key, const char *dflt)
{
    std::string prefix = std::string(key) + "=";
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a.compare(0, prefix.size(), prefix) == 0)
            return a.substr(prefix.size());
    }
    return dflt;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> [force=reference|fast]\n");
        return 2;
    }

    try {
        npy::array<uint8_t> in = npy::read<uint8_t> (argv[1]);
        std::string force = get_param(argc, argv, "force", "reference");

        if ((force != "reference") && (force != "fast")) {
            fprintf(stderr, "driver: force= must be 'reference' or 'fast'\n");
            return 2;
        }

        // Unpack with ch_frb_io's msgpack adaptor. Note this constructs the chunk itself
        // (via assembled_chunk::make), so we cannot choose reference-vs-fast here -- we
        // re-make the chunk below and copy into it.
        msgpack::object_handle oh = msgpack::unpack(
            reinterpret_cast<const char *> (in.data.data()), in.data.size());
        std::shared_ptr<ch_frb_io::assembled_chunk> src;
        oh.get().convert(src);

        ch_frb_io::assembled_chunk::initializer ini;
        ini.beam_id                = src->beam_id;
        ini.nupfreq                = src->nupfreq;
        ini.nrfifreq               = src->nrfifreq;
        ini.nt_per_packet          = src->nt_per_packet;
        ini.fpga_counts_per_sample = src->fpga_counts_per_sample;
        ini.binning                = src->binning;
        ini.ichunk                 = src->ichunk;
        ini.force_reference        = (force == "reference");
        ini.force_fast             = (force == "fast");

        std::shared_ptr<ch_frb_io::assembled_chunk> ch =
            ch_frb_io::assembled_chunk::make(ini);
        ch->fill_with_copy(src);

        int nfreq = ch_frb_io::constants::nfreq_coarse_tot * ch->nupfreq;
        int nt = ch_frb_io::constants::nt_per_assembled_chunk;
        size_t npix = size_t(nfreq) * size_t(nt);

        std::vector<float> out(2 * npix);
        ch->decode(out.data(), out.data() + npix, nt, nt);

        npy::write<float> (argv[2], { size_t(2), size_t(nfreq), size_t(nt) }, out.data());
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
