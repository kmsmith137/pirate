// Old-side driver for the 'assembled_chunk_encode' spot test.
//
//    driver <input.npy> <output.npy> beam_id=N nupfreq=N nt_per_packet=N \
//           fpga_counts_per_sample=N nrfifreq=N ichunk=N seed=N
//
// The input array is unused (pass a 1-element dummy).  The output is the msgpack
// serialization of a randomized assembled_chunk, as a 1-d uint8 array.
//
// This is the writer direction: ch_frb_io constructs and packs, pirate reads.  It is worth
// having on top of the decode test because msgpack has several legal encodings for the same
// value -- an integer can be a fixint, uint8, uint16, uint32 or uint64, and a binary blob
// can carry a 1-, 2- or 4-byte length -- so a reader that has only ever seen its own
// writer's choices can hardcode them without anyone noticing.

// LIBS: -lch_frb_io -lhdf5 -llz4 -lzmq -ljsoncpp -lcurl

#include <ch_frb_io.hpp>
#include <assembled_chunk_msgpack.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <exception>

// Parses a "key=value" command-line argument. Returns 'dflt' if the key is absent.
static long get_param(int argc, char **argv, const char *key, long dflt)
{
    std::string prefix = std::string(key) + "=";
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a.compare(0, prefix.size(), prefix) == 0)
            return strtol(a.c_str() + prefix.size(), nullptr, 10);
    }
    return dflt;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> [key=value ...]\n");
        return 2;
    }

    try {
        ch_frb_io::assembled_chunk::initializer ini;
        ini.beam_id                = int(get_param(argc, argv, "beam_id", 1000));
        ini.nupfreq                = int(get_param(argc, argv, "nupfreq", 1));
        ini.nt_per_packet          = int(get_param(argc, argv, "nt_per_packet", 16));
        ini.fpga_counts_per_sample = int(get_param(argc, argv, "fpga_counts_per_sample", 384));
        ini.nrfifreq               = int(get_param(argc, argv, "nrfifreq", 1024));
        ini.ichunk                 = uint64_t(get_param(argc, argv, "ichunk", 42));
        ini.binning                = int(get_param(argc, argv, "binning", 1));
        ini.frame0_nano            = uint64_t(get_param(argc, argv, "frame0_nano", 1234567890));
        ini.force_reference        = true;

        long seed = get_param(argc, argv, "seed", 137);

        std::shared_ptr<ch_frb_io::assembled_chunk> ch =
            ch_frb_io::assembled_chunk::make(ini);

        // randomize() fills scales, offsets, data and (if nrfifreq > 0) rfi_mask.
        // (Braces, not parens: 'std::mt19937 rng(uint32_t(seed))' is a function declaration.)
        uint32_t seed32 = uint32_t(seed);
        std::mt19937 rng{seed32};
        ch->randomize(rng);
        ch->has_rfi_mask = (ini.nrfifreq > 0);

        // Pack with ch_frb_io's own packer, into memory rather than a file.
        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> packer(&sbuf);
        pack_assembled_chunk(packer, ch, /*compress=*/false, /*buffer=*/nullptr);

        npy::write<uint8_t> (argv[2], { sbuf.size() },
                             reinterpret_cast<const uint8_t *> (sbuf.data()));
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
