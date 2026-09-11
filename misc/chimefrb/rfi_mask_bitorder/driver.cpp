// Old-side driver for the 'rfi_mask_bitorder' spot test.
//
//    driver <input.npy> <output.npy>
//
// Input is an (nfreq, nt) float32 array of RFI-chain weights.
// Output is the (nfreq, nt/8) uint8 bitmask that rf_kernels packs from them.
//
// This runs rf_kernels::mask_counter_data exactly as rf_pipelines::mask_counter_transform
// does when it fills an assembled_chunk's rfi_mask, so the packing convention in the
// output is the one that produced every rfi_mask in every real data file.

// LIBS: -lrf_kernels

#include <rf_kernels/mask_counter.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy>\n");
        return 2;
    }

    try {
        npy::array<float> in = npy::read<float> (argv[1]);

        if (in.shape.size() != 2) {
            fprintf(stderr, "driver: expected a 2-d (nfreq, nt) input array\n");
            return 2;
        }

        int nfreq = int(in.rows());
        int nt = int(in.cols());

        if ((nt % 8) != 0) {
            fprintf(stderr, "driver: nt must be a multiple of 8\n");
            return 2;
        }

        std::vector<uint8_t> out(size_t(nfreq) * (nt/8), 0);

        rf_kernels::mask_counter_data d;
        d.nfreq = nfreq;
        d.nt_chunk = nt;
        d.in = in.data.data();
        d.istride = nt;
        d.out_bitmask = out.data();
        d.out_bmstride = nt/8;

        // Use the reference kernel, not the AVX2 one: rf_kernels' own test checks the two
        // against each other, and the reference is the readable statement of the format.
        d.slow_reference_mask_count();

        npy::write<uint8_t> (argv[2], { size_t(nfreq), size_t(nt/8) }, out.data());
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
