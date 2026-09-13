// Old-side driver for the 'tree_gridding' spot test.
//
//    driver <in.npy> <out_prod.npy> <out_ref.npy> freq_lo=<MHz> freq_hi=<MHz> tree_size=<int>
//
// Runs bonsai::gridding, the first step of the old dedisperser: it rebins frequency
// channels onto "tree" channels equally spaced in freq^(-2), so that a dispersed pulse
// becomes a straight line. Two outputs, from bonsai's two implementations of the same
// weights -- the production (AVX2) kernel and its unoptimized reference path -- so that
// the test can see how far two implementations of the SAME operation sit apart.
//
//    in        (nfreq, ntime)      float32   frequency channels, in BONSAI's order: row 0
//                                            is the TOP of the band (freq_hi)
//    out_prod  (tree_size, ntime)  float32   gridding::grid_data_to_tree()
//    out_ref   (tree_size, ntime)  float32   gridding::reference_grid_data_to_tree()
//
// The gridding is constructed with beta_fid = 0 (no spectral-index weighting) and
// nups = 1 (no time upsampling): the values every CHIME production config uses, and the
// only ones for which the operation has a pirate counterpart. ntime must be a multiple
// of 16 (bonsai's constants::floats_per_cache_line), which bonsai checks itself.
//
// LIBS: -lbonsai

#include <bonsai_internals.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: driver <in.npy> <out_prod.npy> <out_ref.npy> freq_lo=<MHz>"
                " freq_hi=<MHz> tree_size=<int>\n");
        return 2;
    }

    try {
        npy::cmdline cl(argc, argv, 1, 2);

        double freq_lo = cl.kv.get_double("freq_lo", 0.0);
        double freq_hi = cl.kv.get_double("freq_hi", 0.0);
        long tree_size = cl.kv.get_long("tree_size", 0);

        if ((freq_lo <= 0.0) || (freq_hi <= freq_lo) || (tree_size <= 0)) {
            fprintf(stderr, "driver: need 0 < freq_lo < freq_hi and tree_size > 0\n");
            return 2;
        }

        npy::array<float> in = npy::read<float> (cl.inputs[0]);

        if (in.shape.size() != 2) {
            fprintf(stderr, "driver: expected an (nfreq, ntime) input array\n");
            return 2;
        }

        long nfreq = in.shape[0];
        long ntime = in.shape[1];

        // Checked here rather than left to bonsai: the data pointer below is taken from
        // in.data, which must not be empty.
        if ((nfreq <= 0) || (ntime <= 0)) {
            fprintf(stderr, "driver: expected a non-empty (nfreq, ntime) input array\n");
            return 2;
        }

        bonsai::gridding g(nfreq, freq_lo, freq_hi, tree_size, /*beta_fid=*/ 0.0, /*nups=*/ 1);

        // The production kernel requires an aligned destination; bonsai's aligned_alloc()
        // gives 128 bytes. The input pointer need not be aligned.
        float *prod = bonsai::aligned_alloc<float> (tree_size * ntime);
        float *ref = bonsai::aligned_alloc<float> (tree_size * ntime);

        // Arguments: (dst, dst_nt, dst_stride, itree0, itree1, data, data_nt, data_stride).
        g.grid_data_to_tree(prod, ntime, ntime, 0, tree_size, &in.data[0], ntime, ntime);
        g.reference_grid_data_to_tree(ref, ntime, ntime, 0, tree_size, &in.data[0], ntime, ntime);

        std::vector<size_t> shape { size_t(tree_size), size_t(ntime) };
        npy::write<float> (cl.outputs[0], shape, prod);
        npy::write<float> (cl.outputs[1], shape, ref);

        free(prod);
        free(ref);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
