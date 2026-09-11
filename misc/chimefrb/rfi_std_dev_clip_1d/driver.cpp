// Old-side driver for the 'rfi_std_dev_clip_1d' spot test.
//
//    driver <input.npy> <output.npy> sigma=<float>
//
// Runs std_dev_clipper::_clip_1d() on its own: stage 2 of rf_kernels::std_dev_clipper,
// which takes one variance per row and zeroes the outliers among them. It is the one
// function in the old RFI chain with no reference implementation anywhere in the old code,
// so this driver calls the real one directly, on inputs the test constructs -- exactly what
// the old unit test does.
//
//    input:   (K, n) float32   a batch of K variance vectors; n must be a multiple of 8
//                              (the std_dev_clipper constructor's rule)
//    output:  (K, n) float32   each vector after _clip_1d(), zero where clipped
//
// Each vector is copied into a std_dev_clipper's public tmp_v and run through its public
// _clip_1d(). The transform itself is never run; misc/chimefrb/rfi_std_dev_clipper/ covers
// that.
//
// LIBS: -lrf_kernels

#include <rf_kernels/core.hpp>
#include <rf_kernels/std_dev_clipper.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <cstring>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> sigma=<float>\n");
        return 2;
    }

    try {
        npy::array<float> in = npy::read<float> (argv[1]);
        npy::params p(argc, argv);
        double sigma = p.get_double("sigma", 3.0);

        if (in.shape.size() != 2) {
            fprintf(stderr, "driver: expected a (K,n) input array\n");
            return 2;
        }

        size_t K = in.shape[0];
        size_t n = in.shape[1];

        // (nfreq=n, nt_chunk=8, AXIS_TIME) makes ntmp_v == n. The kernel itself is never run.
        rf_kernels::std_dev_clipper sd(n, 8, rf_kernels::AXIS_TIME, sigma, 1, 1, true);

        if (size_t(sd.ntmp_v) != n) {
            fprintf(stderr, "driver: internal error, ntmp_v=%d but n=%zu\n", sd.ntmp_v, n);
            return 1;
        }

        std::vector<float> out(K * n);

        for (size_t k = 0; k < K; k++) {
            memcpy(sd.tmp_v, &in.data[k*n], n * sizeof(float));
            sd._clip_1d();
            memcpy(&out[k*n], sd.tmp_v, n * sizeof(float));
        }

        npy::write<float> (argv[2], { K, n }, &out[0]);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
