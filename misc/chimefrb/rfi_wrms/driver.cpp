// Old-side driver for the 'rfi_wrms' spot test.
//
//    driver <input.npy> <output.npy> niter=<int> iter_sigma=<float> two_pass=<0|1>
//
// Runs rf_kernels::weighted_mean_rms, the weighted mean and variance with iterated
// sigma clipping that both clippers in the old CHIME FRB search's RFI chain are built
// on.
//
// Input is an (intensity, weights) pair stacked along a leading length-2 axis, the
// convention established by misc/chimefrb/rfi_wi_downsample/:
//
//    input:   (2, R, L)  float32     arr[0] = intensity, arr[1] = weights
//    output:  (2, R)     float32     out[0] = mean, out[1] = RMS (not variance)
//
// The old kernel computes one statistic per row of an (R, L) array when it is called
// with axis=AXIS_TIME and Df=Dt=1, which is exactly pirate's GpuWrms signature: pirate
// does the axis handling one layer up, in GpuWiDownsampler.
//
// LIBS: -lrf_kernels

#include <rf_kernels/core.hpp>
#include <rf_kernels/mean_rms.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> niter=<int>"
                " iter_sigma=<float> two_pass=<0|1>\n");
        return 2;
    }

    try {
        npy::array<float> in = npy::read<float> (argv[1]);
        npy::params p(argc, argv);

        long niter = p.get_long("niter", 1);
        double iter_sigma = p.get_double("iter_sigma", 0.0);
        bool two_pass = (p.get_long("two_pass", 1) != 0);

        if ((in.shape.size() != 3) || (in.shape[0] != 2)) {
            fprintf(stderr, "driver: expected a (2,R,L) input array\n");
            return 2;
        }

        size_t R = in.shape[1];
        size_t L = in.shape[2];

        rf_kernels::weighted_mean_rms wrms(R, L, rf_kernels::AXIS_TIME, 1, 1,
                                           niter, iter_sigma, two_pass);

        wrms.compute_wrms(&in.data[0], L, &in.data[R*L], L);

        std::vector<float> out(2 * R);
        for (size_t r = 0; r < R; r++) {
            out[r] = wrms.out_mean[r];
            out[R + r] = wrms.out_rms[r];
        }

        npy::write<float> (argv[2], { 2, R }, &out[0]);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
