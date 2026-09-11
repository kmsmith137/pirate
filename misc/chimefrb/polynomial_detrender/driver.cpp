// Old-side driver for the 'polynomial_detrender' spot test.
//
//    driver <input.npy> <output.npy> polydeg=<int> epsilon=<float> axis=<0|1> nt_chunk=<int>
//
// Runs rf_kernels::polynomial_detrender, the Legendre-polynomial detrender of the old CHIME
// FRB search's RFI chain, on one (intensity, weights) pair stacked along a leading length-2
// axis: arr[0] = intensity, arr[1] = weights. Both come back, since the kernel zeroes the
// weights of rows whose fit is poorly conditioned.
//
//    input:   (2, F, T)  float32
//    output:  (2, F, T)  float32     the detrended intensity, and the weights after
//
// 'axis' is cast straight to rf_kernels::axis_type (0 = FREQ, 1 = TIME). Along time the
// driver fits each block of nt_chunk samples separately, which is what the
// rf_pipelines::polynomial_detrender transform does with its nt_chunk; T must be a
// multiple of nt_chunk. Along frequency there is one fit per time sample and nt_chunk is
// ignored. The old kernel needs the fit length along time to be a multiple of 8.
//
// LIBS: -lrf_kernels

#include <rf_kernels/core.hpp>
#include <rf_kernels/polynomial_detrender.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> polydeg=<int> epsilon=<float>"
                " axis=<0|1> nt_chunk=<int>\n");
        return 2;
    }

    try {
        npy::array<float> in = npy::read<float> (argv[1]);
        npy::params p(argc, argv);

        const int polydeg = int(p.get_long("polydeg", 4));
        const double epsilon = p.get_double("epsilon", 0.01);
        const int axis = int(p.get_long("axis", 1));
        const long nt_chunk = p.get_long("nt_chunk", 1024);

        if ((in.shape.size() != 3) || (in.shape[0] != 2)) {
            fprintf(stderr, "driver: expected a (2,F,T) input array\n");
            return 2;
        }
        if ((axis != 0) && (axis != 1)) {
            fprintf(stderr, "driver: axis must be 0 (FREQ) or 1 (TIME)\n");
            return 2;
        }

        const long F = long(in.shape[1]);
        const long T = long(in.shape[2]);

        // The old kernel works in place on both planes, so detrend a copy of the input.
        std::vector<float> out(in.data.begin(), in.data.end());
        float *intensity = &out[0];
        float *weights = &out[F*T];

        const rf_kernels::axis_type axis_e = rf_kernels::axis_type(axis);
        rf_kernels::polynomial_detrender pd{axis_e, polydeg};   // braces: not a declaration

        const int nfreq_i = int(F), nt_i = int(T);

        if (axis == 0) {
            if (T % 8) {
                fprintf(stderr, "driver: rf_kernels needs T %% 8 == 0 along frequency, got T=%ld\n", T);
                return 2;
            }
            pd.detrend(nfreq_i, nt_i, intensity, nt_i, weights, nt_i, epsilon);
        }
        else {
            if ((nt_chunk <= 0) || (nt_chunk % 8) || (T % nt_chunk)) {
                fprintf(stderr, "driver: need nt_chunk %% 8 == 0 and T %% nt_chunk == 0, got"
                        " T=%ld, nt_chunk=%ld\n", T, nt_chunk);
                return 2;
            }
            const int ntc_i = int(nt_chunk);
            for (long c = 0; c < T / nt_chunk; c++)
                pd.detrend(nfreq_i, ntc_i, intensity + c*nt_chunk, nt_i, weights + c*nt_chunk, nt_i, epsilon);
        }

        npy::write<float> (argv[2], { 2, size_t(F), size_t(T) }, &out[0]);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
