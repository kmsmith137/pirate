// Old-side driver for the 'spline_detrender' spot test.
//
//    driver <input.npy> <output.npy> nbins=<int> epsilon=<float>
//
// Runs rf_kernels::spline_detrender, the frequency-direction spline detrender of the old
// CHIME FRB search's RFI chain, on one (intensity, weights) pair. As in the other RFI
// spot tests the pair travels stacked along a leading length-2 axis: arr[0] = intensity,
// arr[1] = weights. The weights are not modified, so only the intensity comes back.
//
//    input:   (2, F, T)  float32     T % 8 == 0 and F >= 16*nbins, which the old kernel requires
//    output:  (F, T)     float32     the detrended intensity
//
// LIBS: -lrf_kernels

#include <rf_kernels/spline_detrender.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> nbins=<int> epsilon=<float>\n");
        return 2;
    }

    try {
        npy::array<float> in = npy::read<float> (argv[1]);
        npy::params p(argc, argv);

        long nbins = p.get_long("nbins", 6);
        double epsilon = p.get_double("epsilon", 3.0e-4);

        if ((in.shape.size() != 3) || (in.shape[0] != 2)) {
            fprintf(stderr, "driver: expected a (2,F,T) input array\n");
            return 2;
        }

        size_t F = in.shape[1];
        size_t T = in.shape[2];

        if ((T % 8) || (long(F) < 16*nbins)) {
            fprintf(stderr, "driver: rf_kernels needs T %% 8 == 0 and F >= 16*nbins, got"
                    " (F,T)=(%zu,%zu), nbins=%ld\n", F, T, nbins);
            return 2;
        }

        // The old kernel works in place, so detrend a copy of the intensity plane.
        std::vector<float> intensity(in.data.begin(), in.data.begin() + F*T);

        const int nfreq_i = int(F), nbins_i = int(nbins), nt_i = int(T);
        const float epsilon_f = float(epsilon);
        rf_kernels::spline_detrender sd(nfreq_i, nbins_i, epsilon_f);
        sd.detrend(nt_i,
                   &intensity[0], nt_i,          // intensity, istride
                   &in.data[F*T], nt_i);         // weights, wstride

        npy::write<float> (argv[2], { F, T }, &intensity[0]);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
