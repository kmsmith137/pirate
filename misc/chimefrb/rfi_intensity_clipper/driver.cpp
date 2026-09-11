// Old-side driver for the 'rfi_intensity_clipper' spot test.
//
//    driver <input.npy> <weights.npy> <wrms.npy> axis=<0|1|2> sigma=<float>
//           Df=<int> Dt=<int> niter=<int> iter_sigma=<float> two_pass=<0|1>
//
// Runs rf_kernels::intensity_clipper, the principal RFI flagger in the old CHIME FRB
// search's transform chain: it computes a weighted mean and variance over an axis, then
// zeroes the weights of every sample more than 'sigma' standard deviations away.
//
// Input is an (intensity, weights) pair stacked along a leading length-2 axis, the
// convention established by misc/chimefrb/rfi_wi_downsample/:
//
//    input:   (2, F, T)  float32     arr[0] = intensity, arr[1] = weights
//
// Two outputs, both computed from that one input:
//
//    weights:  (F, T)     float32   the clipped weights, from intensity_clipper::clip()
//    wrms:     (2, nout)  float32   out[0] = mean, out[1] = rms, from weighted_mean_rms
//                                   with the clipper's own (Df, Dt, niter, iter_sigma)
//
// The 'wrms' output exists because of how this transform has to be tested. Comparing nine
// float32 refinements against nine float64 ones is a coin flip, not a test -- one sample
// landing on the other side of a threshold moves everything after it. So the primary
// check takes the OLD kernel's own (mean, rms) and references only the final clip, which
// is what rf_kernels' own unit test does (test-intensity-clipper.cpp). Both sides then
// start from bit-identical statistics and there is nothing to amplify.
//
// 'axis' is passed as an integer and cast to rf_kernels::axis_type: the two enums share
// numeric values with pirate::chimefrb::ClipperAxis on purpose (0=FREQ, 1=TIME, 2=NONE).
//
// LIBS: -lrf_kernels

#include <rf_kernels/core.hpp>
#include <rf_kernels/mean_rms.hpp>
#include <rf_kernels/intensity_clipper.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <string>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: driver <input.npy> <weights.npy> <wrms.npy>"
                " axis=<0|1|2> sigma=<float> Df=<int> Dt=<int> niter=<int>"
                " iter_sigma=<float> two_pass=<0|1>\n");
        return 2;
    }

    try {
        npy::cmdline cl(argc, argv, 1, 2);
        const npy::params &p = cl.kv;
        npy::array<float> in = npy::read<float> (cl.inputs[0]);

        int axis = int(p.get_long("axis", 1));
        double sigma = p.get_double("sigma", 5.0);
        int Df = int(p.get_long("Df", 1));
        int Dt = int(p.get_long("Dt", 1));
        int niter = int(p.get_long("niter", 1));
        double iter_sigma = p.get_double("iter_sigma", 0.0);
        bool two_pass = (p.get_long("two_pass", 1) != 0);

        if ((in.shape.size() != 3) || (in.shape[0] != 2)) {
            fprintf(stderr, "driver: expected a (2,F,T) input array\n");
            return 2;
        }

        size_t F = in.shape[1];
        size_t T = in.shape[2];

        const float *intensity = &in.data[0];
        const float *weights = &in.data[F*T];

        {
            rf_kernels::intensity_clipper ic(F, T, rf_kernels::axis_type(axis), sigma,
                                             Df, Dt, niter, iter_sigma, two_pass);

            // clip() modifies the weights in place, so hand it a copy.
            std::vector<float> w(weights, weights + F*T);
            ic.clip(intensity, T, &w[0], T);

            npy::write<float> (cl.outputs[0], { F, T }, &w[0]);
        }

        {
            // Note: rf_kernels calls this argument 'sigma', but it is the REFINEMENT
            // threshold -- the intensity_clipper passes its iter_sigma here.
            rf_kernels::weighted_mean_rms wrms(F, T, rf_kernels::axis_type(axis), Df, Dt,
                                               niter, iter_sigma, two_pass);

            wrms.compute_wrms(intensity, T, weights, T);

            size_t nout = wrms.nout;
            std::vector<float> out(2 * nout);
            for (size_t i = 0; i < nout; i++) {
                out[i] = wrms.out_mean[i];
                out[nout + i] = wrms.out_rms[i];
            }

            npy::write<float> (cl.outputs[1], { 2, nout }, &out[0]);
        }
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
