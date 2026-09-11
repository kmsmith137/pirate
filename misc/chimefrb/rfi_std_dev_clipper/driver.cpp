// Old-side driver for the 'rfi_std_dev_clipper' spot test.
//
//    driver <input.npy> <weights.npy> <clipped_var.npy> <wrms.npy>
//           sigma=<float> axis=<0|1> Df=<int> Dt=<int> two_pass=<0|1>
//
// Runs rf_kernels::std_dev_clipper, which zeroes whole channels (AXIS_TIME) or whole time
// samples (AXIS_FREQ) whose variance is an outlier among its peers: stage 1 computes one
// variance per row, stage 2 (std_dev_clipper::_clip_1d) clips outliers in that array of
// variances, and the apply zeroes the weights of every clipped row.
//
//    input:        (2, F, T)   float32   arr[0] = intensity, arr[1] = weights
//
// and three outputs, all from one run:
//
//    weights       (F, T)      std_dev_clipper::clip(), the clipped weights
//    clipped_var   (ntmp_v,)   tmp_v AFTER that clip() -- the stage-2 output, one
//                              variance per row, zero where clipped
//    wrms          (2, nout)   weighted_mean_rms at niter=1, the clipper's stage 1:
//                              out[0] = mean, out[1] = rms
//
// The input pair is stacked along a leading length-2 axis, the convention of
// misc/chimefrb/rfi_wi_downsample/. 'axis' is cast straight to rf_kernels::axis_type
// (0 = FREQ, 1 = TIME), whose values pirate's ClipperAxis shares. Stage 2 on its own,
// std_dev_clipper::_clip_1d(), is driven directly by misc/chimefrb/rfi_std_dev_clip_1d/.
//
// LIBS: -lrf_kernels

#include <rf_kernels/core.hpp>
#include <rf_kernels/mean_rms.hpp>
#include <rf_kernels/std_dev_clipper.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: driver <input.npy> <weights.npy> <clipped_var.npy> <wrms.npy>"
                " sigma=<float> axis=<0|1> Df=<int> Dt=<int> two_pass=<0|1>\n");
        return 2;
    }

    try {
        npy::cmdline cl(argc, argv, 1, 3);
        const npy::params &p = cl.kv;
        npy::array<float> in = npy::read<float> (cl.inputs[0]);

        double sigma = p.get_double("sigma", 3.0);
        int axis = int(p.get_long("axis", 1));
        int Df = int(p.get_long("Df", 1));
        int Dt = int(p.get_long("Dt", 1));
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
            rf_kernels::std_dev_clipper sd(F, T, rf_kernels::axis_type(axis), sigma,
                                           Df, Dt, two_pass);

            // clip() modifies the weights in place, so hand it a copy.
            std::vector<float> w(weights, weights + F*T);
            sd.clip(intensity, T, &w[0], T);

            npy::write<float> (cl.outputs[0], { F, T }, &w[0]);
            npy::write<float> (cl.outputs[1], { size_t(sd.ntmp_v) }, sd.tmp_v);
        }

        {
            rf_kernels::weighted_mean_rms wrms(F, T, rf_kernels::axis_type(axis), Df, Dt,
                                               1, 0.0, two_pass);
            wrms.compute_wrms(intensity, T, weights, T);

            size_t nout = wrms.nout;
            std::vector<float> out(2 * nout);
            for (size_t i = 0; i < nout; i++) {
                out[i] = wrms.out_mean[i];
                out[nout + i] = wrms.out_rms[i];
            }

            npy::write<float> (cl.outputs[2], { 2, nout }, &out[0]);
        }
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
