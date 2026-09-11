// Old-side driver for the 'rfi_std_dev_clipper' spot test.
//
//    driver <input.npy> <output.npy> output=<clip1d|wrms|weights|clipped_var>
//           sigma=<float> axis=<0|1> Df=<int> Dt=<int> two_pass=<0|1>
//
// Runs rf_kernels::std_dev_clipper, which zeroes whole channels (AXIS_TIME) or whole time
// samples (AXIS_FREQ) whose variance is an outlier among its peers: stage 1 computes one
// variance per row, stage 2 (std_dev_clipper::_clip_1d) clips outliers in that array of
// variances, and the apply zeroes the weights of every clipped row.
//
// 'output=' selects one of four things, since a driver writes one array per run:
//
//    output=clip1d:       input (K, n) float32, a batch of variance vectors. Each is
//                         copied into a std_dev_clipper's public tmp_v and run through its
//                         public _clip_1d() directly -- exactly what the old unit test does
//                         -- and the result written as (K, n). This is _clip_1d in
//                         isolation, the one function in the port with no reference
//                         implementation anywhere in the old code. 'n' must be a multiple
//                         of 8 (the constructor's rule); axis/Df/Dt/two_pass are ignored.
//
//    output=wrms:         input (2, F, T); weighted_mean_rms at niter=1, the clipper's
//                         stage 1, written as (2, nout): out[0] = mean, out[1] = rms.
//
//    output=weights:      input (2, F, T); std_dev_clipper::clip(), written as the (F, T)
//                         clipped weights.
//
//    output=clipped_var:  input (2, F, T); the same run, but writing tmp_v AFTER clip() --
//                         the stage-2 output, one variance per row, zero where clipped.
//
// Input pairs are stacked along a leading length-2 axis, arr[0] intensity and arr[1]
// weights, the convention of misc/chimefrb/rfi_wi_downsample/. 'axis' is cast straight to
// rf_kernels::axis_type (0 = FREQ, 1 = TIME), whose values pirate's ClipperAxis shares.
//
// LIBS: -lrf_kernels

#include <rf_kernels/core.hpp>
#include <rf_kernels/mean_rms.hpp>
#include <rf_kernels/std_dev_clipper.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <exception>

static int run_clip1d(const npy::array<float> &in, double sigma, const char *outpath)
{
    if (in.shape.size() != 2) {
        fprintf(stderr, "driver: output=clip1d expects a (K,n) input array\n");
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

    npy::write<float> (outpath, { K, n }, &out[0]);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> output=<clip1d|wrms|weights|clipped_var>"
                " sigma=<float> axis=<0|1> Df=<int> Dt=<int> two_pass=<0|1>\n");
        return 2;
    }

    try {
        npy::array<float> in = npy::read<float> (argv[1]);
        npy::params p(argc, argv);

        std::string what = p.get("output", "weights");
        double sigma = p.get_double("sigma", 3.0);

        if (what == "clip1d")
            return run_clip1d(in, sigma, argv[2]);

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

        if (what == "wrms") {
            rf_kernels::weighted_mean_rms wrms(F, T, rf_kernels::axis_type(axis), Df, Dt,
                                               1, 0.0, two_pass);
            wrms.compute_wrms(intensity, T, weights, T);

            size_t nout = wrms.nout;
            std::vector<float> out(2 * nout);
            for (size_t i = 0; i < nout; i++) {
                out[i] = wrms.out_mean[i];
                out[nout + i] = wrms.out_rms[i];
            }

            npy::write<float> (argv[2], { 2, nout }, &out[0]);
            return 0;
        }

        if ((what == "weights") || (what == "clipped_var")) {
            rf_kernels::std_dev_clipper sd(F, T, rf_kernels::axis_type(axis), sigma,
                                           Df, Dt, two_pass);

            // clip() modifies the weights in place, so hand it a copy.
            std::vector<float> w(weights, weights + F*T);
            sd.clip(intensity, T, &w[0], T);

            if (what == "weights")
                npy::write<float> (argv[2], { F, T }, &w[0]);
            else
                npy::write<float> (argv[2], { size_t(sd.ntmp_v) }, sd.tmp_v);

            return 0;
        }

        fprintf(stderr, "driver: unknown output=%s\n", what.c_str());
        return 2;
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
