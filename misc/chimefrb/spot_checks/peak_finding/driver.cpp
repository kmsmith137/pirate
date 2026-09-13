// Old-side driver for the 'peak_finding' spot test.
//
//    driver <in_data.npy> <in_wt.npy> <out_tm.npy> <out_tv.npy> <out_coeffs.npy>
//           pf_name=<str> M=<int> N=<int> U=<int>
//
// Runs bonsai::reference_peak_finder(), the last step of the old dedisperser: it convolves
// one dedispersed time series with each of the peak-finding profiles, and returns both a
// weighted max over profiles ('tm', the trigger map) and a per-profile mean square ('tv',
// which bonsai uses to normalize triggers to sigmas).
//
//    in_data     (M*N + npad,)  float32   one time series; npad extra samples are read past
//                                         the end (3 for ab4), and must be present
//    in_wt       (P, U)         float32   per-(profile, upsampled phase) weight
//    out_tm      (N,)           float32   max_p (wt[p,i%U] * pf_p[i]), over i in each
//                                         length-M bin. ZERO-INITIALIZED here, and
//                                         reference_peak_finder() max()es into it, so a
//                                         negative trigger reads back as 0.
//    out_tv      (P, U)         float32   (U/(M*N)) * sum_i pf_p[i]^2, over i with i%U fixed
//    out_coeffs  (P, P)         float32   peak_finder_params::pf_coeffs, the profile
//                                         coefficients as bonsai declares them, zero-padded
//                                         to P per row
//
// The two profile sources are deliberately both returned. 'out_tm' and 'out_tv' come from
// the coefficients that reference_peak_finder() applies (hardcoded there, and again in the
// production kernel ab4_stream); 'out_coeffs' is the separate table that bonsai's
// pulse-simulation path uses (sparse_pulse::convolve_peak_finder). They are written down
// independently in the old code, so a test that wants to claim "pirate's profiles are
// bonsai's" should look at both.
//
// LIBS: -lbonsai

#include <bonsai_internals.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <string>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: driver <in_data.npy> <in_wt.npy> <out_tm.npy> <out_tv.npy>"
                " <out_coeffs.npy> pf_name=<str> M=<int> N=<int> U=<int>\n");
        return 2;
    }

    try {
        npy::cmdline cl(argc, argv, 2, 3);

        std::string pf_name = cl.kv.get("pf_name", "ab4");
        long M = cl.kv.get_long("M", 1);
        long N = cl.kv.get_long("N", 0);
        long U = cl.kv.get_long("U", 1);

        if ((M <= 0) || (N <= 0) || (U <= 0)) {
            fprintf(stderr, "driver: need M, N, U > 0\n");
            return 2;
        }

        // Throws on an unrecognized pf_name, which is the validation we want.
        bonsai::peak_finder_params pf(pf_name, "driver");
        long P = pf.npf;
        long npad = pf.npad;

        npy::array<float> in = npy::read<float> (cl.inputs[0]);
        npy::array<float> wt = npy::read<float> (cl.inputs[1]);

        if ((long)in.size() < M*N + npad) {
            fprintf(stderr, "driver: in_data has %ld samples, need M*N + npad = %ld\n",
                    (long)in.size(), M*N + npad);
            return 2;
        }

        if ((long)wt.size() != P*U) {
            fprintf(stderr, "driver: in_wt has %ld entries, need P*U = %ld\n",
                    (long)wt.size(), P*U);
            return 2;
        }

        // reference_peak_finder() max()es into out_tm rather than overwriting it (it is
        // called once per DM in the real pipeline), so the initial value is ours to choose.
        std::vector<float> out_tm(N, 0.0f);
        std::vector<float> out_tv(P*U, 0.0f);

        bonsai::reference_peak_finder(pf_name, M, N, U, &out_tm[0], &out_tv[0],
                                      &in.data[0], &wt.data[0]);

        std::vector<size_t> tm_shape { size_t(N) };
        std::vector<size_t> tv_shape { size_t(P), size_t(U) };
        std::vector<size_t> pc_shape { size_t(P), size_t(P) };

        npy::write<float> (cl.outputs[0], tm_shape, &out_tm[0]);
        npy::write<float> (cl.outputs[1], tv_shape, &out_tv[0]);
        npy::write<float> (cl.outputs[2], pc_shape, &pf.pf_coeffs[0]);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
