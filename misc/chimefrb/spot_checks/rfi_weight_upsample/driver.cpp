// Old-side driver for the 'rfi_weight_upsample' spot test.
//
//    driver <w_hires.npy> <w_lores.npy> <output.npy> Df=<int> Dt=<int> w_cutoff=<float>
//
// Runs rf_kernels::weight_upsampler, which rf_pipelines::wi_sub_pipeline uses to push the
// mask its sub-chain produced back up to full resolution: every full-resolution weight whose
// (Df x Dt) cell has a low-resolution weight w_lo <= w_cutoff is zeroed, and every other one
// is left alone.
//
// Two inputs of different shapes, which is why this driver takes a list rather than the
// stacked (2, F, T) array the other RFI spot tests use:
//
//    w_hires  (nfreq_lo*Df, nt_lo*Dt)  float32   the full-resolution weights, modified
//    w_lores  (nfreq_lo, nt_lo)        float32   the low-resolution weights, read
//    output   (nfreq_lo*Df, nt_lo*Dt)  float32   w_hires after the upsample
//
// The old kernel needs nt_lo % 8 == 0, and supports Df and Dt in {1, 2, 4} or any multiple
// of 8.
//
// LIBS: -lrf_kernels

#include <rf_kernels/core.hpp>
#include <rf_kernels/upsample.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: driver <w_hires.npy> <w_lores.npy> <output.npy> Df=<int>"
                " Dt=<int> w_cutoff=<float>\n");
        return 2;
    }

    try {
        npy::cmdline cl(argc, argv, 2, 1);
        npy::array<float> hi = npy::read<float> (cl.inputs[0]);
        npy::array<float> lo = npy::read<float> (cl.inputs[1]);

        long Df = cl.kv.get_long("Df", 1);
        long Dt = cl.kv.get_long("Dt", 1);
        double w_cutoff = cl.kv.get_double("w_cutoff", 0.0);

        if ((hi.shape.size() != 2) || (lo.shape.size() != 2)) {
            fprintf(stderr, "driver: expected 2-d w_hires and w_lores arrays\n");
            return 2;
        }

        size_t nfreq_lo = lo.shape[0];
        size_t nt_lo = lo.shape[1];

        if ((hi.shape[0] != nfreq_lo*Df) || (hi.shape[1] != nt_lo*Dt)) {
            fprintf(stderr, "driver: w_hires shape does not match (Df,Dt)=(%ld,%ld)\n", Df, Dt);
            return 2;
        }

        // upsample() modifies the full-resolution weights in place, so hand it a copy.
        std::vector<float> out(hi.data);

        rf_kernels::weight_upsampler up(Df, Dt);

        up.upsample(nfreq_lo, nt_lo,
                    &out[0], nt_lo*Dt,          // full-resolution, and its row stride
                    &lo.data[0], nt_lo,         // low-resolution, and its row stride
                    float(w_cutoff));

        npy::write<float> (cl.outputs[0], { nfreq_lo*Df, nt_lo*Dt }, &out[0]);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
