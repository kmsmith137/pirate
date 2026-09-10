// Old-side driver for the 'rfi_wi_downsample' spot test.
//
//    driver <input.npy> <output.npy> Df=<int> Dt=<int>
//
// Runs rf_kernels::wi_downsampler, the (Df,Dt) downsampler used throughout the old
// CHIME FRB search's RFI chain.
//
// These transforms operate on an (intensity, weights) PAIR, and harness.run_driver()
// moves one array each way, so both arrays travel stacked along a leading length-2
// axis:  arr[0] = intensity, arr[1] = weights.  The RFI spot tests that follow this
// one should use the same convention.
//
//    input:   (2, F, T)        float32
//    output:  (2, F/Df, T/Dt)  float32
//
// LIBS: -lrf_kernels

#include <rf_kernels/downsample.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <vector>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 3) {
	fprintf(stderr, "usage: driver <input.npy> <output.npy> Df=<int> Dt=<int>\n");
	return 2;
    }

    try {
	npy::array<float> in = npy::read<float> (argv[1]);
	npy::params p(argc, argv);

	long Df = p.get_long("Df", 1);
	long Dt = p.get_long("Dt", 1);

	if ((in.shape.size() != 3) || (in.shape[0] != 2)) {
	    fprintf(stderr, "driver: expected a (2,F,T) input array\n");
	    return 2;
	}

	size_t F = in.shape[1];
	size_t T = in.shape[2];

	if ((Df < 1) || (Dt < 1) || (F % Df) || (T % Dt)) {
	    fprintf(stderr, "driver: bad (Df,Dt)=(%ld,%ld) for (F,T)=(%zu,%zu)\n", Df, Dt, F, T);
	    return 2;
	}

	size_t F_ds = F / Df;
	size_t T_ds = T / Dt;

	std::vector<float> out(2 * F_ds * T_ds, 0.0);

	rf_kernels::wi_downsampler ds(Df, Dt);

	ds.downsample(F_ds, T_ds,
		      &out[0], T_ds,                 // out_i
		      &out[F_ds*T_ds], T_ds,         // out_w
		      &in.data[0], T,                // in_i
		      &in.data[F*T], T);             // in_w

	npy::write<float> (argv[2], { 2, F_ds, T_ds }, &out[0]);
    }
    catch (std::exception &e) {
	fprintf(stderr, "driver: %s\n", e.what());
	return 1;
    }

    return 0;
}
