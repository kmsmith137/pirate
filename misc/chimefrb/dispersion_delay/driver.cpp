// Old-side driver for the 'dispersion_delay' spot test.
//
//    driver <input.npy> <output.npy>
//
// Input is an (N,2) float64 array of (DM, frequency in MHz) pairs.
// Output is an (N,) float64 array of dispersion delays in seconds, from bonsai.
//
// This is the smallest driver that can exist, and exists mostly to show the shape of
// one: read one array, compute, write one array, say nothing.  It has no idea what
// pirate is or what will be done with its output.

#include <bonsai.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <exception>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy>\n");
        return 2;
    }

    try {
        npy::array<double> in = npy::read<double> (argv[1]);

        if ((in.shape.size() != 2) || (in.cols() != 2)) {
            fprintf(stderr, "driver: expected an (N,2) input array\n");
            return 2;
        }

        std::vector<double> out(in.rows());

        for (size_t i = 0; i < in.rows(); i++) {
            double dm = in.data[2*i];
            double freq_MHz = in.data[2*i + 1];
            out[i] = bonsai::dispersion_delay(dm, freq_MHz);
        }

        npy::write<double> (argv[2], out);
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
