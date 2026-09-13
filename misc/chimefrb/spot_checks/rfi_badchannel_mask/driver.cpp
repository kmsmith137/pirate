// Old-side driver for the 'rfi_badchannel_mask' spot test.
//
//    driver <input.npy> <output.npy> nfreq=<int> freq_lo_MHz=<float> freq_hi_MHz=<float>
//           nt_chunk=<int> catch=<0|1>
//
// Runs rf_pipelines::badchannel_mask, the transform that zeroes whole frequency channels
// chosen by a list of MHz ranges. It has to go through the real pipeline machinery: the
// transform turns MHz into channel indices in _bind_transform(), which only a pipeline
// calls, and the class is not declared in any header.
//
//    input:   (n, 2) float64    the mask ranges, one (lo, hi) per row, in MHz
//    output:  (nfreq, nt_chunk) float32    the weights after the transform, which start
//             as all ones; or, with catch=1, a (0,) array if the old code threw
//
// catch=1 lets a test check that the old code REFUSES an input. Without it, a throw is an
// ordinary driver failure (nonzero exit), which harness.run_driver() treats as fatal.
//
// LIBS: -lrf_pipelines -ljsoncpp

#include <rf_pipelines.hpp>

#include "../npy.hpp"

#include <cstdio>
#include <memory>
#include <vector>
#include <exception>

using namespace std;
using namespace rf_pipelines;


// A stream whose first chunk is (intensity 0, weights 1), and which ends after it. Its other
// job is to supply the json attributes a CHIME stream would: badchannel_mask reads the band
// edges from them.
struct one_chunk_stream : public wi_stream
{
    const double freq_lo_MHz;
    const double freq_hi_MHz;

    one_chunk_stream(ssize_t nfreq_, ssize_t nt_chunk_, double freq_lo_MHz_, double freq_hi_MHz_) :
        wi_stream("one_chunk_stream"),
        freq_lo_MHz(freq_lo_MHz_),
        freq_hi_MHz(freq_hi_MHz_)
    {
        this->nfreq = nfreq_;
        this->nt_chunk = nt_chunk_;
    }

    virtual void _bind_stream(Json::Value &json_attrs) override
    {
        json_attrs["freq_lo_MHz"] = freq_lo_MHz;
        json_attrs["freq_hi_MHz"] = freq_hi_MHz;
        json_attrs["dt_sample"] = 1.0e-3;
    }

    virtual bool _fill_chunk(float *intensity, ssize_t istride, float *weights, ssize_t wstride, ssize_t pos) override
    {
        for (ssize_t f = 0; f < nfreq; f++) {
            for (ssize_t t = 0; t < nt_chunk; t++) {
                intensity[f*istride + t] = 0.0f;
                weights[f*wstride + t] = 1.0f;
            }
        }

        // Returning false marks end-of-stream; the chunk at pos=0 has been delivered by then.
        return (pos == 0);
    }
};


// Copies the weights of the chunk at pos=0 into 'out', shape (nfreq, nt_chunk).
struct capture_weights : public wi_transform
{
    vector<float> &out;

    capture_weights(ssize_t nt_chunk_, vector<float> &out_) :
        wi_transform("capture_weights"),
        out(out_)
    {
        this->nt_chunk = nt_chunk_;
    }

    virtual void _process_chunk(float *intensity, ssize_t istride, float *weights, ssize_t wstride, ssize_t pos) override
    {
        if (pos != 0)
            return;
        for (ssize_t f = 0; f < nfreq; f++)
            for (ssize_t t = 0; t < nt_chunk; t++)
                out[f*nt_chunk + t] = weights[f*wstride + t];
    }
};


int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: driver <input.npy> <output.npy> nfreq=<int> freq_lo_MHz=<float>"
                " freq_hi_MHz=<float> nt_chunk=<int> catch=<0|1>\n");
        return 2;
    }

    try {
        npy::array<double> in = npy::read<double> (argv[1]);
        npy::params p(argc, argv);

        ssize_t nfreq = p.get_long("nfreq", 0);
        double freq_lo_MHz = p.get_double("freq_lo_MHz", 400.0);
        double freq_hi_MHz = p.get_double("freq_hi_MHz", 800.0);
        ssize_t nt_chunk = p.get_long("nt_chunk", 16);
        bool catch_throws = (p.get_long("catch", 0) != 0);

        if ((in.shape.size() != 2) || (in.shape[1] != 2)) {
            fprintf(stderr, "driver: expected an (n,2) input array of mask ranges\n");
            return 2;
        }
        if ((nfreq <= 0) || (nt_chunk <= 0)) {
            fprintf(stderr, "driver: expected nfreq > 0 and nt_chunk > 0\n");
            return 2;
        }

        vector<pair<double,double>> ranges;
        for (size_t i = 0; i < in.shape[0]; i++)
            ranges.push_back(make_pair(in.data[2*i], in.data[2*i+1]));

        // Anything the capture transform does not overwrite stays -1, which no answer can be.
        vector<float> out(nfreq * nt_chunk, -1.0f);

        try {
            auto stream = make_shared<one_chunk_stream> (nfreq, nt_chunk, freq_lo_MHz, freq_hi_MHz);
            auto mask = make_badchannel_mask("", ranges);   // throws here if some lo >= hi
            auto capture = make_shared<capture_weights> (nt_chunk, out);

            // badchannel_mask leaves its own chunk size to the pipeline; set it to the stream's,
            // so that one chunk in means one chunk through. It does not affect the mask.
            mask->nt_chunk = nt_chunk;

            vector<shared_ptr<pipeline_object>> elements = { stream, mask, capture };
            auto pl = make_shared<pipeline> (elements, "rfi_badchannel_mask");

            run_params rp;
            rp.outdir = "";      // write no files
            rp.verbosity = 0;
            pl->run(rp);         // throws here, in bind, if a range is outside the band
        }
        catch (std::exception &e) {
            if (!catch_throws)
                throw;
            npy::write<float> (argv[2], { size_t(0) }, out.data());
            return 0;
        }

        npy::write<float> (argv[2], { size_t(nfreq), size_t(nt_chunk) }, out.data());
    }
    catch (std::exception &e) {
        fprintf(stderr, "driver: %s\n", e.what());
        return 1;
    }

    return 0;
}
