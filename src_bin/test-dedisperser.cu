#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/internals/Dedisperser.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()

#include <ksgpu/Array.hpp>
#include <ksgpu/xassert.hpp>
#include <ksgpu/rand_utils.hpp>

using namespace std;
using namespace ksgpu;
using namespace pirate;


static void test_dedisperser(const DedispersionConfig &config, int nchunks)
{
    cout << "\n" << "test_dedisperser: nchunks=" << nchunks << endl;
    config.print(cout, 4);
    
    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);
    // plan->print(cout, 8);
    
    int nfreq = pow2(config.tree_rank);
    int nt_chunk = config.time_samples_per_chunk;
    int beams_per_batch = config.beams_per_batch;
    int nbatches = xdiv(config.beams_per_gpu, beams_per_batch);
    int nstreams = config.num_active_batches;
    int nout = plan->stage2_trees.size();

    // FIXME test multi-stream logic in the future.
    // For now, we use the default cuda stream, which simplifies things since we can
    // freely mix operations such as Array::to_gpu() which use the default stream.
    xassert(nstreams == 1);
    
    shared_ptr<ReferenceDedisperserBase> rdd0 = ReferenceDedisperserBase::make(plan, 0);
    shared_ptr<ReferenceDedisperserBase> rdd1 = ReferenceDedisperserBase::make(plan, 1);
    shared_ptr<ReferenceDedisperserBase> rdd2 = ReferenceDedisperserBase::make(plan, 2);
    
    shared_ptr<GpuDedisperser> gdd = make_shared<GpuDedisperser> (plan);
    gdd->allocate();

    // FIXME revisit epsilon if we change the normalization of the dedispersion transform.
    double epsrel_r = 6 * Dtype::native<float>().precision();   // reference
    double epsrel_g = 6 * config.dtype.precision();             // gpu
    double epsabs_r = epsrel_r * pow(1.414, config.tree_rank);  // reference
    double epsabs_g = epsrel_g * pow(1.414, config.tree_rank);  // gpu

    for (int c = 0; c < nchunks; c++) {
	for (int b = 0; b < nbatches; b++) {
	    Array<float> arr({beams_per_batch, nfreq, nt_chunk}, af_uhost | af_random);
	    // Array<float> arr({nfreq,nt_chunk}, af_uhost | af_zero);
	    // arr.at({0,0}) = 1.0;

	    rdd0->input_array.fill(arr);
	    rdd0->dedisperse(b, c);

	    rdd1->input_array.fill(arr);
	    rdd1->dedisperse(b, c);

	    rdd2->input_array.fill(arr);
	    rdd2->dedisperse(b, c);

	    Array<void> &gdd_inbuf = gdd->stage1_dd_bufs.at(0).bufs.at(0);  // (istream,itree) = (0,0)
	    gdd_inbuf.fill(arr.convert(config.dtype));
	    gdd->launch(b, c, 0, nullptr);  // (ibatch, it_chunk, istream, stream)
	    
	    for (int iout = 0; iout < nout; iout++) {
		const Array<float> &rdd0_out = rdd0->output_arrays.at(iout);
		const Array<float> &rdd1_out = rdd1->output_arrays.at(iout);
		const Array<float> &rdd2_out = rdd2->output_arrays.at(iout);
		const Array<void> &gdd_out = gdd->stage2_dd_bufs.at(0).bufs.at(iout);  // (istream,itree) = (0,iout)

		// Last two arguments are (epsabs, epsrel).
		assert_arrays_equal(rdd0_out, rdd1_out, "soph0", "soph1", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);
		assert_arrays_equal(rdd0_out, rdd2_out, "soph0", "soph2", {"beam","dm_brev","t"}, epsabs_r, epsrel_r);
		assert_arrays_equal(rdd0_out, gdd_out, "soph0", "gpu", {"beam","dm_brev","t"}, epsabs_g, epsrel_g);
	    }
	}
    }
    
    cout << endl;
}


// -------------------------------------------------------------------------------------------------


static void run_random_small_configs(int niter)
{
    for (int iter = 0; iter < niter; iter++) {
	cout << "\n    *** Running random small config " << iter << "/" << niter << " ***\n" << endl;
	
	auto config = DedispersionConfig::make_random();
	config.num_active_batches = 1;   // FIXME currently we only support nstreams==1
	config.validate();

	int max_nt = 8192;
	xassert(config.time_samples_per_chunk <= max_nt);
	
	int max_nchunks = max_nt / config.time_samples_per_chunk;  // round down
	int nchunks = ksgpu::rand_int(1, max_nchunks+1);
	
	test_dedisperser(config, nchunks);
    }
}


int main(int argc, char **argv)
{
    if (argc == 1) {
	const int niter = 100;
	
	cout << "No command-line arguments were specified; running "
	     << niter << " randomly generated 'small' configs" << endl;
	
	run_random_small_configs(niter);
	
	cout << "\nThis concludes our test of " << niter << " randomly generated 'small' configs.\n"
	     << "To run a long test, specify a config on the command line, e.g.\n"
	     << "   ./bin/test-dedisperser configs/dedispersion/chord_zen3/chord_zen3_int8_float16.yml\n";
    }

    for (int iarg = 1; iarg < argc; iarg++) {
	auto config = DedispersionConfig::from_yaml(argv[iarg]);
    
	int nt_tot = 1024 * 1024;  // FIXME promote to command-line arg?
	int nchunks = xdiv(nt_tot, config.time_samples_per_chunk);
	test_dedisperser(config, nchunks);
    }
    
    cout << "\ntest-dedisperser: pass" << endl;
    return 0;
}

