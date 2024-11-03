#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/internals/ReferenceDedisperser.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()

#include <gputils/Array.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>    // assert_arrays_equal()

using namespace std;
using namespace gputils;
using namespace pirate;


static void test_reference_dedispersion(const DedispersionConfig &config, int nchunks)
{
    cout << "\n" << "test_reference_dedispersion2: nchunks=" << nchunks << endl;
    config.print(cout, 4);
    
    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);
    // plan->print(cout, 8);
    
    int nfreq = pow2(config.tree_rank);
    int nt_chunk = config.time_samples_per_chunk;
    int beams_per_batch = config.beams_per_batch;
    int nbatches = xdiv(config.beams_per_gpu, beams_per_batch);
    int nout = plan->stage1_trees.size();
    
    shared_ptr<ReferenceDedisperserBase> rdd0 = ReferenceDedisperserBase::make(plan, 0);
    shared_ptr<ReferenceDedisperserBase> rdd1 = ReferenceDedisperserBase::make(plan, 1);
    shared_ptr<ReferenceDedisperserBase> rdd2 = ReferenceDedisperserBase::make(plan, 2);

    for (int c = 0; c < nchunks; c++) {
	for (int b = 0; b < nbatches; b++) {
	    //cout << "chunk " << c << "/" << nchunks
	    //<< ", batch " << b << "/" << nbatches
	    //<< endl;
	
	    Array<float> arr({beams_per_batch, nfreq, nt_chunk}, af_uhost | af_random);
	    // Array<float> arr({nfreq,nt_chunk}, af_uhost | af_zero);
	    // arr.at({0,0}) = 1.0;

	    rdd0->input_array.fill(arr);
	    rdd0->dedisperse(c, b*beams_per_batch);

	    rdd1->input_array.fill(arr);
	    rdd1->dedisperse(c, b*beams_per_batch);

	    rdd2->input_array.fill(arr);
	    rdd2->dedisperse(c, b*beams_per_batch);
	    
	    for (int iout = 0; iout < nout; iout++) {
		const Array<float> &arr0 = rdd0->output_arrays.at(iout);
		const Array<float> &arr1 = rdd1->output_arrays.at(iout);
		const Array<float> &arr2 = rdd2->output_arrays.at(iout);
		assert_arrays_equal(arr0, arr1, "soph0", "soph1", {"beam","dm_brev","t"});
		assert_arrays_equal(arr0, arr2, "soph0", "soph2", {"beam","dm_brev","t"});
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

	int max_nt = 8192;
	assert(config.time_samples_per_chunk <= max_nt);
	
	int max_nchunks = max_nt / config.time_samples_per_chunk;  // round down
	int nchunks = gputils::rand_int(1, max_nchunks+1);
	
	test_reference_dedispersion(config, nchunks);
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
	     << "   ./bin/test-reference-dedisperser configs/dedispersion/chord_zen3/chord_zen3_int8_float16.yml\n";
    }

    for (int iarg = 1; iarg < argc; iarg++) {
	auto config = DedispersionConfig::from_yaml(argv[iarg]);
    
	int nt_tot = 1024 * 1024;  // FIXME promote to command-line arg?
	int nchunks = xdiv(nt_tot, config.time_samples_per_chunk);
	test_reference_dedispersion(config, nchunks);
    }
    
    cout << "\ntest-reference-dedisperser: pass" << endl;
    return 0;
}

