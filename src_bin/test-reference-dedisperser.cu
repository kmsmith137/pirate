#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/internals/ReferenceDedisperser.hpp"
#include "../include/pirate/internals/inlines.hpp"  // pow2()
#include "../include/pirate/internals/utils.hpp"    // make_chord_dedispersion_config()

#include <gputils/Array.hpp>
#include <gputils/rand_utils.hpp>
#include <gputils/test_utils.hpp>    // assert_arrays_equal()

using namespace std;
using namespace gputils;
using namespace pirate;


// -------------------------------------------------------------------------------------------------


static void test_reference_dedispersion(const DedispersionConfig &config, int soph1, int soph2, int nchunks, bool noisy=false)
{
    if (noisy) {
	cout << "test_reference_dedispersion(soph1=" << soph1
	     << ", soph2=" << soph2 << ", nchunks=" << nchunks << ")" << endl;
	config.print(cout, 4);
    }
    
    shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);
    ReferenceDedisperser rdd1(plan, soph1);
    ReferenceDedisperser rdd2(plan, soph2);

    assert(rdd1.output_ntrees == rdd2.output_ntrees);
    
    int nfreq = pow2(config.tree_rank);
    int nt_chunk = config.time_samples_per_chunk;
    int ntrees = rdd1.output_ntrees;
    
    for (int c = 0; c < nchunks; c++) {
	if (noisy)
	    cout << "    chunk " << c << "/" << nchunks << endl;
	
	Array<float> arr({nfreq,nt_chunk}, af_uhost | af_random);
	// Array<float> arr({nfreq,nt_chunk}, af_uhost | af_zero);
	// arr.at({0,0}) = 1.0;
	
	rdd1.dedisperse(arr);
	rdd2.dedisperse(arr);

	for (int itree = 0; itree < ntrees; itree++) {
	    const Array<float> &arr1 = rdd1.output_arrays.at(itree);
	    const Array<float> &arr2 = rdd2.output_arrays.at(itree);
	    assert_arrays_equal(arr1, arr2, "A", "B", {"dm_brev","t"});
	}
    }
}


// -------------------------------------------------------------------------------------------------


static void run_random_small_configs(int niter)
{
    for (int iter = 0; iter < niter; iter++) {
	cout << "\n    *** Running random small config " << iter << "/" << niter << " ***\n" << endl;
	
	auto config = DedispersionConfig::make_random();
	config.planner_verbosity = 1;

	int max_nt = 8192;
	assert(config.time_samples_per_chunk <= max_nt);
	
	int max_nchunks = max_nt / config.time_samples_per_chunk;  // round down
	int nchunks = gputils::rand_int(1, max_nchunks+1);
	
	test_reference_dedispersion(config, 0, 3, nchunks, true);  // noisy=true
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
	test_reference_dedispersion(config, 0, 3, nchunks, true);  // noisy=true
    }
    
    cout << "\ntest-reference-dedisperser: pass" << endl;
    return 0;
}

