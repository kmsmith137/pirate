#include "../include/pirate/internals/ReferenceDedisperser.hpp"
#include "../include/pirate/internals/inlines.hpp"
#include "../include/pirate/internals/utils.hpp"

#include <gputils/rand_utils.hpp>    // rand_int()
#include <gputils/test_utils.hpp>    // assert_arrays_equal()
#include <gputils/string_utils.hpp>  // tuple_str()

using namespace std;
using namespace pirate;
using namespace gputils;


// -------------------------------------------------------------------------------------------------


// This utility function currently isn't used anywhere except test_non_incremental_dedispersion().
static int dedispersion_delay(int rank, int freq, int dm_brev)
{
    int delay = 0;
    int delay0 = 0;

    for (int r = 0; r < rank; r++) {
	int d = (dm_brev & 1) ? (delay0+1) : delay0;
	delay += ((freq & 1) ? 0 : d);
	delay0 += d;
	dm_brev >>= 1;
	freq >>= 1;
    }

    return delay;
}


static void test_non_incremental_dedispersion(int rank, int ntime, int dm_brev, int t0, bool noisy=false)
{
    if (noisy) {
	cout << "test_non_incremental_dedispersion(rank=" << rank << ", ntime=" << ntime
	     << ", dm_brev=" << dm_brev << ", t0=" << t0 << ")" << endl;
    }
	
    check_rank(rank, "test_non_incremental_dedispersion");
    assert((dm_brev >= 0) && (dm_brev < pow2(rank)));
    assert((t0 >= 0) && (t0 < ntime));

    int nchan = pow2(rank);
    Array<float> arr({nchan, ntime}, af_uhost | af_random);

    float x = 0.0;
    for (int ifreq = 0; ifreq < nchan; ifreq++) {
	int t = t0 - dedispersion_delay(rank, ifreq, dm_brev);
	if (t >= 0)
	    x += arr.at({ifreq,t});
    }

    dedisperse_non_incremental(arr);
    float y = arr.at({dm_brev,t0});

    float eps = fabs(x-y) / sqrt(nchan);
    assert(eps < 1.0e-5);
}


static void test_non_incremental_dedispersion(bool noisy=false)
{
    int rank = rand_int(0, 9);
    int ntime = rand_int(1, 500);
    int dm_prev = rand_int(0, pow2(rank));
    int t0 = rand_int(0, ntime);
    
    test_non_incremental_dedispersion(rank, ntime, dm_prev, t0, noisy);
}


// -------------------------------------------------------------------------------------------------


static void test_reference_lagbuf(const vector<int> &lags, int nt_chunk, int nchunks, bool noisy=false)
{
    if (noisy) {
	cout << "test_reference_lagbuf(lags=" << gputils::tuple_str(lags)
	     << ", nt_chunk=" << nt_chunk << ", nchunks=" << nchunks << ")" << endl;
    }

    int nchan = lags.size();
    int nt_tot = nt_chunk * nchunks;
    
    ReferenceLagbuf rbuf(lags, nt_chunk);
    Array<float> arr0({nchan,nt_tot}, af_uhost | af_random);
    Array<float> arr1 = arr0.clone();

    lag_non_incremental(arr0, lags);
        
    for (int c = 0; c < nchunks; c++) {
	Array<float> chunk = arr1.slice(1, c*nt_chunk, (c+1)*nt_chunk);
	rbuf.apply_lags(chunk);
    }

    // (arr0, arr1, name0, name1, axis_names)
    gputils::assert_arrays_equal(arr0, arr1, "non-incremental", "incremental", {"chan","t"});
}


static void test_reference_lagbuf(bool noisy=false)
{
    int nchan = rand_int(1, 11);
    int nt_chunk = rand_int(1, 21);
    int nchunks = rand_int(1, 21);
    
    vector<int> lags(nchan);
    for (int c = 0; c < nchan; c++)
	lags[c] = rand_int(1, 21);

    test_reference_lagbuf(lags, nt_chunk, nchunks, noisy);
}


// -------------------------------------------------------------------------------------------------


static void test_reference_tree(int rank, int nt_chunk, int nchunks, bool noisy=false)
{
    if (noisy) {
	cout << "test_reference_tree(rank=" << rank << ", nt_chunk="
	     << nt_chunk << ", nchunks=" << nchunks << ")" << endl;
    }

    check_rank(rank, "test_reference_tree");
    
    int nfreq = pow2(rank);
    int nt_tot = nt_chunk * nchunks;
    
    Array<float> arr0({nfreq,nt_tot}, af_uhost | af_random);
    Array<float> arr1 = arr0.clone();

    ReferenceTree rtree(rank, nt_chunk);
    assert(rtree.nrstate == rstate_len(rank));
    
    Array<float> rstate({rtree.nrstate}, af_uhost | af_zero);
    Array<float> scratch({rtree.nscratch}, af_uhost | af_random);

    dedisperse_non_incremental(arr0);

    for (int c = 0; c < nchunks; c++) {
	Array<float> chunk = arr1.slice(1, c*nt_chunk, (c+1)*nt_chunk);
	rtree.dedisperse(chunk, rstate.data, scratch.data);
    }

    // (arr0, arr1, name0, name1, axis_names)
    gputils::assert_arrays_equal(arr0, arr1, "non-incremental", "incremental", {"dm_brev","t"});
}


static void test_reference_tree(bool noisy=false)
{
    int rank = rand_int(0, 9);
    int nt_chunk = rand_int(1, pow2(rank+1));
    int maxchunks = std::max(3L, 10000 / (pow2(rank) * nt_chunk));
    int nchunks = rand_int(1, maxchunks+1);

    test_reference_tree(rank, nt_chunk, nchunks, noisy);
}


// -------------------------------------------------------------------------------------------------


static void test_tree_recursion(int rank0, int rank1, int nt_chunk, int nchunks, bool uflag, bool noisy=false)
{
    if (noisy) {
	cout << "test_reference_tree_recursion(rank0=" << rank0 << ", rank1=" << rank1
	     << ", nt_chunk=" << nt_chunk << ", nchunks=" << nchunks
	     << ", uflag=" << (uflag ? "true" : "false") << ")" << endl;
    }

    int rank_tot = rank0 + rank1;
    int rank_big = uflag ? (rank_tot+1) : rank_tot;
    
    check_rank(rank0, "test_tree_recursion [rank0]");
    check_rank(rank1, "test_tree_recursion [rank1]");
    check_rank(rank_big, "test_tree_recursion [rank_big]");
	       
    int nfreq_big = pow2(rank_big);
    int nfreq_tot = pow2(rank_tot);
    int nfreq0 = pow2(rank0);
    int nfreq1 = pow2(rank1);
    
    vector<int> lags(nfreq_tot);
    for (int i = 0; i < nfreq1; i++)
	for (int j = 0; j < nfreq0; j++)
	    lags[i*nfreq0+j] = rb_lag(i, j, rank0, rank1, uflag);

    ReferenceTree big_tree(rank_big, nt_chunk);
    ReferenceTree tree0(rank0, nt_chunk);
    ReferenceTree tree1(rank1, nt_chunk);
    ReferenceLagbuf lagbuf(lags, nt_chunk);
    shared_ptr<ReferenceReducer> reducer;
    
    if (uflag)
	reducer = make_shared<ReferenceReducer> (rank0, rank1, nt_chunk);
    
    int nrstate1 = (nfreq1 * tree0.nrstate) + (nfreq0 * tree1.nrstate);
    Array<float> rstate0({big_tree.nrstate}, af_uhost | af_zero);
    Array<float> rstate1({nrstate1}, af_uhost | af_zero);

    int nscratch = big_tree.nscratch;
    nscratch = std::max(nscratch, tree0.nscratch);
    nscratch = std::max(nscratch, tree1.nscratch);    
    Array<float> scratch({nscratch}, af_uhost | af_random);

    if (!uflag)
	assert(nrstate1 + lagbuf.nrstate == rstate_len(rank_tot));

    Array<float> chunk1_big({nfreq_big, nt_chunk}, af_uhost | af_zero);
    Array<float> chunk1({nfreq_tot, nt_chunk}, af_uhost | af_zero);
    Array<float> chunk0({nfreq_tot, nt_chunk}, af_uhost | af_zero);
    
    for (int c = 0; c < nchunks; c++) {
	Array<float> chunk0_big({nfreq_big, nt_chunk}, af_uhost | af_random);

	// First step: chunk0_big -> chunk1 -> dedisperse

	if (uflag) {
	    chunk1_big.fill(chunk0_big);
	    reducer->reduce(chunk1_big, chunk1);
	}
	else
	    chunk1.fill(chunk0_big);
	
	float *rp = rstate1.data;
	
	for (int i = 0; i < nfreq1; i++) {
	    tree0.dedisperse(chunk1.data + i*nfreq0*nt_chunk, nt_chunk, rp, scratch.data);
	    rp += tree0.nrstate;
	}

	lagbuf.apply_lags(chunk1);

	for (int j = 0; j < nfreq0; j++) {
	    tree1.dedisperse(chunk1.data + j*nt_chunk, nfreq0*nt_chunk, rp, scratch.data);
	    rp += tree1.nrstate;
	}

	// Second step: chunk0_big -> dedisperse -> chunk0
	
	big_tree.dedisperse(chunk0_big, rstate0.data, scratch.data);

	if (uflag)
	    reference_extract_odd_channels(chunk0_big, chunk0);
	else
	    chunk0.fill(chunk0_big);

	// Third step: compare chunk0 / chunk1

	// (arr0, arr1, name0, name1, axis_names)
	gputils::assert_arrays_equal(chunk0, chunk1, "unfactored", "factored", {"dm_brev","t"});
    }
}


static void test_tree_recursion(bool noisy=false)
{
    int rank = rand_int(0, 9);
    int rank0 = rand_int(0, rank+1);
    int rank1 = rank - rank0;
    int nt_chunk = rand_int(1, pow2(std::max(rank0,rank1)+1));
    int maxchunks = std::max(3L, 10000 / (pow2(rank) * nt_chunk));
    int nchunks = rand_int(1, maxchunks+1);
    bool uflag = rand_int(0, 2);
    
    test_tree_recursion(rank0, rank1, nt_chunk, nchunks, uflag, noisy);
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    bool noisy = true;
    
    for (int n = 0; n < 400; n++) {
	test_non_incremental_dedispersion(noisy);
	test_reference_lagbuf(noisy);
	test_reference_tree(noisy);
	test_tree_recursion(noisy);
    }

    cout << "test-reference-tree: pass" << endl;
    return 0;
}
