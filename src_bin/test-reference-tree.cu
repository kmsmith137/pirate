#include "../include/pirate/internals/ReferenceTree.hpp"
#include "../include/pirate/internals/ReferenceLagbuf.hpp"
#include "../include/pirate/internals/inlines.hpp"
#include "../include/pirate/internals/utils.hpp"  // dedisperse_non_incremental(), lag_non_incremental()

#include <ksgpu/rand_utils.hpp>    // rand_int()
#include <ksgpu/string_utils.hpp>  // tuple_str()
#include <ksgpu/test_utils.hpp>    // make_random_strides()

using namespace std;
using namespace pirate;
using namespace ksgpu;


// -------------------------------------------------------------------------------------------------
//
// Test dedisperse_non_incremental(), by comparing it to a brute force non-recursive algorithm.


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
    xassert((dm_brev >= 0) && (dm_brev < pow2(rank)));
    xassert((t0 >= 0) && (t0 < ntime));

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
    xassert(eps < 1.0e-5);
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
//
// Test 'class ReferenceLagbuf', by comparing it to a brute-force non-incremental implementation.


static void lag_non_incremental(Array<float> &arr, const Array<int> &lags)
{
    xassert(arr.ndim > 1);
    xassert(lags.shape_equals(arr.ndim-1, arr.shape));
    xassert(arr.is_fully_contiguous());

    long nchan = lags.size;
    long nt = arr.shape[arr.ndim-1];
    
    Array<float> arr_2d = arr.reshape({nchan, nt});
    Array<int> lags_1d = lags.clone();
    lags_1d = lags_1d.reshape({nchan});

    for (long i = 0; i < nchan; i++) {
	float *row = arr_2d.data + i*nt;
	long lag = lags_1d.data[i];

	lag = min(lag, nt);
	memmove(row+lag, row, (nt-lag) * sizeof(float));
	memset(row, 0, lag * sizeof(float));
    }
}


static void test_reference_lagbuf(const Array<int> &lags, const vector<long> data_strides, int nt_chunk, int nchunks)
{
    cout << "test_reference_lagbuf:"
	 << " lags.shape=" << lags.shape_str()
	 << ", lags.strides=" << lags.stride_str()
	 << ", data_strides=" << tuple_str(data_strides)
	 << ", nt_chunk=" << nt_chunk
	 << ", nchunks=" << nchunks << endl;

    xassert(long(data_strides.size()) == lags.ndim+1);
    xassert(data_strides[lags.ndim] == 1);
    
    int d = lags.ndim;
    int nt_tot = nt_chunk * nchunks;

    // Creating axis names feels silly, but assert_arrays_equal() requires them.
    vector<string> axis_names(d+1);
    for (int i = 0; i < d; i++)
	axis_names[i] = "ix" + to_string(i);
    axis_names[d] = "t";
    
    vector<long> shape_lg(d+1);
    vector<long> shape_sm(d+1);
    
    for (int i = 0; i < d; i++)
	shape_lg[i] = shape_sm[i] = lags.shape[i];

    shape_lg[d] = nt_tot;
    shape_sm[d] = nt_chunk;

    Array<float> arr_lg(shape_lg, af_uhost | af_random);
    Array<float> arr_lg_ref = arr_lg.clone();
    lag_non_incremental(arr_lg_ref, lags);
    
    Array<float> arr_sm(shape_sm, data_strides, af_uhost | af_zero);  // note strides
    Array<float> arr_sm_ref(shape_sm, af_uhost | af_zero);

    ReferenceLagbuf rbuf(lags, nt_chunk);
    
    for (int c = 0; c < nchunks; c++) {
	// Extract chunk (arr_lg) -> (arr_sm)
	Array<float> s = arr_lg.slice(d, c*nt_chunk, (c+1)*nt_chunk);
	arr_sm.fill(s);

	// Apply lagbuf
	rbuf.apply_lags(arr_sm);

	// Extract chunk (arr_lg_ref) -> (arr_sm_ref)
	s = arr_lg_ref.slice(d, c*nt_chunk, (c+1)*nt_chunk);
	arr_sm_ref.fill(s);

	// Compare arr_sm, arr_sm_ref.
	ksgpu::assert_arrays_equal(arr_sm, arr_sm_ref, "incremental", "non-incremental", axis_names);
    }
}


static void test_reference_lagbuf()
{
    // Number of dimensions in 'lags' array
    int nd = rand_int(1, 4);

    // lags.shape + (nt_chunk, nchunks)
    vector<long> v = random_integers_with_bounded_product(nd+2, 10000);
    int nt_chunk = v[nd];
    int nchunks = v[nd+1];
    
    vector<long> lag_shape(nd);
    vector<long> data_shape(nd+1);
    memcpy(&data_shape[0], &v[0], (nd+1) * sizeof(long));
    memcpy(&lag_shape[0], &v[0], nd * sizeof(long));

    vector<long> lag_strides = make_random_strides(lag_shape);
    vector<long> data_strides = make_random_strides(data_shape, 1);   // time axis guaranteed continuous

    Array<int> lags(lag_shape, lag_strides, af_uhost | af_zero);
    double maxlog = log(1.5 * nt_chunk * nchunks);
    
    for (auto ix = lags.ix_start(); lags.ix_valid(ix); lags.ix_next(ix)) {
	double t = rand_uniform(-1.0, maxlog);
	lags.at(ix) = int(exp(t));
    }

    test_reference_lagbuf(lags, data_strides, nt_chunk, nchunks);
}


// -------------------------------------------------------------------------------------------------
//
// Test 'class ReferenceTree', by comparing its incremental dedispersion to dedisperse_non_incremental().


static void test_reference_tree(const vector<long> &shape, const vector<long> &strides, int freq_axis, int nchunks)
{
    cout << "test_reference_tree: shape=" << tuple_str(shape)
	 << ", strides=" << tuple_str(strides)
	 << ", freq_axis=" << freq_axis
	 << ", nchunks=" << nchunks << endl;

    int ndim = shape.size();
    int nfreq = shape.at(freq_axis);
    int nt_chunk = shape.at(ndim-1);
    int nt_tot = nt_chunk * nchunks;
    
    // Input data (multiple chunks)
    vector<long> big_shape = shape;
    big_shape[ndim-1] *= nchunks;
    Array<float> arr0(big_shape, af_uhost | af_random);

    // Let's apply non-incremental dedispersion to arr0.
    // Step 1. transpose to axis ordering (spectators, freq, time).
    
    vector<int> ax_swap(ndim);
    for (int d = 0; d < ndim; d++)
	ax_swap[d] = d;
    ax_swap[freq_axis] = ndim-2;
    ax_swap[ndim-2] = freq_axis;

    Array<float> arr1 = arr0.transpose(ax_swap);

    // Step 2. reshape to (nouter, nfreq, nt_tot), with precisely one spectator axis.

    int nouter = 1;
    for (int d = 0; d < ndim-2; d++)
	nouter *= arr1.shape[d];
    
    arr1 = arr1.clone();  // note deep copy here
    arr1 = arr1.reshape({nouter, nfreq, nt_tot});

    // Step 3. loop over outer spectator axis, and call dedisperse_non_incremental().
    
    for (int i = 0; i < nouter; i++) {
	Array<float> view_2d = arr1.slice(0, i);  // shape (nfreq, nt_tot)
	dedisperse_non_incremental(view_2d);
    }

    // Step 4. transpose back to original axis ordering.
    // (This concludes the non-incremental dedispersion.)

    vector<long> transposed_shape = big_shape;
    std::swap(transposed_shape[freq_axis], transposed_shape[ndim-2]);
    
    arr1 = arr1.reshape(transposed_shape);
    arr1 = arr1.transpose(ax_swap);

    // Now apply incremental dedispersion in chunks, and compare.
    
    ReferenceTree rtree(shape.size(), &shape[0], freq_axis);
    Array<float> chunk(shape, strides, af_uhost | af_zero);

    // Apply incremental dedispersion to arr0 (in place)
    for (int c = 0; c < nchunks; c++) {
	Array<float> slice = arr0.slice(ndim-1, c*nt_chunk, (c+1)*nt_chunk);
	chunk.fill(slice);
	rtree.dedisperse(chunk);
	slice.fill(chunk);
    }

    // Need axis names for assert_arrays_equal().
    vector<string> axis_names(ndim);
    for (int d = 0; d < ndim-2; d++)
	axis_names[d] = "ispec" + to_string(d);
    axis_names[ndim-2] = "dm_brev";
    axis_names[ndim-1] = "t";
    std::swap(axis_names[freq_axis], axis_names[ndim-2]);

    ksgpu::assert_arrays_equal(arr1, arr0, "non-incremental", "incremental", axis_names);
}


static void test_reference_tree()
{
    int rank = rand_int(1, 9);
    int ndim = rand_int(2, 6);
    int freq_axis = rand_int(0, ndim-1);

    vector<long> shape = ksgpu::random_integers_with_bounded_product(ndim, 30000 / pow2(rank));
    int nchunks = shape[freq_axis];
    shape[freq_axis] = pow2(rank);

    vector<long> strides = ksgpu::make_random_strides(shape, 1);  // ncontig=1

    test_reference_tree(shape, strides, freq_axis, nchunks);
}


// -------------------------------------------------------------------------------------------------


static void test_tree_recursion(int rank0, int rank1, int nt_chunk, int nchunks)
{
    cout << "test_reference_tree_recursion: rank0=" << rank0 << ", rank1=" << rank1
	 << ", nt_chunk=" << nt_chunk << ", nchunks=" << nchunks << endl;

    int rank_tot = rank0 + rank1;
    
    check_rank(rank0, "test_tree_recursion [rank0]");
    check_rank(rank1, "test_tree_recursion [rank1]");
    check_rank(rank_tot, "test_tree_recursion [rank_tot]");
	       
    int nfreq_tot = pow2(rank_tot);
    int nfreq0 = pow2(rank0);
    int nfreq1 = pow2(rank1);

    Array<int> lags({nfreq1,nfreq0}, af_uhost | af_zero);
    for (int i = 0; i < nfreq1; i++)
	for (int j = 0; j < nfreq0; j++)
	    lags.at({i,j}) = rb_lag(i, j, rank0, rank1, false);  // uflag=false

    ReferenceTree big_tree({nfreq_tot, nt_chunk});
    ReferenceTree tree0({nfreq1, nfreq0, nt_chunk}, 1);
    ReferenceTree tree1({nfreq1, nfreq0, nt_chunk}, 0);
    ReferenceLagbuf lagbuf(lags, nt_chunk);

    for (int c = 0; c < nchunks; c++) {
	Array<float> chunk0({nfreq_tot, nt_chunk}, af_uhost | af_random);

	// "Two-step" dedispersion.
	Array<float> chunk1 = chunk0.clone();
	chunk1 = chunk1.reshape({nfreq1, nfreq0, nt_chunk});
	tree0.dedisperse(chunk1);
	lagbuf.apply_lags(chunk1);
	tree1.dedisperse(chunk1);
	chunk1 = chunk1.reshape({nfreq_tot, nt_chunk});

	// "One-step" dedispersion.
	big_tree.dedisperse(chunk0);

	// Third step: compare chunk0 / chunk1
	// (arr0, arr1, name0, name1, axis_names)
	ksgpu::assert_arrays_equal(chunk0, chunk1, "1-step", "2-step", {"dm_brev","t"});
    }
}


static void test_tree_recursion()
{
    int rank = rand_int(0, 9);
    int rank0 = rand_int(0, rank+1);
    int rank1 = rank - rank0;
    int nt_chunk = rand_int(1, pow2(std::max(rank0,rank1)+1));
    int maxchunks = std::max(3L, 10000 / (pow2(rank) * nt_chunk));
    int nchunks = rand_int(1, maxchunks+1);
    
    test_tree_recursion(rank0, rank1, nt_chunk, nchunks);
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    bool noisy = true;

    for (int n = 0; n < 400; n++) {
	test_non_incremental_dedispersion(noisy);
	test_reference_lagbuf();
	test_reference_tree();
	test_tree_recursion();
    }

    cout << "test-reference-tree: pass" << endl;
    return 0;
}
