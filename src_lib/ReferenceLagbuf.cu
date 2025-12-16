#include "../include/pirate/ReferenceLagbuf.hpp"

#include <cstring>
#include <ksgpu/rand_utils.hpp>    // rand_int(), rand_uniform(), random_integers_with_bounded_product()
#include <ksgpu/string_utils.hpp>  // tuple_str()
#include <ksgpu/test_utils.hpp>    // make_random_strides(), assert_arrays_equal()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// ReferenceLagbuf constructor.


static long _sum_lags(const int *lags, int nd, const long *shape, const long *strides)
{
    xassert(nd > 0);
    
    long n = shape[0];
    long s = strides[0];    
    long ret = 0;
    
    if (nd == 1) {
        for (long i = 0; i < n; i++) {
            int lag = lags[i*s];
            xassert(lag >= 0);
            ret += lag;
        }
    }
    else {
        for (long i = 0; i < n; i++)
            ret += _sum_lags(lags + i*s, nd-1, shape+1, strides+1);
    }

    return ret;
}


ReferenceLagbuf::ReferenceLagbuf(const ksgpu::Array<int> &lags_, int ntime_) :
    lags(lags_), ntime(ntime_)
{
    xassert(ntime > 0);
    xassert(lags.ndim > 0);

    int d = lags.ndim;
    long nr = _sum_lags(lags.data, d, lags.shape, lags.strides);
    
    expected_shape.resize(d+1);
    expected_shape[d] = ntime;
    for (int i = 0; i < d; i++)
        expected_shape[i] = lags.shape[i];

    this->pstate = Array<float> ({nr}, af_uhost | af_zero);
    this->scratch = Array<float> ({ntime}, af_uhost | af_zero);
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceLagbuf::apply_lags()


// Applies (N-1)-dimensional lag array to N-dimensional data array.
static float *_apply_lags(float *data, const int *lags, float *pstate, float *scratch,
                          int data_ndim, const long *data_shape, const long *data_strides,
                          const long *lag_strides)
{
    if (data_ndim == 1) {
        int lag = *lags;

        if (lag <= 0)
            return pstate;

        int ntime = data_shape[0];
        int n = std::min(lag, ntime);
        
        // Inefficient for (lag >= ntime).
        memcpy(scratch, data + (ntime-n), n * sizeof(float));
        memmove(data+n, data, (ntime-n) * sizeof(float));
        memcpy(data, pstate, n * sizeof(float));
        memmove(pstate, pstate+n, (lag-n)*sizeof(float));
        memcpy(pstate + (lag-n), scratch, n * sizeof(float));
        
        return pstate + lag;
    }

    long n = data_shape[0];
    long ds = data_strides[0];
    long ls = lag_strides[0];

    for (long i = 0; i < n; i++) {
        pstate = _apply_lags(data + i*ds,     // data
                             lags + i*ls,     // lags
                             pstate,          // pstate
                             scratch,         // scratch
                             data_ndim-1,     // data_ndim
                             data_shape+1,    // shape
                             data_strides+1,  // data_strides
                             lag_strides+1);  // lag_strides
    }

    return pstate;
}


void ReferenceLagbuf::apply_lags(ksgpu::Array<float> &arr) const
{
    if (!arr.shape_equals(expected_shape)) {
        stringstream ss;
        ss << "ReferenceLagbuf::apply_lags(): arr.shape=" << arr.shape_str()
           << ", expected_shape=" << ksgpu::tuple_str(expected_shape);
        throw runtime_error(ss.str());
    }

    xassert(arr.shape_equals(expected_shape));
    xassert((arr.shape[arr.ndim-1] == 1) || (arr.strides[arr.ndim-1] == 1));
    
    float *pstate_end = _apply_lags(arr.data, lags.data, pstate.data, scratch.data, arr.ndim, arr.shape, arr.strides, lags.strides);                            
    xassert(pstate_end == pstate.data + pstate.size);
}


// -------------------------------------------------------------------------------------------------
//
// ReferenceLagbuf::test()


// Static member function
void ReferenceLagbuf::test()
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

    // Apply lags non-incrementally (not using ReferenceLagbuf).
    {
        xassert(arr_lg_ref.ndim > 1);
        xassert(lags.shape_equals(arr_lg_ref.ndim-1, arr_lg_ref.shape));
        xassert(arr_lg_ref.is_fully_contiguous());

        long nchan = lags.size;
        long nt = arr_lg_ref.shape[arr_lg_ref.ndim-1];
    
        Array<float> arr_2d = arr_lg_ref.reshape({nchan, nt});
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


}  // namespace pirate
