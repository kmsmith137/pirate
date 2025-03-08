#include "../include/pirate/DedispersionBuffer.hpp"
#include "../include/pirate/constants.hpp"
#include "../include/pirate/inlines.hpp"  // pow2(), xdiv()

#include <ksgpu/xassert.hpp>
#include <ksgpu/string_utils.hpp>  // tuple_str()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void DedispersionBufferParams::print(std::ostream &os, int indent) const
{
    print_kv("dtype", this->dtype, os, indent);
    print_kv("beams_per_batch", this->beams_per_batch, os, indent);
    print_kv("nbuf", this->nbuf, os, indent);
    print_kv("buf_rank", ksgpu::tuple_str(this->buf_rank), os, indent);
    print_kv("buf_ntime", ksgpu::tuple_str(this->buf_ntime), os, indent);
}


long DedispersionBufferParams::get_nelts() const
{
    xassert_eq((long)buf_rank.size(), nbuf);
    xassert_eq((long)buf_ntime.size(), nbuf);
    
    long ret = 0;
    for (long i = 0; i < nbuf; i++)
	ret += beams_per_batch * pow2(buf_rank[i]) * buf_ntime[i];
    
    return ret;
}


void DedispersionBufferParams::validate() const
{
    xassert(!dtype.is_empty());
    xassert(beams_per_batch > 0);
    xassert(nbuf > 0);
    
    xassert_eq(long(buf_rank.size()), nbuf);
    xassert_eq(long(buf_ntime.size()), nbuf);
    
    if ((dtype != Dtype::native<float>()) && (dtype != Dtype::native<__half>()))
	throw runtime_error("DedispersionBufferParams: unsupported dtype: " + dtype.str());    

    long nt_divisor = xdiv(8 * constants::bytes_per_gpu_cache_line, dtype.nbits);;
    
    for (long i = 0; i < nbuf; i++) {
	xassert((buf_rank[i] >= 0) && (buf_rank[i] <= 16));
	xassert(buf_ntime[i] > 0);
	xassert_divisible(buf_ntime[i], nt_divisor);
    }
}


// -------------------------------------------------------------------------------------------------


DedispersionBuffer::DedispersionBuffer(const DedispersionBufferParams &params_)
    : params(params_)
{
    params.validate();
}


void DedispersionBuffer::allocate(int aflags)
{
    params.validate(); // paranoid
    
    if (is_allocated)
	throw runtime_error("double call to DedispersionBuffer::allocate()");

    long nb = params.beams_per_batch;
    long nbuf = params.nbuf;
    
    // We use a specific memory layout for the arrays, where the beam axis is
    // non-contiguous but the inner two axes are contiguous. This layout (or
    // something similar) is currently required by GpuLaggedDownsamplingKernel.
    // See comments in GpuLaggedDownsamplingKernel.cu for details.

    long bstride = 0;
    for (long i = 0; i < nbuf; i++)
	bstride += pow2(params.buf_rank[i]) * params.buf_ntime[i];

    this->ref = Array<void> (params.dtype, {nb,bstride}, aflags);
    this->bufs.resize(nbuf);

    long j = 0;
    for (long i = 0; i < nbuf; i++) {
	long nr = pow2(params.buf_rank[i]);
	long nt = params.buf_ntime[i];
	Array<void> a = ref.slice(1, j, j + nr*nt);
	this->bufs[i] = a.reshape({ nb, nr, nt });
	j += nr*nt;
    }

    xassert(bstride == j);
    this->is_allocated = true;
}


bool DedispersionBuffer::on_host() const
{
    xassert(is_allocated);
    return ref.on_host();
}


bool DedispersionBuffer::on_gpu() const
{
    xassert(is_allocated);
    return ref.on_gpu();
}


}  // namespace pirate
