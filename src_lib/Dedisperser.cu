#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/internals/CacheLineRingbuf.hpp"

#include <gputils/cuda_utils.hpp>  // CUDA_CALL()

using namespace std;
using namespace gputils;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Helper for Dedisperser constructor.
// Prevents constructor from segfaulting, if invoked with empty shared_ptr.
static DedispersionPlan *deref(const shared_ptr<DedispersionPlan> &p)
{
    if (!p)
	throw runtime_error("Dedisperser constructor called with empty shared_ptr");
    return p.get();
}


Dedisperser::Dedisperser(const DedispersionConfig &config_) :
    config(config_)
{
    this->plan = make_shared<DedispersionPlan> (config_);
    this->cache_line_ringbuf = plan->cache_line_ringbuf;
}


Dedisperser::Dedisperser(const shared_ptr<DedispersionPlan> &plan_) :
    config(deref(plan_)->config)
{
    this->plan = plan_;
    this->cache_line_ringbuf = plan->cache_line_ringbuf;
}


void Dedisperser::allocate()
{
    assert(!host_buffer);
    assert(!gpu_buffer);

    int mflag = config.use_hugepages ? af_mmap_huge : 0;
    this->host_buffer = af_alloc<char> (plan->hmem_nbytes_ringbuf, af_rhost | af_zero | mflag);
    this->gpu_buffer = af_alloc<char> (plan->gmem_nbytes_tot, af_gpu | af_zero);
}


void Dedisperser::launch_h2g_copies(ssize_t chunk, int beam, cudaStream_t stream)
{
    assert(chunk >= 0);
    assert((beam >= 0) && (beam < config.beams_per_gpu));
    assert((beam % config.beams_per_batch) == 0);

    int active_beams = config.beams_per_batch * config.num_active_batches;

    ssize_t isrc = chunk * config.beams_per_gpu + beam;  // flattened
    int idst = beam % active_beams;                      // wrapped

    for (unsigned int rb_lag = 0; rb_lag < cache_line_ringbuf->buffers.size(); rb_lag++) {
	const auto &buf = cache_line_ringbuf->buffers[rb_lag];

	if (buf.on_gpu)
	    continue;
	if (buf.total_ringbuf_nbytes == 0)
	    continue;

	ssize_t i = isrc % (rb_lag * config.beams_per_gpu);  // wrapped
	ssize_t n = config.beams_per_batch * buf.total_nbytes_per_beam_per_chunk;
	
	char *src = host_buffer.get() + buf.hmem_ringbuf_byte_offset + (i * buf.total_nbytes_per_beam_per_chunk);
	char *dst = gpu_buffer.get() + buf.staging_buffer_byte_offset + (idst * buf.total_nbytes_per_beam_per_chunk);

	CUDA_CALL(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream));
    }
}


void Dedisperser::launch_g2h_copies(ssize_t chunk, int beam, cudaStream_t stream)
{
    assert(chunk >= 0);
    assert((beam >= 0) && (beam < config.beams_per_gpu));
    assert((beam % config.beams_per_batch) == 0);

    int active_beams = config.beams_per_batch * config.num_active_batches;

    int isrc = active_beams + (beam % active_beams);      // wrapped
    ssize_t idst = chunk * config.beams_per_gpu + beam;   // flattened

    for (unsigned int rb_lag = 0; rb_lag < cache_line_ringbuf->buffers.size(); rb_lag++) {
	const auto &buf = cache_line_ringbuf->buffers[rb_lag];

	if (buf.on_gpu)
	    continue;
	if (buf.total_ringbuf_nbytes == 0)
	    continue;

	ssize_t i = idst % (rb_lag * config.beams_per_gpu);  // wrapped
	ssize_t n = config.beams_per_batch * buf.total_nbytes_per_beam_per_chunk;
	
	char *src = gpu_buffer.get() + buf.staging_buffer_byte_offset + (isrc * buf.total_nbytes_per_beam_per_chunk);
	char *dst = host_buffer.get() + buf.hmem_ringbuf_byte_offset + (i * buf.total_nbytes_per_beam_per_chunk);

	CUDA_CALL(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream));
    }    
}
    

}  // namespace pirate
