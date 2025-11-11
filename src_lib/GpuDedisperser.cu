#include "../include/pirate/Dedisperser.hpp"
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/constants.hpp"  // xdiv(), pow2()
#include "../include/pirate/inlines.hpp"  // xdiv(), pow2()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Helper for GpuDedisperser constructor.
// Prevents constructor from segfaulting, if invoked with empty shared_ptr.
static DedispersionPlan *deref(const shared_ptr<DedispersionPlan> &p)
{
    if (!p)
        throw runtime_error("GpuDedisperser constructor called with empty shared_ptr");
    return p.get();
}


GpuDedisperser::GpuDedisperser(const shared_ptr<DedispersionPlan> &plan_) :
    plan(plan_),
    config(deref(plan_)->config)
{
    // Some features are not implemented yet.
    xassert(plan->g2g_rb_locs.size == 0);
    xassert(plan->h2h_rb_locs.size == 0);

    // There's some cut-and-paste between this constructor and the ReferenceDedisperser
    // constructor, but not enough to bother defining a common base class.
    
    this->dtype = config.dtype;
    this->input_rank = config.tree_rank;
    this->input_ntime = config.time_samples_per_chunk;
    this->total_beams = config.beams_per_gpu;
    this->beams_per_batch = config.beams_per_batch;
    this->gpu_ringbuf_nelts = plan->gmem_ringbuf_nseg * plan->nelts_per_segment;
    this->host_ringbuf_nelts = plan->hmem_ringbuf_nseg * plan->nelts_per_segment;
    this->nbatches = xdiv(total_beams, beams_per_batch);
    this->nstreams = config.num_active_batches;

    const DedispersionBufferParams &out_params = plan->stage2_dd_buf_params;
    this->output_ntrees = out_params.nbuf;
    this->output_rank = out_params.buf_rank;
    this->output_ntime = out_params.buf_ntime;
    this->output_ds_level = plan->stage2_ds_level;

    long nbits_per_segment = plan->nelts_per_segment * dtype.nbits;
    xassert_eq(nbits_per_segment, 8 * constants::bytes_per_gpu_cache_line);  // currently assumed in a few places
    
    // Construct, but do not allocate, the following members:
    //
    //   std::vector<DedispersionBuffer> stage1_dd_bufs;  // length nstreams
    //   std::vector<DedispersionBuffer> stage2_dd_bufs;  // length nstreams
    //   std::vector<std::shared_ptr<GpuDedispersionKernel>> stage1_dd_kernels;
    //   std::vector<std::shared_ptr<GpuDedispersionKernel>> stage2_dd_kernels;
    //   std::shared_ptr<GpuLaggedDownsamplingKernel> lds_kernel;
    
    for (long i = 0; i < nstreams; i++) {
        stage1_dd_bufs.push_back(DedispersionBuffer(plan->stage1_dd_buf_params));
        stage2_dd_bufs.push_back(DedispersionBuffer(plan->stage2_dd_buf_params));
    }

    for (const DedispersionKernelParams &kparams: plan->stage1_dd_kernel_params) {
        auto kernel = make_shared<GpuDedispersionKernel> (kparams);
        this->stage1_dd_kernels.push_back(kernel);
        this->bw_per_launch += kernel->bw_per_launch;
    }

    for (const DedispersionKernelParams &kparams: plan->stage2_dd_kernel_params) {
        auto kernel = make_shared<GpuDedispersionKernel> (kparams);
        this->stage2_dd_kernels.push_back(kernel);
        this->bw_per_launch += kernel->bw_per_launch;
    }

    this->lds_kernel = GpuLaggedDownsamplingKernel::make(plan->lds_params);
    this->bw_per_launch += lds_kernel->bw_per_launch;
}


void GpuDedisperser::allocate()
{
    if (this->is_allocated)
        throw runtime_error("double call to GpuDedisperser::allocate()");

    for (auto &buf: stage1_dd_bufs)
        buf.allocate(af_zero | af_gpu);
    
    for (auto &buf: stage2_dd_bufs)
        buf.allocate(af_zero | af_gpu);
    
    for (auto &kernel: this->stage1_dd_kernels)
        kernel->allocate();
    
    for (auto &kernel: this->stage2_dd_kernels)
        kernel->allocate();

    this->gpu_ringbuf = Array<void>(dtype, { gpu_ringbuf_nelts }, af_gpu | af_zero);
    this->host_ringbuf = Array<void>(dtype, { host_ringbuf_nelts }, af_rhost | af_zero);    
    this->lds_kernel->allocate();

    this->is_allocated = true;
}


void GpuDedisperser::launch(long ibatch, long it_chunk, long istream, cudaStream_t stream)
{
    const int BT = this->config.beams_per_gpu;            // total beams
    const int BB = this->config.beams_per_batch;          // beams per batch
    const int BA = this->config.num_active_batches * BB;  // active beams
    const int SB = constants::bytes_per_gpu_cache_line;   // bytes per segment

    xassert((ibatch >= 0) && (ibatch < nbatches));
    xassert((istream >= 0) && (istream < nstreams));
    xassert(it_chunk >= 0);
    xassert(is_allocated);
    
    long iframe = (it_chunk * BT) + (ibatch * BB);
    
    // Step 1: run LaggedDownsampler.
    lds_kernel->launch(stage1_dd_bufs.at(istream), ibatch, it_chunk, stream);

    // Step 2: run stage1 dedispersion kernels (output to ringbuf)
    for (uint i = 0; i < stage1_dd_kernels.size(); i++) {
        shared_ptr<GpuDedispersionKernel> kernel = stage1_dd_kernels.at(i);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage1_dd_bufs.at(istream).bufs.at(i);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.amb_rank), pow2(kp.dd_rank), kp.ntime });
        kernel->launch(dd_buf, this->gpu_ringbuf, ibatch, it_chunk, stream);
    }

    // Step 3: extra copying steps needed for early triggers.
    // Placeholder: not implemented yet.

    // Step 4: copy host <-> xfer
    // There is some cut-and-paste with ReferenceDedisperser, but not enough to bother refactoring.
    // FIXME: currently putting copies on the compute stream!
    // This is terrible for performance, but I'm just testing correctness for now.
           
    xassert(plan->host_ringbufs.size() == uint(plan->max_clag+1));
    xassert(plan->xfer_ringbufs.size() == uint(plan->max_clag+1));
    xassert_divisible(BT, BB);   // assert that length-BB copies don't "wrap"
    
    for (int clag = 0; clag <= plan->max_clag; clag++) {
        DedispersionPlan::Ringbuf &rb_host = plan->host_ringbufs.at(clag);
        DedispersionPlan::Ringbuf &rb_xfer = plan->xfer_ringbufs.at(clag);

        xassert(rb_host.nseg_per_beam == rb_xfer.nseg_per_beam);
        xassert(rb_host.rb_len == clag*BT + BA);
        xassert(rb_xfer.rb_len == 2*BA);

        if (rb_host.nseg_per_beam == 0)
            continue;
        
        char *hp = reinterpret_cast<char *> (this->host_ringbuf.data) + (rb_host.base_segment * SB);
        char *xp = reinterpret_cast<char *> (this->gpu_ringbuf.data) + (rb_xfer.base_segment * SB);
        
        long hsrc = (iframe + BA) % rb_host.rb_len;  // host src phase
        long hdst = (iframe) % rb_host.rb_len;       // host dst phase
        long xsrc = (iframe) % rb_xfer.rb_len;       // xfer src phase
        long xdst = (iframe + BA) % rb_xfer.rb_len;  // xfer dst phase
        
        long m = rb_host.nseg_per_beam * SB;  // nbytes per frame
        long n = BB * m;                      // nbytes to copy
        
        CUDA_CALL(cudaMemcpyAsync(xp + xdst*m, hp + hsrc*m, n, cudaMemcpyHostToDevice, stream));
        CUDA_CALL(cudaMemcpyAsync(hp + hdst*m, xp + xsrc*m, n, cudaMemcpyDeviceToHost, stream));
    }
    
    // Step 5: run stage2 dedispersion kernels (input from ringbuf)
    for (uint i = 0; i < stage2_dd_kernels.size(); i++) {
        shared_ptr<GpuDedispersionKernel> kernel = stage2_dd_kernels.at(i);
        const DedispersionKernelParams &kp = kernel->params;
        Array<void> dd_buf = stage2_dd_bufs.at(istream).bufs.at(i);

        // See comments in DedispersionKernel.hpp for an explanation of this reshape/transpose operation.
        dd_buf = dd_buf.reshape({ kp.beams_per_batch, pow2(kp.dd_rank), pow2(kp.amb_rank), kp.ntime });
        dd_buf = dd_buf.transpose({0,2,1,3});
        kernel->launch(this->gpu_ringbuf, dd_buf, ibatch, it_chunk, stream);
    }
}


}  // namespace pirate
