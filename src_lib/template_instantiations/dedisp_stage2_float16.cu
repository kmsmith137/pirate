#include "../../include/pirate/cuda_kernels/dedispersion.hpp"
#include "../../include/pirate/cuda_kernels/dedispersion_iobufs.hpp"
    
namespace pirate
{
    using Inbuf = pirate::dedispersion_ring_inbuf<__half>;   // lagged
    using Outbuf = pirate::dedispersion_simple_outbuf<__half>;
    
    INSTANTIATE_DEDISPERSION_KERNELS(__half2, Inbuf, Outbuf);
}
