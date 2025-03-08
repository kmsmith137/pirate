#include "../../include/pirate/internals/dedispersion_iobufs.hpp"
#include "../../include/pirate/internals/dedispersion_kernel_implementation.hpp"
    
namespace pirate
{
    using Inbuf = pirate::dedispersion_simple_inbuf<float, false>;  // Lagged=false
    using Outbuf = pirate::dedispersion_ring_outbuf<float>;
    
    INSTANTIATE_DEDISPERSION_KERNELS(float, Inbuf, Outbuf);
}
