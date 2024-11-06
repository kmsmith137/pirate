#include "../../include/pirate/internals/dedispersion_inbufs.hpp"
#include "../../include/pirate/internals/dedispersion_outbufs.hpp"
#include "../../include/pirate/internals/dedispersion_kernel_implementation.hpp"
    
namespace pirate
{
    using Inbuf = pirate::dedispersion_simple_inbuf<float, false>;  // Lagged=true
    using Outbuf = pirate::dedispersion_ring_outbuf<float>;
    
    INSTANTIATE_DEDISPERSION_KERNELS(float, Inbuf, Outbuf);
}
