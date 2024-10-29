#include "../../include/pirate/internals/dedispersion_inbufs.hpp"
#include "../../include/pirate/internals/dedispersion_outbufs.hpp"
#include "../../include/pirate/internals/dedispersion_kernel_implementation.hpp"
    
namespace pirate
{
    using Inbuf = pirate::dedispersion_simple_inbuf<__half, false>;  // Lagged=false
    using Outbuf = pirate::dedispersion_simple_outbuf<__half>;
    
    INSTANTIATE_DEDISPERSION_KERNELS(__half2, Inbuf, Outbuf);
}
