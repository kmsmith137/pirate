#include "../../include/pirate/cuda_kernels/dedispersion.hpp"
#include "../../include/pirate/cuda_kernels/dedispersion_iobufs.hpp"
    
namespace pirate
{
    using Inbuf = pirate::dedispersion_simple_inbuf<float, false>;  // Lagged=false
    using Outbuf = pirate::dedispersion_simple_outbuf<float>;
    
    INSTANTIATE_DEDISPERSION_KERNELS(float, Inbuf, Outbuf);
}
