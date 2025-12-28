#ifndef _PIRATE_TRACKERS_HPP
#define _PIRATE_TRACKERS_HPP

namespace pirate {
#if 0
}  // editor auto-indent
#endif


struct BandwidthTracker
{
    long nbytes_gmem = 0;  // GPU memory bandwidth, including PCIe transfers
    long nbytes_hmem = 0;  // Host memory bandwidth, including PCIe transfers
    long nbytes_h2g = 0;   // PCIe bandwidth, (host -> GPU) direction
    long nbytes_g2h = 0;   // PCIe bandwidth, (GPU -> host) direction
    long kernel_launches = 0;
    long memcpy_h2g_calls = 0;  // calls to cudaMemcpy() host to device
    long memcpy_g2h_calls = 0;  // calls to cudaMemcpy() device to host

    BandwidthTracker &operator+=(const BandwidthTracker &);
    BandwidthTracker &operator*=(long);
};


extern BandwidthTracker operator+(const BandwidthTracker &, const BandwidthTracker &);
extern BandwidthTracker operator*(long, const BandwidthTracker &);


}  // namespace pirate

#endif // _PIRATE_TRACKERS_HPP

