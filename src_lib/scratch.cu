#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <string>

#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/ChimeBeamformer.hpp"


using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Called by 'python -m pirate_frb scratch'. Intended for quick throwaway tests.
void scratch()
{
    uint host_map[256];

    for (int freq_mhz = 100; freq_mhz <= 800; freq_mhz += 100) {
        calculate_cl_index(host_map, double(freq_mhz), 60.0);

        cout << "freq=" << freq_mhz << " MHz, northmost_beam=60 deg:" << endl;
        for (int i = 0; i < 256; i++) {
            if (i > 0) cout << " ";
            cout << host_map[i];
        }
        cout << endl << endl;
    }
}


}  // namespace pirate

