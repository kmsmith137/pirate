#include <iostream>
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/Dedisperser.hpp"

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Called by 'python -m pirate_frb scratch'. Intended for quick throwaway tests.
void scratch()
{
    cout << "pirate::scratch() called -- this is a place for quick throwaway tests" << endl;

    vector<string> filenames = {
        "configs/dedispersion/chime.yml",
        "configs/dedispersion/chime_sb1.yml",
        "configs/dedispersion/chime_sb2.yml",
        "configs/dedispersion/chime_sb2_et.yml",
        "configs/dedispersion/chord_sb0.yml",
        "configs/dedispersion/chord_sb1.yml",
        "configs/dedispersion/chord_sb2.yml",
        "configs/dedispersion/chord_sb2_et.yml"
    };

    for (const string &filename: filenames) {
        DedispersionConfig config = DedispersionConfig::from_yaml(filename);
        shared_ptr<DedispersionPlan> plan = make_shared<DedispersionPlan> (config);
        shared_ptr<GpuDedisperser> dd = make_shared<GpuDedisperser> (plan);
    }
}


}  // namespace pirate

