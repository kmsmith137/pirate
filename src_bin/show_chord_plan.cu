// FIXME after I define a YAML format for configs, this program can go away.
// (To be replaced by a utility like 'show_dedispersion_plan <config.yaml>')

#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/internals/ReferenceDedisperser.hpp"
#include "../include/pirate/internals/utils.hpp"  // make_chord_dedispersion_config()

using namespace std;
using namespace pirate;


int main(int argc, char **argv)
{
    // Usage: show_chord_plan [compressed_dtype] [uncompressed_dtype]
    // Default dtypes are int8 (compressed) + float32 (uncompressed)

    string compressed_dtype = (argc >= 2) ? argv[1] : "int8";
    string uncompressed_dtype = (argc >= 3) ? argv[2] : "float32";
    DedispersionConfig config = make_chord_dedispersion_config(compressed_dtype, uncompressed_dtype);

    cout << "DedispersionPlan" << endl;
    auto plan = make_shared<DedispersionPlan> (config);
    plan->print(cout, 4);

#if 0
    cout << "\nReferenceDedisperser" << endl;
    auto dedisp = make_shared<ReferenceDedisperser> (plan, 3);
    dedisp->print(cout, 4);
#endif
	
    return 0;
}
