#include "../include/pirate/DedispersionPlan.hpp"

using namespace std;
using namespace pirate;


int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "usage: show_dedispersion_plan <dedispersion_config.yml>" << endl;
        exit(2);
    }

    auto config = DedispersionConfig::from_yaml(argv[1]);
    
#if 0
    cout << config.to_yaml_string() << endl;
#endif
    
    auto plan = make_shared<DedispersionPlan> (config);
    plan->print(cout, 4);
    
#if 0
    cout << "\nReferenceDedisperser" << endl;
    auto dedisp = make_shared<ReferenceDedisperser> (plan, 3);
    dedisp->print(cout, 4);
#endif

    return 0;
}
