#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <string>

#include "../include/pirate/AssembledFrame.hpp"


using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Called by 'python -m pirate_frb scratch'. Intended for quick throwaway tests.
void scratch()
{
    auto f = AssembledFrame::make_random();
    f->write_asdf("test.asdf");
    cout << "Wrote test.asdf" << endl;
}


}  // namespace pirate

