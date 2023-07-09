// A placeholder file (already integrated into Makefile) for debugging

#include <gputils.hpp>
#include "../include/pirate/DedispersionConfig.hpp"

using namespace std;
using namespace gputils;
using namespace pirate;


int main(int argc, char **argv)
{
    for (int i = 0; i < 20; i++) {
	auto config = DedispersionConfig::make_random();
	cout << "Iteration " << i << endl;
	config.print(cout, 4);
    }
    
    return 0;
}
