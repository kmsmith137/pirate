// A placeholder file (already integrated into Makefile) for debugging

// #include <ksgpu.hpp>
#include "../include/pirate/ReferenceLagbuf.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;


int main(int argc, char **argv)
{
    Array<int> lags({3,256}, af_uhost);
    for (int i = 0; i < 3; i++)
	for (int j = 0; j < 256; j++)
	    lags.at({i,j}) = j;

    vector<ReferenceLagbuf> v;
    for (int i = 0; i < 5; i++)
	v.push_back({lags, 100});
   
    return 0;
}
