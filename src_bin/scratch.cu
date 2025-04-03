// A placeholder file (already integrated into Makefile) for debugging

#include <ksgpu.hpp>
// #include "../include/pirate/file_utils.hpp"

using namespace std;
// using namespace ksgpu;
// using namespace pirate;


int main(int argc, char **argv)
{
    double x = 0.0;
    struct timeval tv0 = ksgpu::get_time();
    
    for (long i = 0; i < 30*1000*1000; i++)
	x += ksgpu::time_since(tv0);

    return 0;
}
