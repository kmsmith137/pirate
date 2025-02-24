// A placeholder file (already integrated into Makefile) for debugging

#include <ksgpu.hpp>
#include "../include/pirate/DedispersionConfig.hpp"
#include "../include/pirate/internals/YamlFile.hpp"

using namespace std;
using namespace ksgpu;
using namespace pirate;


int main(int argc, char **argv)
{
    YamlFile f("/home/kmsmith/git_blue/pirate/src_bin/x.yml");
    
    YamlFile ets = f["early_triggers"];
    for (long i = 0; i < ets.size(); i++) {
	YamlFile et = ets[i];
	int ds = et.get_scalar<int> ("ds_level");
	int tr = et.get_scalar<int> ("tree_rank");
	cout << i << " " << ds << " " << tr << endl;
	et.check_for_invalid_keys();
    }
	
    return 0;
}
