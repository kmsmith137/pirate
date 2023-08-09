#include "../include/pirate/internals/Directory.hpp"
#include <stdexcept>

using namespace std;

namespace pirate {
#if 0
}   // compiler pacifier
#endif


Directory::Directory(const string &dirname_) :
    dirname(dirname_)
{
    this->dirp = opendir(dirname.c_str());
    
    if (!dirp)
	throw runtime_error(dirname + ": opendir() failed: " + strerror(errno));
}


Directory::~Directory()
{
    if (dirp) {
	closedir(dirp);
	dirp = nullptr;
    }
}


dirent *Directory::read_next()
{
    assert(dirp != nullptr);

    dirent *entry = readdir(dirp);

    if (!entry && errno)
	throw runtime_error(dirname + ": readdir() failed: " + strerror(errno));
    
    return entry;
}


}  // namespace pirate
