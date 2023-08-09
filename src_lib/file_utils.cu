#include "../include/pirate/internals/file_utils.hpp"
#include "../include/pirate/internals/Directory.hpp"

#include <iostream>
#include <sys/stat.h>
#include <fts.h>

using namespace std;

namespace pirate {
#if 0
}   // compiler pacifier
#endif


bool file_exists(const string &filename)
{
    struct stat s;

    int err = stat(filename.c_str(), &s);
    if (err >= 0)
        return true;
    if (errno == ENOENT)
        return false;

    throw runtime_error(filename + ": " + strerror(errno));
}


bool is_directory(const string &filename)
{
    struct stat s;

    int err = stat(filename.c_str(), &s);
    if (err < 0)
	throw runtime_error(filename + ": " + strerror(errno));

    return S_ISDIR(s.st_mode);
}


bool is_empty_directory(const string &dirname)
{
    Directory dir(dirname);

    for (;;) {
	struct dirent *entry = dir.read_next();

	if (!entry)
	    return true;
	if (!strcmp(entry->d_name, "."))
	    continue;
	if (!strcmp(entry->d_name, ".."))
	    continue;
	
	return false;
    }
}


void makedir(const string &filename, bool throw_exception_if_directory_exists, mode_t mode)
{
    int err = mkdir(filename.c_str(), mode);

    if (err >= 0)
	return;
    if (throw_exception_if_directory_exists || (errno != EEXIST))
	throw runtime_error(filename + ": mkdir() failed: " + strerror(errno));
    
    // If we get here, then mkdir() failed with EEXIST, and throw_exception_if_directory_exists=false.
    // We still throw an exception if the file is not a directory.

    struct stat s;
    err = stat(filename.c_str(), &s);

    // A weird corner case.
    if (err < 0)
	throw runtime_error(filename + ": mkdir() returned EEXIST but stat() failed, not sure what is going on");

    if (!S_ISDIR(s.st_mode))
	throw runtime_error(filename + ": file exists but is not a directory");
}


vector<string> listdir(const string &dirname)
{
    vector<string> filenames;

    Directory dir(dirname);

    for (;;) {
	struct dirent *entry = dir.read_next();
	
	if (entry)
	    filenames.push_back(entry->d_name);
	else
	    return filenames;
    }
}


size_t disk_space_used(const string &dirname)
{
    FTS* hierarchy;
    char** paths;
    size_t totalsize = 0;

    paths = (char**)alloca(2 * sizeof(char*));
    paths[0] = (char*)dirname.c_str();
    paths[1] = NULL;
    hierarchy = fts_open(paths, FTS_LOGICAL, NULL);
    if (!hierarchy) {
        throw runtime_error(dirname + ": fts_open() failed: " + strerror(errno));
    }
    while (1) {
        FTSENT *entry = fts_read(hierarchy);
        if (!entry && (errno == 0))
            break;
        if (!entry)
            throw runtime_error(dirname + ": fts_read() failed: " + strerror(errno));
        if (entry->fts_info & FTS_F) {
            // The entry is a file.
            struct stat *st = entry->fts_statp;
            totalsize += st->st_size;
            cout << "path " << entry->fts_path << " size " << st->st_size << endl;
        }
        
    }
    fts_close(hierarchy);
    return totalsize;
}


}   // namespace pirate
