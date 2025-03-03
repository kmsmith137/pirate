#ifndef _PIRATE_INTERNALS_DIRECTORY_HPP
#define _PIRATE_INTERNALS_DIRECTORY_HPP

#include <sys/types.h>
#include <dirent.h>
#include <string>


namespace pirate {
#if 0
}  // compiler pacifier
#endif


// Directory: RAII wrapper for (DIR *).
//
// Note: instead of using this class, you may prefer pirate::listdir(),
// which is declared in file_utils.hpp and returns a vector<string>.


struct Directory
{
    std::string dirname;
    DIR *dirp = nullptr;

    Directory(const std::string &dirname);
    ~Directory();

    // Returns NULL when there are no more entries to read.
    // Reminder: dirent->d_name is the NULL-terminated filename.
    struct dirent *read_next();
    
    // The Directory class is noncopyable, but if copy semantics are needed, you can do
    //   shared_ptr<File> fp = make_shared<File> (filename, oflags, mode);
    Directory(const Directory &) = delete;
    Directory &operator=(const Directory &) = delete;
    
    // FIXME: write boilerplate so that the following syntax is valid:
    //
    //     Directory dir(dirname);
    //     for (dirent *entry: dir)
    //        ...
};


}  // namespace pirate

#endif  // _PIRATE_INTERNALS_DIRECTORY_HPP
