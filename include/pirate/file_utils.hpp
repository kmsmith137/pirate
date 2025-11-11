#ifndef _PIRATE_FILE_UTILS_HPP
#define _PIRATE_FILE_UTILS_HPP

#include <string>
#include <vector>
#include <fcntl.h>  // open flags (O_RDONLY, etc.)
#include <sys/types.h>
#include <dirent.h>

namespace pirate {
#if 0
}  // compiler pacifier
#endif


extern bool file_exists(const std::string &filename);
extern bool is_directory(const std::string &filename);
extern bool is_empty_directory(const std::string &dirname);

// Note: umask will be applied to 'mode'
extern void makedir(const std::string &filename,
                    bool throw_exception_if_directory_exists = true,
                    mode_t mode = 0777);

// Note: includes '.' and '..'
extern std::vector<std::string> listdir(const std::string &dirname);

extern void delete_file(const std::string &filename);

extern long disk_space_used(const std::string &dirname);


// -------------------------------------------------------------------------------------------------
//
// File: RAII wrapper for unix file descriptor.


struct File
{
    std::string filename;
    int fd = -1;
    
    // Constructor opens file.
    // Suggest oflags = (O_RDONLY) for reading, and (O_WRONLY | O_CREAT | O_TRUNC) for writing.
    
    File(const std::string &filename, int oflags, int mode=0644);
    ~File();

    void write(const void *p, long nbytes);
    // FIXME add more member functions, including read().
    
    // The File class is noncopyable, but if copy semantics are needed, you can do
    //   shared_ptr<File> fp = make_shared<File> (filename, oflags, mode);
    
    File(const File &) = delete;
    File &operator=(const File &) = delete;
};


// -------------------------------------------------------------------------------------------------
//
// Directory: RAII wrapper for (DIR *), from the C standard library.
//
// Note: instead of using this class, you may prefer pirate::listdir(),
// which is declared above and returns a vector<string>.


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

#endif  // _PIRATE_FILE_UTILS_HPP
