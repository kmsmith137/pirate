#ifndef _PIRATE_INTERNALS_FILE_UTILS_HPP
#define _PIRATE_INTERNALS_FILE_UTILS_HPP

#include <string>
#include <vector>
#include <sys/types.h>  // size_t

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

// Cut-and-paste from CHIME FRB search.
// FIXME could use a little cleanup (e.g. stray debugging print-statement).
extern size_t disk_space_used(const std::string &dirname);


}  // namespace pirate

#endif  // _PIRATE_INTERNALS_FILE_UTILS_HPP
