// This file was auto-generated for pirate build (not using cmake).
// It defines which optional features are enabled.

#ifndef ASDF_CONFIG_HPP
#define ASDF_CONFIG_HPP

#include <string>

namespace ASDF {

// Standard version

#define ASDF_STANDARD_VERSION_MAJOR 1
#define ASDF_STANDARD_VERSION_MINOR 2
#define ASDF_STANDARD_VERSION_PATCH 0

#define ASDF_STANDARD_VERSION "1.2.0"

int asdf_standard_version_major();
int asdf_standard_version_minor();
int asdf_standard_version_patch();

std::string asdf_standard_version();

// Software version

#define ASDF_CXX_NAME "asdf-cxx"
#define ASDF_CXX_AUTHOR "Erik Schnetter"
#define ASDF_CXX_HOMEPAGE "https://github.com/eschnett/asdf-cxx"

#define ASDF_CXX_VERSION_MAJOR 8
#define ASDF_CXX_VERSION_MINOR 0
#define ASDF_CXX_VERSION_PATCH 0

#define ASDF_CXX_VERSION "8.0.0"

int asdf_cxx_version_major();
int asdf_cxx_version_minor();
int asdf_cxx_version_patch();

std::string asdf_cxx_version();

// Support for some types (check at compile time)
// These are disabled for pirate builds because nvcc doesn't fully support them.
// _Float16 and __int128 are available on modern GCC but nvcc has issues with them.
#undef ASDF_HAVE_FLOAT16
#undef ASDF_HAVE_INT128

// blosc support - disabled
#undef ASDF_HAVE_BLOSC

// blosc2 support - disabled
#undef ASDF_HAVE_BLOSC2

// bzip2 support - disabled (to minimize dependencies)
#undef ASDF_HAVE_BZIP2

// liblz4 support - disabled
#undef ASDF_HAVE_LIBLZ4

// libzstd support - disabled
#undef ASDF_HAVE_LIBZSTD

// OpenSSL support - disabled (to minimize dependencies)
#undef ASDF_HAVE_OPENSSL

// zlib support - disabled (to minimize dependencies)
#undef ASDF_HAVE_ZLIB

// Consistency check

void check_version(const char *header_version, bool have_float16,
                   bool have_int128);

#ifdef ASDF_HAVE_FLOAT16
#define ASDF_FLOAT16_SUPPORTED 1
#else
#define ASDF_FLOAT16_SUPPORTED 0
#endif

#ifdef ASDF_HAVE_INT128
#define ASDF_INT128_SUPPORTED 1
#else
#define ASDF_INT128_SUPPORTED 0
#endif

#define ASDF_CHECK_VERSION()                                                   \
  (::ASDF::check_version(ASDF_CXX_VERSION, ASDF_FLOAT16_SUPPORTED,             \
                         ASDF_INT128_SUPPORTED))

} // namespace ASDF

#define ASDF_CONFIG_HPP_DONE
#endif // #ifndef ASDF_CONFIG_HPP
#ifndef ASDF_CONFIG_HPP_DONE
#error "Cyclic include depencency"
#endif
