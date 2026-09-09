// This file was auto-generated for pirate build (not using cmake).
// It is copied to asdf-cxx/include/asdf/config.hxx by the Makefile, and
// defines which optional asdf-cxx features are enabled.
//
// Regenerate from asdf-cxx/include/asdf/config.hxx.in after a submodule bump.
// Note: the ASDF *standard* version is no longer set here. As of asdf-cxx
// 8.0.0 it is a property of each file, chosen at write time from the
// content (see asdf/version.hxx), which is what lets float16 arrays
// declare standard 1.6.0 and stay readable by python-asdf.

#ifndef ASDF_CONFIG_HPP
#define ASDF_CONFIG_HPP

#include <string>

namespace ASDF {

// Standard version
//
// The ASDF standard versions this library supports live in
// `asdf/version.hxx`; they are a property of a file, not of the build.

// Software version

#define ASDF_CXX_NAME "asdf-cxx"
#define ASDF_CXX_AUTHOR "Erik Schnetter"
#define ASDF_CXX_HOMEPAGE "https://github.com/eschnett/asdf-cxx"

#define ASDF_CXX_VERSION_MAJOR 8
#define ASDF_CXX_VERSION_MINOR 0
#define ASDF_CXX_VERSION_PATCH 0

#define ASDF_CXX_VERSION                                                       \
  "8.0.0"

int asdf_cxx_version_major();
int asdf_cxx_version_minor();
int asdf_cxx_version_patch();

std::string asdf_cxx_version();

// Suport for some types
// FLOAT16 is enabled: older nvcc had issues, but recent versions
// (as of May 2026) handle _Float16 fine. This is what makes
// scales_offsets writable as float16 -- see AssembledFrame::write_asdf().
#define ASDF_HAVE_FLOAT16
// INT128 stays disabled: libstdc++'s std::is_integral<__int128> is false,
// which trips asdf-cxx's get_scalar_type_id<int128_t> static_assert.
#undef ASDF_HAVE_INT128

// blosc support

#if 0
#define ASDF_HAVE_BLOSC 1
#else
#undef ASDF_HAVE_BLOSC
#endif

// blosc2 support

#if 0
#define ASDF_HAVE_BLOSC2 1
#else
#undef ASDF_HAVE_BLOSC2
#endif

// bzip2 support

#if 0
#define ASDF_HAVE_BZIP2 1
#else
#undef ASDF_HAVE_BZIP2
#endif

// liblz4 support

#if 0
#define ASDF_HAVE_LIBLZ4 1
#else
#undef ASDF_HAVE_LIBLZ4
#endif

// libzstd support

#if 0
#define ASDF_HAVE_LIBZSTD 1
#else
#undef ASDF_HAVE_LIBZSTD
#endif

// OpenSSL support

#if 0
#define ASDF_HAVE_OPENSSL 1
#else
#undef ASDF_HAVE_OPENSSL
#endif

// zlib support

#if 0
#define ASDF_HAVE_ZLIB 1
#else
#undef ASDF_HAVE_ZLIB
#endif

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
