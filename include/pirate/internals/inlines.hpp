#ifndef _PIRATE_INTERNALS_INLINES_HPP
#define _PIRATE_INTERNALS_INLINES_HPP

#include <ksgpu/string_utils.hpp>  // ksgpu::nbytes_to_str()

#include <cmath>
#include <cassert>
#include <string>
#include <iostream>
#include <cuda_fp16.h>  // __half, __half2

namespace pirate {
#if 0
}  // editor auto-indent
#endif


inline bool is_power_of_two(long n)
{
    return (n > 0) && !(n & (n-1));
}

inline long pow2(int n)
{
    assert((n >= 0) && (n <= 32));
    return 1L << n;
}

inline long align_up(long n, long nalign)
{
    assert(n >= 0);
    assert(nalign > 0);
    assert(is_power_of_two(nalign));
    return (n + nalign - 1) & ~(nalign - 1);
}

inline long round_up_to_power_of_two(long n)
{
    if (n <= 1)
	return 1;
    double x = log2(n - 0.5);
    return 1L << (int(x) + 1);
}

inline long round_down_to_power_of_two(long n)
{
    assert(n >= 1);
    double x = log2(n + 0.5);
    return 1L << int(x);
}

inline long xdiv(long m, long n)
{
    assert(m >= 0);
    assert(n > 0);
    assert((m % n) == 0);
    return m/n;
}

inline bool is_empty_string(const std::string &s)
{
    return s.size() == 0;
}

template<typename T>
inline bool is_sorted(const std::vector<T> &v, int min_length=0, bool allow_duplicates=false)
{
    assert((int)v.size() >= min_length);

    for (unsigned int i = 1; i < v.size(); i++) {
	if (v[i-1] > v[i])
	    return false;
	if ((v[i-1] == v[i]) && !allow_duplicates)
	    return false;
    }

    return true;
}


inline bool is_aligned(const void *ptr, int nalign, bool allow_null=false)
{
    if (!allow_null && !ptr)
	return false;
    
    assert(nalign > 0);
    return (((unsigned long)ptr) % nalign) == 0;
}
		       

// -------------------------------------------------------------------------------------------------
//
// FIXME make new source file (io_utils.hpp?) for these functions?


struct Indent
{
    const int n;
    Indent(int n_) : n(n_) { }
};

inline std::ostream &operator<<(std::ostream &os, const Indent &ind)
{
    for (int i = 0; i < ind.n; i++)
	os << " ";
    return os;
}


template<typename T>
inline void print_kv(const char *key, T val, std::ostream &os, int indent, const char *units = nullptr)
{
    os << Indent(indent) << key << " = " << val;

    if (units != nullptr)
	os << " " << units;
    
    os << std::endl;
}

// Specialization print_kv<bool>()
template<>
inline void print_kv(const char *key, bool val, std::ostream &os, int indent, const char *units)
{
    print_kv(key, (val ? "true" : "false"), os, indent, units);
}


inline void print_kv_nbytes(const char *key, long nbytes, std::ostream &os, int indent)
{
    os << Indent(indent) << key;
    
    if (nbytes >= 1024)
	os << " = " << nbytes;

    os << " = " << ksgpu::nbytes_to_str(nbytes) << std::endl;
}


// -------------------------------------------------------------------------------------------------
//
// simd32_type<T>::type
//   -> float if T=float (or any other 32-bit type)
//   -> __half2 if T=__half

template<typename T>
struct simd32_type
{
    static_assert(sizeof(T) == 4);
    using type = T;
};


template<>
struct simd32_type<__half>
{
    using type = __half2;
};


}  // namespace pirate

#endif // _PIRATE_INTERNALS_INLINES_HPP

