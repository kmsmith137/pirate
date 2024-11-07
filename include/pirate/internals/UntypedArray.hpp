#ifndef _PIRATE_INTERNALS_UNTYPED_ARRAY_HPP
#define _PIRATE_INTERNALS_UNTYPED_ARRAY_HPP

#include <gputils/Array.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// FIXME temporary hack that will go away, after I implement gputils::Array<void>.
struct UntypedArray
{
    gputils::Array<float> data_float32;
    gputils::Array<__half> data_float16;

    void allocate(std::initializer_list<ssize_t> shape, int aflags, bool is_float32);
    void allocate(std::initializer_list<ssize_t> shape, std::initializer_list<ssize_t> strides, int aflags, bool is_float32);
    
    void fill(const UntypedArray &x);

    UntypedArray slice(int axis, int ix) const;
    UntypedArray slice(int axis, int start, int stop) const;
    UntypedArray reshape_ref(std::initializer_list<ssize_t> shape) const;
    
    bool _is_float32(const char *name) const;
};


// Usage: Array<float> arr = uarr_get<float> (x, "x");   // where x is an UntypedArray
template<typename T>
extern gputils::Array<T> uarr_get(const UntypedArray &arr, const char *arr_name);


}  // namespace pirate

#endif  // _PIRATE_INTERNALS_UNTYPED_ARRAY_HPP
