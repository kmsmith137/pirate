#include <cassert>
#include "../include/pirate/internals/UntypedArray.hpp"

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void UntypedArray::allocate(std::initializer_list<ssize_t> shape, int aflags, bool is_float32)
{
    bool have_float32 = (this->data_float32.data != nullptr);
    bool have_float16 = (this->data_float16.data != nullptr);
    
    assert(!have_float32 && !have_float16);

    if (is_float32)
	this->data_float32 = Array<float> (shape, aflags);
    else
	this->data_float16 = Array<__half> (shape, aflags);
}


void UntypedArray::allocate(std::initializer_list<ssize_t> shape, std::initializer_list<ssize_t> strides, int aflags, bool is_float32)
{
    bool have_float32 = (this->data_float32.data != nullptr);
    bool have_float16 = (this->data_float16.data != nullptr);
    
    assert(!have_float32 && !have_float16);

    if (is_float32)
	this->data_float32 = Array<float> (shape, strides, aflags);
    else
	this->data_float16 = Array<__half> (shape, strides, aflags);
}


template<> Array<float> uarr_get(const UntypedArray &uarr, const char *arr_name)
{
    bool have_float32 = (uarr.data_float32.data != nullptr);
    bool have_float16 = (uarr.data_float16.data != nullptr);

    if (!have_float32 || have_float16) {
	stringstream ss;
	ss << "uarr_get(): expected "
	   << arr_name << ".data_float32 to be nonempty, and "
	   << arr_name << ".data_float16 to be empty";
	throw runtime_error(ss.str());
    }

    return uarr.data_float32;
}

template<> Array<__half> uarr_get(const UntypedArray &uarr, const char *arr_name)
{
    bool have_float32 = (uarr.data_float32.data != nullptr);
    bool have_float16 = (uarr.data_float16.data != nullptr);

    if (have_float32 || !have_float16) {
	stringstream ss;
	ss << "uarr_get(): expected "
	   << arr_name << ".data_float32 to be empty, and "
	   << arr_name << ".data_float16 to be nonempty";
	throw runtime_error(ss.str());
    }

    return uarr.data_float16;
}


bool UntypedArray::_is_float32(const char *name) const
{
    bool have_float32 = (this->data_float32.data != nullptr);
    bool have_float16 = (this->data_float16.data != nullptr);

    if (have_float32 && !have_float16)
	return true;
    else if (!have_float32 && have_float16)
	return false;
    else if (have_float32 && have_float16)
	throw runtime_error(string(name) + " has multiple dtypes?!");
    else
	throw runtime_error(string(name) + " is empty or uninitialized");
}


UntypedArray UntypedArray::slice(int axis, int ix) const
{
    UntypedArray ret;

    if (this->_is_float32("UntypedArray::slice() argument"))
	ret.data_float32 = this->data_float32.slice(axis, ix);
    else
	ret.data_float16 = this->data_float16.slice(axis, ix);

    return ret;
}

    
UntypedArray UntypedArray::slice(int axis, int start, int stop) const
{
    UntypedArray ret;

    if (this->_is_float32("UntypedArray::slice() argument"))
	ret.data_float32 = this->data_float32.slice(axis, start, stop);
    else
	ret.data_float16 = this->data_float16.slice(axis, start, stop);

    return ret;
}


UntypedArray UntypedArray::reshape_ref(std::initializer_list<ssize_t> shape) const
{
    UntypedArray ret;

    if (this->_is_float32("UntypedArray::reshape_ref() argument"))
	ret.data_float32 = this->data_float32.reshape_ref(shape);
    else
	ret.data_float16 = this->data_float16.reshape_ref(shape);

    return ret;
}


void UntypedArray::fill(const UntypedArray &x)
{
    bool dst32 = this->_is_float32("UntypedArray::fill() destination argument");
    bool src32 = x._is_float32("UntypedArray::fill() source argument");

    if (dst32 != src32)
	throw runtime_error("UntypedArray::fill(): source and destination types do not match");
    
    if (dst32)
	this->data_float32.fill(x.data_float32);
    else
	this->data_float16.fill(x.data_float16);
}


}  // namespace pirate
