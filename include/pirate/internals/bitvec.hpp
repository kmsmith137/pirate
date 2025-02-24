// Functions for manipulating sequences of bits as std::vector<bool>.
// Slow, intended for testing!

#ifndef _PIRATE_INTERNALS_BITVEC_HPP
#define _PIRATE_INTERNALS_BITVEC_HPP

#include <vector>
#include <cassert>
#include <cstdint>
#include <ksgpu/rand_utils.hpp>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Makes random bitvec.
inline std::vector<bool> make_bitvec(int nbits)
{
    std::vector<bool> ret(nbits);
    for (int i = 0; i < nbits; i++)
	ret[i] = (ksgpu::rand_uniform() > 0.5);
    return ret;
}

// Makes "one-hot" bitvec if 0 <= i < nbits, else zero bitvec
inline std::vector<bool> make_bitvec(int nbits, int i)
{
    std::vector<bool> ret(nbits, false);
    if ((i >= 0) && (i < nbits))
	ret[i] = true;
    return ret;
}

inline std::vector<bool> concat_bitvec(const std::vector<bool> &v1, const std::vector<bool> &v2)
{
    int n1 = v1.size();
    int n2 = v2.size();
    std::vector<bool> ret(n1+n2);
    
    for (int i = 0; i < n1; i++)
	ret[i] = v1[i];
    for (int i = 0; i < n2; i++)
	ret[n1+i] = v2[i];

    return ret;
}

inline void write_bitvec(unsigned char *p, const std::vector<bool> &v)
{
    int nbits = v.size();
    int nbytes = (v.size() + 7) / 8;
    
    for (int i = 0; i < nbytes; i++) {
	unsigned char x = 0;
	for (int j = 0; j < 8; j++) {
	    int b = 8*i+j;
	    if ((b < nbits) && v[b])
		x |= (1 << j);
	}
	p[i] = x;
    }
}

inline std::vector<bool> read_bitvec(const unsigned char *p, int nbits)
{
    assert(nbits >= 0);
    std::vector<bool> ret(nbits);
    
    for (int i = 0; i < nbits; i++) {
	unsigned char bit = p[i/8] & (1 << (i%8));
	ret[i] = (bit != 0);
    }

    return ret;
}

inline uint64_t bitvec_to_uint64(const std::vector<bool> &v, int nbits, int pos)
{
    assert(nbits >= 0);
    assert(nbits < 64);
    assert(pos >= 0);
    assert(pos+nbits <= int(v.size()));
    
    uint64_t ret = 0;
    for (int i = 0; i < nbits; i++)
	if (v[pos+i])
	    ret |= (1L << i);

    return ret;
}

inline uint64_t bitvec_to_uint64(const std::vector<bool> &v)
{
    assert(v.size() <= 64);
    return bitvec_to_uint64(v, v.size(), 0);
}


}  // namespace pirate

#endif // _PIRATE_INTERNALS_BITVEC_HPP

