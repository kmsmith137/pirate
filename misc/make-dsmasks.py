#!/usr/bin/env python
#
# Masks for AVX256 downsampling kernels (see include/ch_frb_server/avx256/downsample.hpp)


def dsmask(nbits, mult):
    y = (1 << nbits) - 1
    ret = 0
    for _ in range(mult):
        ret = (ret << (2*nbits)) | y
    return ret


for (nbits,mult) in [ (5,4), (10,2), (20,1), (6,4), (12,2), (24,1), (7,4), (14,2), (28,1) ]:
    d = dsmask(nbits, mult)
    print(f'\tdsmask_{nbits}_{mult} = _mm256_set1_epi64x(0x{d:x}L);')
