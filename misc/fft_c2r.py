import numpy as np


def fft_c2c(x, pos_phase, one_over_n=False):
    if pos_phase:
        norm = 'backward' if one_over_n else 'forward'
        return np.fft.ifft(x, norm=norm)
    else:
        norm = 'forward' if one_over_n else 'backward'
        return np.fft.fft(x, norm=norm)


def fft_r2c(x, factored=False):
    assert x.ndim == 1
    assert x.dtype == float
    
    n = len(x)
    nk = (n//2) + 1
    
    if not factored:
        y = fft_c2c(x, pos_phase=False)
        return y[:nk]

    assert (n % 2) == 0
    y = x[::2] + 1j*x[1::2]
    y = fft_c2c(y, pos_phase=False)
    z = np.zeros(nk, dtype=complex)
    
    for k in range(nk):
        kk = ((n//2) - k) if (k > 0) else 0
        yk = y[k % (n//2)]
        ykk = np.conj(y[kk])
        z[k] = 0.5 * (yk + ykk) - 0.5j * np.exp(-2j*np.pi*k/n) * (yk - ykk)

    return z


def fft_c2r(z, n, factored=False):
    nk = (n//2) + 1
    assert z.shape == (nk,)
    assert z.dtype == complex

    if not factored:
        x = np.zeros(n, dtype=complex)
        x[:nk] = z[:]
        for k in range(1,nk):
            x[-k] = np.conj(z[k])
        
        x[0] = x[0].real
        if (n % 2) == 0:
            x[n//2] = x[n//2].real
        
        x = fft_c2c(x, pos_phase=True)
        return np.copy(x.real)

    y = np.zeros(n//2, dtype=complex)

    for k in range(n//2):
        zk = z[k]
        zkk = np.conj(z[(n//2)-k])
        y[k] = (zk + zkk) + 1j * np.exp(2j*np.pi*k/n) * (zk - zkk)

    y = fft_c2c(y, pos_phase=True)

    x = np.zeros(n)
    x[::2] = y.real
    x[1::2] = y.imag
    return x
    

def test_c2c():
    """Visual test."""
    x = np.zeros(10)
    x[1] = 1

    for pos_phase in [True,False]:
        for one_over_n in [True,False]:
            y = fft_c2c(x, pos_phase=pos_phase, one_over_n=one_over_n)
            print(f'{pos_phase=} {one_over_n=} {y[:2]=}')


def test_r2c(n=16):
    x = np.random.normal(size=n)
    y1 = fft_r2c(x, factored=False)
    y2 = fft_r2c(x, factored=True)
    eps = np.max(np.abs(y1-y2))
    print(f'test_r2c({n=}): {eps=}')


def test_c2r(n=16):
    assert (n % 2) == 0
    
    nk = (n//2) + 1
    z = np.random.normal(size=nk) + 1j*np.random.normal(size=nk)
    z[0] = z[0].real
    z[-1] = z[-1].real

    x1 = fft_c2r(z, n, factored=False)
    x2 = fft_c2r(z, n, factored=True)
    eps = np.max(np.abs(x1-x2))
    # print(x1)
    # print(x2)
    print(f'test_c2r({n=}): {eps=}')
    
    
if __name__ == '__main__':
    test_c2r(16)
    test_c2r(18)
