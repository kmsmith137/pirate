"""
The state-space model behind the fixed-lag detrender (see plans/detrend_1d_kalman.md,
section "The model", and notes/tree_dedispersion.tex,
section "Time detrending algorithm 2: Kalman filter").

The trend is a k-fold integrated random walk,

    (Delta^k f)[t] = w[t],   w ~ N(0, q) iid,      d[t] ~ N(f[t], sigma^2) at valid t,

which is Markov in the state x[t] = (f[t], (Delta f)[t], ..., (Delta^(k-1) f)[t]):

    x[t+1] = A x[t] + g w[t],   A = I + N,   g = e_(k-1),   d[t] = e_0^T x[t] + noise,

where N is the superdiagonal shift.  A is the Pascal matrix: (A^s)[j,l] = C(s, l-j).

Everything is written with sigma^2 = 1.  This costs no generality: the estimator
depends on (k, rho) only, where rho = sigma^2/q is the penalty weight of the
equivalent Whittaker-Henderson problem, so sigma never has to be known.  What that
does assume is that sigma is independent of time within a row.

Units.  tau = rho^(1/2k) is the smoothing timescale in samples and is the parameter
a caller should think in; ell = tau/sin(pi/2k) is the correlation length of N^-1,
which is what governs how much lookahead is needed; c_k = 1/(2k sin(pi/2k)) is the
flux-loss constant, so a pulse of width w loses a fraction ~ c_k w/tau of its flux
and the estimator removes c_k/tau of the variance.

No state rescaling.  One might expect to work in x~ = D x with D = diag(1, tau, ...,
tau^(k-1)), since the components of x differ in scale by powers of tau.  We do not:
D is a diagonal scaling, Cholesky is equivariant under it, and the equilibrated
condition number of J is flat in tau (measured 5.83 at k=2), so it buys nothing but
overflow headroom -- which at k=2 is not needed, the raw spread being only tau^2.
Keeping the unscaled basis makes A and A^-1 exact integer matrices.
"""

import numpy as np


class StateSpaceModel:
    """
    Model constants for one (k, tau), cast to a working dtype.

    Attributes: k, tau, rho, q, invq (= rho), ell, c_k, and the matrices A, Ainv
    (both exact integers), with g = e_(k-1) represented implicitly as index k-1.
    """

    def __init__(self, k, tau, dtype=np.float64):
        if k < 1:
            raise ValueError(f'StateSpaceModel: k={k} must be >= 1')
        if not (tau > 0):
            raise ValueError(f'StateSpaceModel: tau={tau} must be positive')

        self.k = int(k)
        self.tau = float(tau)
        self.dtype = np.dtype(dtype)

        # rho = tau^(2k) is the Whittaker penalty weight; q = 1/rho is the process
        # noise.  The recursions only ever use 1/q, so that is what we store.
        self.rho = self.tau ** (2*self.k)
        self.q = 1.0 / self.rho
        self.invq = self.dtype.type(self.rho)

        s = np.sin(np.pi / (2*self.k))
        self.c_k = 1.0 / (2*self.k*s)
        self.ell = self.tau / s

        # A = I + N has the single superdiagonal N; its inverse is the full
        # alternating sum I - N + N^2 - ..., which terminates because N is nilpotent.
        # Both are exact in any float type, the entries being +-1.
        self.A = np.eye(self.k, dtype=dtype)
        if self.k > 1:
            self.A += np.diag(np.ones(self.k-1, dtype=dtype), 1)
        self.Ainv = np.eye(self.k, dtype=dtype)
        for p in range(1, self.k):
            self.Ainv += ((-1)**p) * np.diag(np.ones(self.k-p, dtype=dtype), p)

    def __repr__(self):
        return (f'StateSpaceModel(k={self.k}, tau={self.tau:g}, ell={self.ell:.4g}, '
                f'rho={self.rho:.4g}, dtype={self.dtype.name})')

    def A_pow(self, s):
        """A^s = the Pascal matrix C(s, l-j).  Used by the tests, not by the recursions."""
        from math import comb
        out = np.zeros((self.k, self.k), dtype=self.dtype)
        for j in range(self.k):
            for l in range(j, self.k):
                out[j, l] = comb(s, l-j) if (l-j) <= s else 0
        return out


def tau_from_equivalent_W(k, W):
    """
    The tau which matches a local polynomial fit of half-width W, by equating the
    smoothing kernels at zero lag: h[0] = 9/(8W) there and c_k/tau here.  This also
    matches the flux loss and the degrees of freedom removed, so it is the right way
    to set up a like-for-like comparison of the two detrenders.

    At k=2 this is tau = 0.314 W.
    """
    c_k = 1.0 / (2*k*np.sin(np.pi/(2*k)))
    return 8.0 * c_k * W / 9.0
