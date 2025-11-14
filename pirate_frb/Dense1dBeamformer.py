import numpy as np

    
class Dense1dBeamformer:
    # Speed of light in weird units meters-MHz
    c = 299.79
    
    def __init__(self, freq, feed_positions, sin_za_grid):
        """
        Currently, this class is only used to make some exploratory/forecasting
        plots for the CASM beamforming design doc.
        
        Args:
        
          - freq: in MHz
          - feed_positions: 1d array of length nfeeds, units meters
          - sin_za_grid: 1d array, dimensionless in [-1,1]

        Cubic interpolation is hardcoded. (Could generalize by including
        an interpolation_width argument.)
        """

        self.freq = freq
        self.feed_positions = np.asarray(feed_positions, dtype=float)
        self.sin_za_grid = np.asarray(sin_za_grid, dtype=float)

        assert 100 <= freq <= 10000   # catch accidental use of wrong units (e.g. Hz instead of MHz)
        assert self.feed_positions.ndim == self.sin_za_grid.ndim == 1

        self.baselines = self.feed_positions[:,None] - self.feed_positions[None,:]  # shape (nfeeds,nfeeds)
        self.nfeeds = len(self.feed_positions)
        self.nbeams = len(self.sin_za_grid)
        
        assert self.nfeeds >= 2
        assert self.nbeams >= 4
        assert self.is_equally_spaced(sin_za_grid)
        assert self.sin_za_grid[1] <= -(1.0 - 1.0e-10)
        assert self.sin_za_grid[-2] >= (1.0 - 1.0e-10)

        
    @classmethod
    def is_equally_spaced(cls, x):
        assert (x.ndim == 1) and (len(x) >= 2)
        xlin = np.linspace(x[0], x[-1], len(x))
        return np.max(np.abs(x-xlin)) <= 1.0e-12 * np.max(x)
        
        
    def phase_matrix(self, sin_za):
        """Returns Hermitian (nfeeds,nfeeds) matrix."""
        return np.exp((2j*np.pi/self.c * self.freq * sin_za) * self.baselines)


    def dot(self, m1, m2):
        """Takes dot product of two Hermitian (nfeeds,nfeeds) phase matrices."""
        assert m1.shape == m2.shape == (self.nfeeds, self.nfeeds)
        return (np.vdot(m1,m2).real) / self.nfeeds**2   # np.vdot() automatically supplies complex conjugate on first factor

    
    def corrcoef(self, m1, m2):
        return self.dot(m1,m2) / (self.dot(m1,m1) * self.dot(m2,m2))**0.5
    
    
    def locate_sin_za(self, sin_za, weights=True):
        """
        Returns integer n, such that wavenumber_grid[n] <= k <= wavenumber_grid[n+1].
        If weights=True, also returns interpolation weights [w0,w1,w2,w3].
        """

        # rescale (sin_za_min, sin_za_max) -> (0,n-1)
        smin, smax = self.sin_za_grid[0], self.sin_za_grid[-1]
        x = (self.nbeams-1) * (sin_za-smin) / (smax-smin)

        eps = 1.0e-10
        assert np.all(x >= 1-eps)
        assert np.all(x <= self.nbeams-2+eps)

        n = np.array(x, dtype=int)
        n = np.maximum(n, 1)
        n = np.minimum(n, self.nbeams-3)

        if not weights:
            return n

        s0, s1 = self.sin_za_grid[n], self.sin_za_grid[n+1]
        x = (sin_za-s0) / (s1-s0)

        if True:        
            # Cubic interpolation (empirically best)
            w0 = -(x)*(x-1)*(x-2) / 6.
            w1 = (x+1)*(x-1)*(x-2) / 2.
            w2 = -(x+1)*(x)*(x-2) / 2.
            w3 = (x+1)*(x)*(x-1) / 6.
        elif False:
            # Linear interpolation
            w0 = 0
            w1 = 1-x
            w2 = x
            w3 = 0
        else:
            # Lanczos interpolation
            w0 = np.sinc(x+1)
            w1 = np.sinc(x)
            w2 = np.sinc(x-1)
            w3 = np.sinc(x-2)

        return n, np.array([w0,w1,w2,w3])


    def interpolated_phase_matrix(self, sin_za):
        n, w = self.locate_sin_za(sin_za, weights=True)
        sgrid = self.sin_za_grid[(n-1):(n+3)]
        mgrid = np.array([ self.phase_matrix(s) for s in sgrid ])    # FIXME precompute?
        m = np.sum(w[:,None,None] * mgrid, axis=0)
        return m / self.dot(m,m)**0.5
            

    def evaluate_beam_quality(self, sin_za=None):
        if sin_za is None:
            sin_za = np.random.uniform(-1,1)

        mexact = self.phase_matrix(sin_za)
        minterp = self.interpolated_phase_matrix(sin_za)
        print(f'{k=}, r_interp={self.corrcoef(mexact,minterp):.06f}')


    def prepare_beam_plot(self, sin_za0):
        """Returns (sin_za, rx, ri), where each element is a 1-d numpy array for plotting"""

        m_exact = self.phase_matrix(sin_za0)
        m_interp = self.interpolated_phase_matrix(sin_za0)

        sin_za = np.linspace(-1, 1, 1000)
        rx_vec = np.zeros(len(sin_za))
        ri_vec = np.zeros(len(sin_za))
        
        for i,sza in enumerate(sin_za):
            m = self.phase_matrix(sza)
            rx_vec[i] = self.corrcoef(m, m_exact)
            ri_vec[i] = self.corrcoef(m, m_interp)

        return sin_za, rx_vec, ri_vec


    def beam_correlation(self, sin_za):
        m_exact = self.phase_matrix(sin_za)
        m_interp = self.interpolated_phase_matrix(sin_za)
        return self.corrcoef(m_exact, m_interp)

        
    @classmethod
    def make_casm256_ew(cls, freq, nfft=24):
        assert nfft >= 4
        
        grid_ds = 2 / (nfft-3)
        grid_smax = (nfft-1) * (grid_ds/2.)
        sin_za_grid = np.linspace(-grid_smax, grid_smax, nfft)

        # Feed spacings [ 38cm, 44.5cm, 38cm, 44.5cm, 38cm ], from Liam slack
        feed_positions = np.cumsum([ 0, 0.38, 0.445, 0.38, 0.445, 0.38 ])
        return cls(freq, feed_positions, sin_za_grid)


    @classmethod
    def make_casm256_ns(cls, freq, nfft=128):
        assert nfft >= 4
        
        feed_spacing = 0.5   # 50 cm
        feed_positions = feed_spacing * np.arange(43)

        ds = cls.c / (nfft * freq * feed_spacing)
        n = int(1.0/ds)
        sin_za_grid = ds * np.arange(-n-2,n+3)
        
        return cls(freq, feed_positions, sin_za_grid)
