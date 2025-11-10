import sys
import time
import math  # lcm()
import numpy as np


class CasmReferenceBeamformer:
    # Feed spacings are in meters.
    default_ns_feed_spacing = 0.50
    default_ew_feed_spacings = (0.38, 0.445, 0.38, 0.445, 0.38)
        
    def __init__(self,
                 frequencies,               # shape=(F,), dtype=float
                 feed_indices,              # shape=(256,2), dtype=int
                 beam_locations,            # shape=(B,2), dtype=float
                 downsampling_factor,       # scalar (integer)
                 ns_feed_spacing = None,    # meters
                 ew_feed_spacings = None):  # meters
        """
        CasmReferenceBeamformer: implements the following, in single-threaded numpy
        code which is slow and easy to read.

          (1) Exact beamforming, by summing all electric fields with beamforming phases,
              squaring, and averaging over time/polarization. This is useful as a
              reference for the details of the beamforming computation (especially
              sign conventions).
        
              Implemented as self.beamform(..., interpolate=False).

          (2) Interpolated beamforming. This is the exact computation done by the GPU
              kernel, and may also be useful as a reference.
        
              Implemented as self.beamform(..., interpolate=True).

          (3) A unit test which verifies that (1)+(2) are >99% correlated. This checks
              that interpolated beamforming is a good approximation to exact beamforming,
              and checks that all conventions (e.g. signs) are consistent between (1)+(2).

          (4) A unit test which verifies that the cuda kernel agrees with
              python interpolated beamforming (2) to machine precision. This
              tests correctness of the cuda kernel.

        For more info on the beamforming operation itself, see the beamform()
        docstring below.
                 
        Critical assumptions that would be very painful to change
        ---------------------------------------------------------
        
          - Long axis is uniformly spaced, and has length approximately 43.
        
          - Short axis has length 6, with approximate spacings [40cm, 50cm, 40cm, 50cm, 40cm].
            The details of the spacings are not so important, but the spacings being invariant
            under reversing the order is important.

        Assumptions that would be moderately painful to change
        ------------------------------------------------------

          - Beams are non-tracking.

          - Both polarizations use the same index ordering.
        
          - 256 dual-pol antennas (or fewer) in total.

          - Electric field array is int4+4, and laid out in global GPU memory with
            axes ordered (time,freq,pol,dish) from slowest to fastest, and the inner
            two axes (pol,dish) having shape (2,256).
       
        Assumptions that would be non-painful to change
        -----------------------------------------------
 
          - Phase conventions are such that the beamforming phase (before squaring
            the electric field to get intensity) is:

              exp(+ 2*pi*i*freq*c * (dish location) . (beam direction))

            where "." denotes the 3-d vector dot product, and the sign inside the
            exp(...) is "+" not "-".

        Notation
        --------
    
           T = number of time samples, must be a multiple of 'downsampling_factor'.
           F = number of frequency channels
           B = number of output beams

        Beam locations are represented by two "zenith angles" (theta_N, theta_E),
        defined formally as follows.  In a coordinate system where

           (0,0,1) = unit vector pointing toward zenith
           (1,0,0) = unit vector pointing north
           (0,1,0) = unit vector pointing west

        each beam location can be represented by a unit vector (nx,ny,nz). Then
        the zenith angles (theta_N, theta_E) are defined by

           (nx, ny) = (sin(theta_N), sin(theta_E))
        
        Constructor arguments
        ---------------------

          - frequencies: shape (F,) array containing frequencies in MHz.
        
          - feed_indices: integer-valued shape (256,2) array which encodes
            the mapping between a 1-d antenna index 0 <= i < 256, and a 2-d
            array location (j,k), where 0 <= j < 43 and 0 <= k < 6. This
            mapping i -> (j,k) is assumed to be the same for both polarizations,
            and is given by:

              j = feed_indices[i,0]
              k = feed_indices[i,1]

          - beam_locations: shape (B,2) array containing sky locations of beams,
            represented as (sin(theta_N), sin(theta_E)) where the zenith angles
            (theta_N, theta_E) are defined above. Note the sines!
        
          - downsampling_factor: integer level of downsampling between electric
            field array (after complex-valued PFB channelization) and FRB
            beamformed timestreams.

            Note that for CASM, the time sampling rate of the channelized electric
            field array is:

              dt_in = 4096 / (125 MHz) = 32.768 microseconds

            and so the time sampling rate of the FRB beamformed timestreams will be

              dt_out = (dt_in / downsampling_factor)
                     = (1.049 ms) * (32 / downsampling_factor).

          - ns_feed_spacing: spacing (in meters) of feeds along the north-south axis

          - ew_feed_spacings: length-5 array containing spacings (in meters) along
            the east-west axis. Must be flip-symmetric.
        """

        if ns_feed_spacing is None:
            ns_feed_spacing = cls.default_ns_feed_spacing
        if ew_feed_spacings is None:
            ew_feed_spacings = cls.default_ew_feed_spacings
        
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.feed_indices = np.asarray(feed_indices)
        self.beam_locations = np.asarray(beam_locations)
        self.downsampling_factor = int(downsampling_factor)
        self.ns_feed_spacing = float(ns_feed_spacing)
        self.ew_feed_spacings = np.asarray(ew_feed_spacings, dtype=float)
    
        # Beamformer has only been validated for frequencies in [400,500] MHz.
        # This assert also guards against using the wrong units (should be MHz).
        assert self.frequencies.ndim == 1
        assert np.all(self.frequencies >= 399.)
        assert np.all(self.frequencies <= 501.)

        assert self.feed_indices.shape == (256,2)
        assert self.feed_indices.dtype == int
        assert np.all(self.feed_indices >= 0)
        assert np.all(self.feed_indices[:,0] < 43)
        assert np.all(self.feed_indices[:,1] < 6)

        assert self.beam_locations.ndim == 2
        assert self.beam_locations.shape[1] == 2
        assert np.all(self.beam_locations >= -1.0)
        assert np.all(self.beam_locations <= 1.0)

        assert self.downsampling_factor > 0
        assert 0.3 <= self.ns_feed_spacing <= 0.6

        assert self.ew_feed_spacings.shape == (5,)
        assert np.all(self.ew_feed_spacings >= 0.3)
        assert np.all(self.ew_feed_spacings <= 0.6)
        assert np.max(np.abs(self.ew_feed_spacings - self.ew_feed_spacings[::-1])) < 1.0e-10  # flip-symmetric

        self.F = len(self.frequencies)      # number of frequencies
        self.B = len(self.beam_locations)   # number of beams
        self.c = 299.79                         # speed of light in weird units meters-MHz

        # self.ns_feed_locations: shape (43,), units meters
        # self.ew_feed_locations: shape (6,), units meters
        self.ns_feed_locations = np.arange(43,dtype=float) * self.ns_feed_spacing
        self.ew_feed_locations = np.cumsum([0] + list(self.ew_feed_spacings))

        # It's convenient to use a coordinate system where the feed locations are centered at the origin.
        self.ew_feed_locations -= np.mean(self.ew_feed_locations)
        self.ns_feed_locations -= np.mean(self.ns_feed_locations)

        # self.feed_locations_2d: shape (2,256), units meters, incorporates 'feed_indices'.
        self.feed_locations_2d = np.zeros((2,256), dtype=float)
        self.feed_locations_2d[0,:] = self.ns_feed_locations[self.feed_indices[:,0]]
        self.feed_locations_2d[1,:] = self.ew_feed_locations[self.feed_indices[:,1]]
        
        # self.exact_beamforming_phases: shape (F,B,256) array used for exact (non-interpolative) beamforming.
        # These are beamforming phases exp(2pi*i*freq/c * feed_location . beam_direction)
        t = np.einsum('f,id,bi->fbd', self.frequencies, self.feed_locations_2d, self.beam_locations)
        self.exact_beamforming_phases = np.exp(2j*np.pi/self.c * t)

        # Initializations related to interpolative beamforming are done in a separate function.
        self._setup_interpolation()
        

    def beamform(self, e_arr, feed_weights, interpolate=True):
        """
        Arguments
        ---------
        
          - e_arr: shape (T,F,2,256) complex-valued array, where:
        
              T = number of time samples, must be a multiple of 'downsampling_factor'.
              F = number of frequency channels
              2 = number of polarizations
              256 = number of antennas ("dishes" in the code)

          - feed_weights: shape (F,2,256) complex-valued array containing
            per-feed beamforming weights. The first step in the beamformer
            is multiplying 'e_arr' by the weights. The optimal beamforming
            weight is roughly (g^*/sigma^2), where g is the complex gain
            and sigma is the variance of the timestream (without undoing
            the gain).

          - interpolate: this boolean flag switches between exact beamforming
            (interpolate=False) and interpolative beamforming (interpolate=True).

        Outputs an array of shape (Tout,F,B) containing FRB beams.

        In the case of exact beamforming (interpolate=False), we first compute the
        beamformed electric field E'_{tfpb} from the per-feed electric field E_{tfpd}:

           E'_{tfpb} = sum_d A_{fbd} W_{fpd} E_{tfpd}   (*)

        where W_{fpd} is the 'feed_weights' array, and A_{fbd} is an array of
        beamforming phases np.exp(2j*np.pi/self.c * feed_location . beam_direction).
        Then we average |E'_{tfpb}|^2 over polarization index p, and downsample in
        time, to obtain the final output array.

        Interpolative beamforming (interpolate=True) is an excellent (>99% correlated)
        approximation to exact beamforming (*) which is faster to compute on the GPU
        for large numbers of beams. The details are described in my TeX notes.
        """

        assert e_arr.ndim == 4
        assert e_arr.shape == (len(e_arr), self.F, 2, 256)
        assert e_arr.dtype == complex

        assert feed_weights.shape == (self.F, 2, 256)
        assert feed_weights.dtype == complex

        T = len(e_arr)
        assert T % self.downsampling_factor == 0
        
        # Apply feed weights to E-array.
        e_weighted = feed_weights[None,:,:,:] * e_arr   # shape (T,F,2,256), dtype complex

        if interpolate:
            return self._beamform_interpolated(e_weighted)
        else:
            return self._beamform_exact(e_weighted)


    def _beamform_exact(self, e_weighted):
        """
        This is the function you should read, if you want a reference for the details
        of the beaforming computation (e.g. sign conventions, coordinate conventions.)
        """
        
        T, F, B = len(e_weighted), self.F, self.B
        assert e_weighted.shape == (T,F,2,256)
        assert e_weighted.dtype == complex
        
        # Apply beamforming phases to the electric field. Shape is (T,F,P,B).
        e_beamformed = np.einsum('fbd,tfpd->tfpb', self.exact_beamforming_phases, e_weighted)
            
        # Square the beamformed E-array, average over polarizations, and average over time.
        Tout, df = (T // self.downsampling_factor), self.downsampling_factor
        intensity = e_beamformed.real**2 + e_beamformed.imag**2   # shape (T,F,2,B)
        intensity = np.reshape(intensity, (Tout,df,F,2,B))        # shape (Tout,df,F,2,B)
        intensity = np.einsum('tsfpb->tfb', intensity)            # shape (Tout,F,B)
        return intensity / (2*df)

    
    ##################################   Interpolative beamforming   ###################################
    
    def _setup_interpolation(self):
        """Called by constructor, to perform initializations related to interpolative beamforming."""

        F, B = self.F, self.B
        
        # NS grid spacing: shape (F,) array.
        # NS grid locations: shape (F,128) array.
        self.ns_grid_spacing = self.c / (128 * self.ns_feed_spacing * self.frequencies)
        self.ns_grid_locations = np.einsum('f,j->fj', self.ns_grid_spacing, np.arange(128,dtype=float))

        # EW grid spacing: scalar
        # EW grid locations: shape (24,) array.
        self.ew_ngrid = 24
        self.ew_grid_spacing = (2 + 1.0e-12) / (self.ew_ngrid - 3)
        self.ew_grid_locations = self.ew_grid_spacing * (np.arange(self.ew_ngrid) - (self.ew_ngrid-1)/2)

        # EW phases: shape (F,24,6) array.
        self.ew_phases = np.exp(2j*np.pi/self.c
                                * self.frequencies[:,None,None]
                                * self.ew_grid_locations[None,:,None]
                                * self.ew_feed_locations[None,None,:])

        # Beam locations in "grid coordinates" (float, before "clamp to int")
        self.beam_grid_coords = np.zeros((F, B, 2))
        self.beam_grid_coords[:,:,0] = self.beam_locations[None,:,0] / self.ns_grid_spacing[:,None]
        self.beam_grid_coords[:,:,1] = self.beam_locations[None,:,1] / self.ew_grid_spacing + (self.ew_ngrid-1)/2

        self.interp_indices = np.array(np.floor(self.beam_grid_coords), dtype=int)   # shape (F,B,2)
        self.interp_weights = np.zeros((F, B, 2, 4))                                 # shape (F,B,2,4)

        # Cubic interpolation.
        x = self.beam_grid_coords - self.interp_indices
        self.interp_weights[:,:,:,0] = -(x)*(x-1)*(x-2) / 6.
        self.interp_weights[:,:,:,1] = (x+1)*(x-1)*(x-2) / 2.
        self.interp_weights[:,:,:,2] = -(x+1)*(x)*(x-2) / 2.
        self.interp_weights[:,:,:,3] = (x+1)*(x)*(x-1) / 6.

        # Shift by 1, make NS coordinate periodic.
        self.interp_indices -= 1
        self.interp_indices[:,:,0] %= 128

        assert np.all(self.interp_indices[:,:,0] >= 0)
        assert np.all(self.interp_indices[:,:,1] < self.ew_ngrid-3)

        
    def _beamform_interpolated(self, e_weighted):        
        T, F, B = len(e_weighted), self.F, self.B

        # Re-index 1-d antenna axis 0 <= i < 256 to 2-d antenna location (j,k), where 0 <= j < 43
        # and 0 <= k < 6. Note that we pad the j index from length-43 to length-128.
        e_gridded = np.zeros((T,F,2,128,6), dtype=complex)
        for i in range(256):
            j, k = self.feed_indices[i,:]
            e_gridded[:,:,:,j,k] = e_weighted[:,:,:,i]

        # FFT beamform in the NS direction, and dense beamform in the EW direction.
        e_gridded = np.fft.ifft(e_gridded, axis=3, norm='forward')             # shape (T,F,2,128,6)
        e_gridded = np.einsum('fba,tfpia->tfpib', self.ew_phases, e_gridded)   # shape (T,F,2,128,24)

        # Square the beamformed E-array, average over polarizations, and average over time
        Tout, df = (T // self.downsampling_factor), self.downsampling_factor
        i_gridded = e_gridded.real**2 + e_gridded.imag**2          # shape (T,F,2,128,24)
        i_gridded = np.reshape(i_gridded, (Tout,df,F,2,128,24))    # shape (Tout,df,F,2,128,24)
        i_gridded = np.einsum('tsfpij->tfij', i_gridded) / (2*df)  # shape (Tout,F,128,24)
        
        # Interpolate gridded intensities.
        # Reminder: self.interp_indices has shape (F,B,2).
        # Reminder: self.interp_weights has shape (F,B,2,4).
        
        i_out = np.zeros((Tout,F,B))
        
        for f in range(F):
            for b in range(B):
                ins, iew = self.interp_indices[f,b,:]     # scalars
                wns, wew = self.interp_weights[f,b,:,:]   # shape (4,)
                for m in range(4):
                    t = wns[m] * i_gridded[:, f, (ins+m) % 128, iew:(iew+4)]
                    i_out[:,f,b] += np.dot(t, wew)

        return i_out

    
    ##############   Unit test: exact and interpolative beamforming are >99% correlated   ##############

    
    @classmethod
    def rand_complex(cls, size):
        return np.random.uniform(-1,1,size) + 1j*np.random.uniform(-1,1,size)
    

    @classmethod
    def make_random_feed_indices(cls):
        """Returns shape (256,2) array, representing 256 randomly ordered locations in a (43,6) grid."""

        feed_indices = [ (i,j) for i in range(43) for j in range(6) ]
        feed_indices = np.random.permutation(feed_indices)
        return feed_indices[:256]


    @classmethod
    def make_regular_feed_indices(cls):
        feed_indices = np.zeros((256,2), dtype=int)
        
        for d in range(256):
            feed_indices[d,0] = (d % 43)
            feed_indices[d,1] = (d // 43)

        return feed_indices
        
        
    @classmethod
    def make_random(cls, F=4, B=5, D=8, randomize_spacings=False):
        frequencies = np.random.uniform(400., 500., size=F)
        feed_indices = cls.make_random_feed_indices()
        beam_locations = np.random.uniform(-1., 1., size=(B,2))

        ns_feed_spacing = cls.default_ns_feed_spacing
        ew_feed_spacings = cls.default_ew_feed_spacings

        if randomize_spacings:
            ns_feed_spacing = np.random.uniform(0.3, 0.6)
            ew_feed_spacings = np.random.uniform(0.3, 0.6, size=5)
            ew_feed_spacings = (ew_feed_spacings + ew_feed_spacings[::-1]) / 2.0
        
        return cls(frequencies, feed_indices, beam_locations, D, ns_feed_spacing, ew_feed_spacings)

    
    @classmethod
    def randomly_split(cls, n):
        """Randomly split 'n' into n = a*b, and return (a,b)."""

        a, b, p = 1, 1, 2
        
        while n > 1:
            if (n % p) == 0:
                flag = (np.random.uniform() > 0.5)
                a = (a*p) if flag else (a)
                b = (b) if flag else (b*p)
                n /= p
            else:
                p += 1

        return a, b
    

    @classmethod
    def test_interpolative_beamforming(cls, T=1024, F=4, B=5, D=4):
        """Run with 'python -m pirate_frb test [--casm]'."""
        
        print(f'test_casm_interpolative_beamforming: start, {T=}, {F=}, {B=}, {D=}')
        
        bf = cls.make_random(F,B,D)
        e_arr = cls.rand_complex((T,F,2,256))
        feed_weights = cls.rand_complex((F,2,256))
        
        i_exact = bf.beamform(e_arr, feed_weights, interpolate=False)
        i_interp = bf.beamform(e_arr, feed_weights, interpolate=True)
        assert i_exact.shape == i_interp.shape == (T//D, F, B)

        # Compute matrix of correlations between i_exact and i_interp.
        r = np.zeros((F,B))
        for f in range(F):
            for b in range(B):
                r[f,b] = np.corrcoef(i_exact[:,f,b], i_interp[:,f,b])[0,1]

        if np.min(r) < 0.99:
            print(r)
            raise RuntimeError(f'test_casm_interpolative_beamforming: fail, min correlation = {np.min(r)}')
            
        print(f'test_casm_interpolative_beamforming: pass, min correlation = {np.min(r)}')

    
    #############  Unit test: python and cuda implementations agree to machine precision   #############
    
    
    @classmethod
    def test_cuda_implementation(cls):
        # Hack: we put these imports (used only in this function) here, instead of
        # at the top of the file, so that this file will be importable on a machine
        # that doesn't have cuda.
        
        import cupy as cp
        from . import pirate_pybind11
        
        # Randomize (B/32), D, F, Tout, such that product is <= 10**4
        t = np.random.gamma(1.0, 1.0, size=4)
        t = np.exp(np.log(10**4) * t / np.sum(t))
        B32, D, F, Tout = [ int(x) for x in t ]
        
        B = min(32*B32, pirate_pybind11.CasmBeamformer.get_max_beams())
        Tin = Tout * D

        print(f'test_cuda_implementation({F=}, {B=}, {D=}, {Tin=}, {Tout=}): start')
        bf_py = cls.make_random(F=F, B=B, D=D, randomize_spacings=True)
        
        # FIXME(?) pybind11->cuda interface currently requires annoying dtype conversions.
        # (Currently the python interface is only used for testing, so it's not a serious issue.)
        
        bf_cuda = pirate_pybind11.CasmBeamformer(
            frequencies = np.asarray(bf_py.frequencies, dtype=np.float32),
            feed_indices = np.asarray(bf_py.feed_indices, dtype=np.int32),
            beam_locations = np.asarray(bf_py.beam_locations, dtype=np.float32),
            downsampling_factor = bf_py.downsampling_factor,
            ns_feed_spacing = bf_py.ns_feed_spacing,
            ew_feed_spacings = np.asarray(bf_py.ew_feed_spacings, dtype=np.float32)
        )

        # Use uint8, since numpy doesn't have int4+4.
        e44 = np.random.randint(0, 256, size=(Tin,F,2,256), dtype=np.uint8)

        # Convert to np.complex.
        e_re = ((e44 ^ 0x88) & 0xf) - 8.0
        e_im = (((e44 ^ 0x88) >> 4) & 0xf) - 8.0
        e = e_re + e_im*1j

        feed_weights = cls.rand_complex((F,2,256))

        # Convert complex64+64 -> float32[2]
        fw32 = np.zeros((F,2,256,2), dtype=np.float32)
        fw32[:,:,:,0] = feed_weights.real
        fw32[:,:,:,1] = feed_weights.imag

        # Call reference beamformer.
        i_py = bf_py.beamform(e, feed_weights)

        # Call cuda beamformer.
        e44 = cp.asarray(e44)
        fw32 = cp.asarray(fw32)
        i_cuda = cp.zeros((Tout,F,B), dtype=cp.float32)
        bf_cuda.launch_beamformer(e44, fw32, i_cuda)

        # Compare results from python/cuda.
        i_cuda = cp.asnumpy(i_cuda) / (2*D)    # XXX normalized by hand!!
        rms = np.mean(np.abs(i_py))
        eps = np.max(np.abs(i_py - i_cuda)) / rms
        
        print(f'test_cuda_implementation({F=}, {B=}, {D=}, {Tin=}, {Tout=}): {eps=}')
        assert eps < 1.0e-4
