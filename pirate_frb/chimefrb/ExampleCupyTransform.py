"""ExampleCupyTransform: a worked example of a chimefrb transform written in cupy, on top of
CupyTransformBase. It exists to show the shape of a transform -- the constructor, the
computation, the two yaml methods -- and is not meant for use: GpuIntensityClipper is the
real thing (see the class docstring).
"""

from .CupyTransformBase import CupyTransformBase
from .transform_io import check_yaml_keys


class ExampleCupyTransform(CupyTransformBase):
    """Masks every sample more than ``sigma`` standard deviations from its row's mean.

    Per (beam, channel) row, over time: the weighted mean and variance of the intensity, then
    the weight of every sample with ``|intensity - mean| > sigma * sqrt(variance)`` is set to
    zero. The intensity is not modified. A row with no weight, or with zero variance, has
    nothing masked.

    This is :class:`GpuIntensityClipper` with ``axis='time'``, ``Df = Dt = 1``, ``niter = 1``
    and ``two_pass = True``, except for two details of the real thing: the clipper also
    rejects a variance below the ``wrms_eps_2`` cutoff, masking the whole row, and it uses
    ``>=`` where this uses ``>``. It is here as the example to copy when writing a transform
    in cupy, not as something to run.

    Attributes (read-only by convention): ``nbeams``, ``nfreq``, ``ntime``, ``scratch_nelts``
    (always 0), and ``sigma``.
    """

    def __init__(self, nbeams, nfreq, ntime, sigma=3.0):
        # scratch_nelts stays 0: cupy allocates the few temporaries below from its pool.
        super().__init__(nbeams, nfreq, ntime)

        if not (sigma > 0):
            raise ValueError(f'ExampleCupyTransform: expected sigma > 0, got {sigma!r}')
        self.sigma = float(sigma)

    def launch_checked(self, intensity, weights, scratch):
        import cupy as cp

        wsum = weights.sum(axis=2, keepdims=True)                          # (nbeams, nfreq, 1)
        den = cp.where(wsum > 0, wsum, 1.0)                                # no divide by zero
        wi = cp.where(weights != 0, weights * intensity, 0.0)              # a NaN at zero weight stays out
        mean = wi.sum(axis=2, keepdims=True) / den
        d = intensity - mean
        var = cp.where(weights != 0, weights * d * d, 0.0).sum(axis=2, keepdims=True) / den
        weights[cp.abs(d) > self.sigma * cp.sqrt(var)] = 0                 # var == 0 masks nothing

    def to_yaml_dict(self):
        return {'class_name': 'ExampleCupyTransform', 'sigma': self.sigma}

    @classmethod
    def from_yaml_dict(cls, d, nbeams, nfreq, ntime):
        check_yaml_keys(d, 'ExampleCupyTransform', ['sigma'])
        return cls(nbeams, nfreq, ntime, sigma=d['sigma'])
