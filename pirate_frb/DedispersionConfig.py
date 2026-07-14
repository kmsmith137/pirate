"""DedispersionConfig method injections (+ re-export of the pybind11 class)."""

import ksgpu
from .pirate_pybind11 import DedispersionConfig


@ksgpu.inject_methods(DedispersionConfig)
class DedispersionConfigInjections:
    # No class docstring here: DedispersionConfig's docstring lives in the pybind11
    # binding (option 1 in notes/docstrings.md); this injector adds a flexible dtype
    # setter that accepts strings, numpy/cupy dtypes, etc.

    # Save reference to C++ dtype attribute
    _cpp_dtype = DedispersionConfig.dtype

    @property
    def dtype(self):
        """Data type for dedispersion.

        Returns
        -------
        ksgpu.Dtype
            The current dtype setting.
        """
        return self._cpp_dtype

    @dtype.setter
    def dtype(self, value):
        """Set the data type for dedispersion.

        Parameters
        ----------
        value : str, numpy.dtype, cupy.dtype, or ksgpu.Dtype
            Data type specification. Examples:
            - 'float32', 'float16'
            - np.float32, cp.float16
            - ksgpu.Dtype('float32')

        Examples
        --------
        >>> config = DedispersionConfig()
        >>> config.dtype = 'float32'
        >>> config.dtype = np.float16
        >>> config.dtype = ksgpu.Dtype('float32')
        """
        self._cpp_dtype = ksgpu.Dtype(value)
