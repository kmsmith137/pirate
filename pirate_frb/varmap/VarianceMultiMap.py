"""class VarianceMultiMap: one VarianceMap per DedispersionTree of a DedispersionConfig."""

import numpy as np

from .VarianceMap import VarianceMap


class VarianceMultiMap:
    """One VarianceMap per DedispersionTree of a DedispersionConfig.

    ONE MAP PER TREE, FOR EVERY TREE. Production needs peak-finding weights for all of them,
    and the brute-force sweep computes them all in one pass over the input channels anyway
    (they share the dedisperser), so this is the natural unit both for the sweep and for
    file I/O.

    An instance is IMMUTABLE, like VarianceMap.

    Attributes (read-only):

    - ``config`` -- the DedispersionConfig. The SAME object every per-tree map holds a
      reference to, so the two cannot drift.
    - ``detrender`` -- the Detrender2dParams, or None. Likewise shared.
    - ``ntrees`` (int) -- number of trees, == ``len(maps)``.
    - ``maps`` (tuple) -- the per-tree VarianceMaps, with ``maps[i].itree == i``.
    - ``provenance`` (dict) -- free-form, describing how the SWEEP was run: algorithm name,
      config overrides applied, device, sweep geometry, wall times, host. Carried into the
      file verbatim. Distinct from ``VarianceMap.history``, which is a per-tree list of the
      transformations that map has been through: this one is about how the maps were MADE,
      that one about what has been done to them since. Both are stored, at their own levels
      of the file, and giving them one name would make a file's two records
      indistinguishable.

    Note that config and detrender are metadata of the whole object, and the per-tree maps
    hold the tree-specific geometry. There is no per-tree config.
    """

    def __init__(self, config, maps, *, detrender=None, provenance=None):
        """Checks that every map has the same config object, the same detrender, and
        ``itree`` equal to its position in the list."""

        maps = tuple(maps)
        if len(maps) == 0:
            raise RuntimeError('VarianceMultiMap: expected at least one per-tree map')

        for (i, m) in enumerate(maps):
            if not isinstance(m, VarianceMap):
                raise RuntimeError(f'VarianceMultiMap: maps[{i}] is a {type(m).__name__},'
                                   ' expected a VarianceMap')
            if m.itree != i:
                raise RuntimeError(f'VarianceMultiMap: maps[{i}].itree = {m.itree}, expected'
                                   f' {i} (the maps must be in tree order, one per tree)')
            if m.config is not config:
                raise RuntimeError(f'VarianceMultiMap: maps[{i}] holds a different'
                                   ' DedispersionConfig object than the multimap. The two'
                                   ' must be the same object, so that they cannot drift.')
            if m.detrender is not detrender:
                raise RuntimeError(f'VarianceMultiMap: maps[{i}] holds a different'
                                   ' Detrender2dParams than the multimap')

        # A multimap must cover EVERY tree: that is what makes it the unit production and the
        # sweep both work in. A short list is a bug in whatever assembled it, not a subset.
        ntrees = int(config.num_dedispersion_trees)
        if len(maps) != ntrees:
            raise RuntimeError(f'VarianceMultiMap: got {len(maps)} maps, but this config has'
                               f' {ntrees} dedispersion trees. A multimap holds one map per'
                               ' tree, for every tree.')

        object.__setattr__(self, 'config', config)
        object.__setattr__(self, 'detrender', detrender)
        object.__setattr__(self, 'maps', maps)
        object.__setattr__(self, 'ntrees', len(maps))
        object.__setattr__(self, 'provenance', dict(provenance) if provenance else {})


    def __setattr__(self, k, v):
        raise AttributeError(f'VarianceMultiMap is immutable (tried to set {k!r});'
                             ' use with_maps()')

    def __len__(self):
        return self.ntrees

    def __getitem__(self, itree):
        return self.maps[itree]

    def __iter__(self):
        return iter(self.maps)

    def __repr__(self):
        return f'VarianceMultiMap(ntrees={self.ntrees}, maps={list(self.maps)})'


    def with_maps(self, maps, *, provenance=None):
        """Return a new VarianceMultiMap with the same config and detrender and the given
        per-tree maps, re-validating.

        This is the general per-tree transformation::

            vmm2 = vmm.with_maps([m.coarse_grain(L[i]) for i, m in enumerate(vmm)])

        coarse_grain() and apply() are the two per-tree operations common enough to name;
        everything else goes through here. There is deliberately no per-tree ``map(fn)``
        helper that hides the loop: the loop is one line, and it is where the caller sees
        which tree gets which reference and which rank.
        """
        return type(self)(self.config, maps, detrender=self.detrender,
                          provenance=(self.provenance if provenance is None else provenance))


    def coarse_grain(self, L):
        """Per-tree coarse_grain().

        'L' may be a single int (applied to every tree) or a length-ntrees sequence, since the
        trees have different ranks r and the legal range ``R <= L <= r`` therefore differs per
        tree. A scalar L that is out of range for some tree is an error, not something to
        clamp.
        """

        Ls = [int(L)] * self.ntrees if np.isscalar(L) else [int(x) for x in L]
        if len(Ls) != self.ntrees:
            raise RuntimeError(f'VarianceMultiMap.coarse_grain: got {len(Ls)} values of L for'
                               f' {self.ntrees} trees')
        return self.with_maps([m.coarse_grain(Ls[i]) for (i, m) in enumerate(self.maps)])


    def apply(self, freq_variances):
        """Length-ntrees list of per-tree apply() results."""
        return [m.apply(freq_variances) for m in self.maps]


    def measure_admissibility(self, ref, **kwargs):
        """Per-tree measure_admissibility() against the matching tree of 'ref'.

        Unlike a distance, this DOES have a well-defined aggregate -- a multimap is admissible
        iff every tree is, so the aggregate ``max_r`` is the max over trees. Returns the list
        of per-tree AdmissibilityResults; aggregate with
        ``all(r.admissible for r in results)``.
        """
        if len(ref) != self.ntrees:
            raise RuntimeError(f'VarianceMultiMap.measure_admissibility: ref has'
                               f' {len(ref)} trees, expected {self.ntrees}')
        return [m.measure_admissibility(ref[i], **kwargs) for (i, m) in enumerate(self.maps)]


    # ---------------- I/O ----------------
    #
    # The format itself lives in varmap/asdf_io.py, which these forward to. It is imported
    # inside the methods rather than at module scope because it imports this module back.

    def write_asdf(self, filename, *, provenance=None):
        """Write this multimap to 'filename'. See varmap/asdf_io.py for the format."""
        from .asdf_io import write_multimap
        write_multimap(self, filename, provenance=provenance)


    @classmethod
    def from_asdf(cls, filename):
        """Read a variance-map file EAGERLY: ordinary in-memory arrays, no open file
        handle, no lifetime contract. This is the one to use unless you have a specific
        reason not to.

        The file must cover every tree of its config, since a multimap does; read a
        single-tree file with ``VarianceMap.from_asdf(filename, itree)``.
        """
        from .asdf_io import read_multimap
        return read_multimap(filename)


    @classmethod
    def open_asdf(cls, filename):
        """Read a variance-map file with its arrays MEMMAPPED, as a context manager::

            with VarianceMultiMap.open_asdf(path) as vmm:
                ...

        The arrays are views into an open file handle and are invalid once the with-block
        exits. Only worth it for a file too large to hold.
        """
        from .asdf_io import open_multimap
        return open_multimap(filename)


    # NOTE: there is deliberately no get_distance() here. D is defined per tree, and nothing
    # in notes/variance_map.tex defines a distance for a whole plan; any aggregate we invented
    # (an nalpha-weighted mean, say) would be a number with no agreed meaning, which is
    # exactly what the distance function's comparability contract exists to prevent. Write
    # [m.get_distance() for m in vmm] and aggregate at the call site, where the choice is
    # visible.
