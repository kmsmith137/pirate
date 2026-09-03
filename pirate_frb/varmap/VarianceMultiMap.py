"""class VarianceMultiMap: one VarianceMap per DedispersionTree of a DedispersionConfig."""

import numpy as np

from .VarianceMap import VarianceMap, make_plan


def restrict_fine_vector(y, plan, iparent, ichild):
    """Restrict tree 'iparent''s FINE (length-nalpha) vector to tree 'ichild''s rows.

    'y' is indexed by ``alpha = (d*M_parent + m)*P + p`` -- the convention documented in the
    VarianceMap module docstring -- and the return value is the child's ``(D, M_child, P)``
    array, i.e. the same convention with the multiplet axis restricted. 'plan' may have any
    completeness: only its trees are read.

    This is the row map of Proposition 1 of the appendix "Variance maps of a config's trees
    are row-restrictions of one another" in notes/variance_map.tex: the child's variance map
    is a subset of the parent's ROWS, so the child's apply() result is the corresponding
    subset of the parent's. Row selection commutes with a matrix-vector product -- if
    A_child = A_parent[rows] then A_child @ v == (A_parent @ v)[rows] -- which is what lets a
    caller restrict the small vector rather than the large matrix.

    'y' must already be FINE. On a coarse-grained map, call VarianceMap.apply_fine() rather
    than apply(): lifting before restricting is what keeps a coarse-graining rank for the
    CHILD from ever being needed (it would be L - early_trigger_level, not L).

    This is the only place that turns the subband mapping into a row map, and the only place
    that checks the two quantities m_index_mapping() deliberately does not: the coarse-DM
    count and the profile count. Both are equal for every tree of a primary-tree family --
    see the appendix's Observation (b) for D, and DedispersionTree.primary_tree (an exact copy of
    config.primary_trees[ipri]) for nprofiles -- so these are tripwires against a future change
    to the tree construction, not validation of the caller.
    """

    trees = plan.trees          # one list build; plan.trees copies on every access
    parent_tree, child_tree = trees[iparent], trees[ichild]

    D_p = 1 << (int(parent_tree.tree_rank) - int(parent_tree.frequency_subbands.pf_rank))
    D_c = 1 << (int(child_tree.tree_rank) - int(child_tree.frequency_subbands.pf_rank))
    P_p, P_c = int(parent_tree.nprofiles), int(child_tree.nprofiles)

    if D_c != D_p:
        raise RuntimeError(f'restrict_fine_vector: coarse-DM count is {D_c} in the child tree'
                           f' and {D_p} in the parent. An early trigger removes subband levels'
                           ' and tree rank in equal measure, so these must agree.')
    if P_c != P_p:
        raise RuntimeError(f'restrict_fine_vector: nprofiles is {P_c} in the child tree and'
                           f' {P_p} in the parent. Both come from the same'
                           ' config.primary_trees entry, so these must agree.')

    M_p = int(parent_tree.frequency_subbands.M)
    y = np.asarray(y)
    if y.shape != (D_p * M_p * P_p,):
        raise RuntimeError(f'restrict_fine_vector: expected a length-{D_p*M_p*P_p} FINE vector'
                           f' for the parent tree, got shape {y.shape}')

    m_map = plan.m_index_mapping(iparent, ichild)
    return y.reshape(D_p, M_p, P_p)[:, m_map, :]


def expand_fine_vectors(plan, per_primary):
    """Length-ntrees list of (D, M, P) arrays from one FINE vector per PRIMARY tree.

    'per_primary' is one flat, length-nalpha array per primary tree, in gamma order; the
    result is indexed by ITREE, and each entry is in its own tree's geometry. Every entry is
    fresh: a parent's is a reshaped view of what the caller passed in, and a child's is a
    copy.

    THE CHILD TREES COME FROM PROPOSITION 1 of the appendix "Variance maps of a config's trees
    are row-restrictions of one another" in notes/variance_map.tex: an early-trigger tree's
    map is a subset of its parent's ROWS, and row selection commutes with A @ v, so its result
    is the corresponding subset of the parent's. See restrict_fine_vector(), which is the row
    map. Nothing here assumes anything about the upstream chain, so unlike Proposition 2 this
    holds with a detrender too.

    THE PARENT IS THE LAST TREE OF ITS FAMILY, not the first: early_trigger_level DESCENDS
    within a family, so iparent is NOT itree - e. That trap is the reason this is one function
    rather than a loop each caller writes.

    'plan' may have any completeness -- only its config and trees are read.
    """

    config = plan.config
    npri = int(config.num_primary_trees)
    per_primary = list(per_primary)
    if len(per_primary) != npri:
        raise RuntimeError(f'expand_fine_vectors: got {len(per_primary)} vectors for {npri}'
                           ' primary trees. One per PRIMARY tree is required, in gamma order'
                           ' -- a short list would leave holes in the result.')

    ntrees = int(plan.ntrees)
    trees = plan.trees          # one list build; plan.trees copies on every access
    out = [None] * ntrees

    for gamma in range(npri):
        # See the appendix's fact (a): every primary tree HAS an e == 0 tree, so this lookup
        # cannot fail.
        iparent = int(plan.dedispersion_tree_index(gamma, 0))
        parent = trees[iparent]
        y = np.asarray(per_primary[gamma])

        D = 1 << (int(parent.tree_rank) - int(parent.frequency_subbands.pf_rank))
        M, P = int(parent.frequency_subbands.M), int(parent.nprofiles)
        if y.shape != (D * M * P,):
            raise RuntimeError(f'expand_fine_vectors: primary tree {gamma} needs a flat'
                               f' length-{D*M*P} FINE vector, got shape {y.shape}')
        out[iparent] = y.reshape(D, M, P)

        net = int(config.primary_trees[gamma].num_early_triggers)
        for e in range(1, net + 1):
            ichild = int(plan.dedispersion_tree_index(gamma, e))
            out[ichild] = restrict_fine_vector(y, plan, iparent, ichild)

    assert all(x is not None for x in out)
    return out


class VarianceMultiMap:
    """One VarianceMap per PRIMARY tree of a DedispersionConfig.

    ONE MAP PER PRIMARY TREE, NOT PER TREE. The variance map of tree (gamma, e) is a subset of
    the ROWS of the map of tree (gamma, 0) -- Proposition 1 of the appendix "Variance maps of
    a config's trees are row-restrictions of one another" in notes/variance_map.tex -- so
    storing an early-trigger tree's map would be storing a copy of rows it already has. What
    production needs from a child tree is not its matrix but its apply() result, and that is
    the corresponding subset of the parent's, which apply_fine() below computes directly.

    So: there is NO accessor returning an early-triggered tree's VarianceMap, and there should
    not be. At CHORD scale materializing one would form a 12.0 GiB matrix to collapse it to a
    0.44 MiB vector, and offering it would make "a VarianceMap is a primary tree's map"
    unenforceable. The cross-tree knowledge lives HERE, which is why VarianceMap has no notion
    of a child tree.

    An instance is IMMUTABLE, like VarianceMap.

    Attributes (read-only):

    - ``config`` -- the DedispersionConfig. The SAME object every stored map holds a
      reference to, so the two cannot drift.
    - ``plan`` -- the "minimal" DedispersionPlan the geometry comes from. Likewise
      shared with the stored maps. See VarianceMap.make_plan().
    - ``detrender`` -- the Detrender2dParams, or None. Likewise shared.
    - ``num_primary_trees`` (int) -- number of stored maps.
    - ``ntrees`` (int) -- number of DEDISPERSION trees, i.e. the length of apply_fine()'s
      result.
      Generally larger than num_primary_trees.
    - ``maps`` (tuple) -- the stored VarianceMaps, in gamma order. Prefer primary_map(gamma).
    - ``provenance`` (dict) -- free-form, describing how the SWEEP was run: algorithm name,
      config overrides applied, device, sweep geometry, wall times, host. Carried into the
      file verbatim. Distinct from ``VarianceMap.history``, which is a per-tree list of the
      transformations that map has been through: this one is about how the maps were MADE,
      that one about what has been done to them since. Both are stored, at their own levels
      of the file, and giving them one name would make a file's two records
      indistinguishable.

    Note that config and detrender are metadata of the whole object, and the stored maps hold
    the tree-specific geometry. There is no per-tree config.
    """

    def __init__(self, config, maps, *, detrender=None, provenance=None, plan=None):
        """'maps' is one VarianceMap per primary tree, in gamma order.

        Checks that every map has the same config object, the same detrender, and an ``itree``
        naming the ``early_trigger_level == 0`` tree of its own gamma.

        'plan' defaults to make_plan(config); pass one to reuse it (the file reader does).
        """

        maps = tuple(maps)
        if len(maps) == 0:
            raise RuntimeError('VarianceMultiMap: expected at least one per-primary-tree map')

        npri = int(config.num_primary_trees)
        plan = make_plan(config) if (plan is None) else plan

        # A multimap must cover EVERY primary tree: that is what makes it the unit production
        # and the sweep both work in. A short list is a bug in whatever assembled it, not a
        # subset.
        if len(maps) != npri:
            raise RuntimeError(f'VarianceMultiMap: got {len(maps)} maps, but this config has'
                               f' {npri} primary trees. A multimap holds one map per PRIMARY'
                               ' tree, for every primary tree. (Early-trigger trees are'
                               ' derived from their parent, not stored -- see the class'
                               ' docstring.)')

        for (gamma, m) in enumerate(maps):
            if not isinstance(m, VarianceMap):
                raise RuntimeError(f'VarianceMultiMap: maps[{gamma}] is a {type(m).__name__},'
                                   ' expected a VarianceMap')

            # No theorem covers this: it is a property of whatever assembled the list, not of
            # the config, and restricting FROM the wrong map would be silent.
            want = int(plan.dedispersion_tree_index(gamma, 0))
            if m.itree != want:
                raise RuntimeError(f'VarianceMultiMap: maps[{gamma}].itree = {m.itree}, but'
                                   f' primary tree {gamma} has its early_trigger_level == 0'
                                   f' tree at itree {want}. The maps must be in gamma order,'
                                   ' and each must be the map of its own parent tree. (Note'
                                   ' itree is NOT gamma, and the parent is the LAST tree of'
                                   ' its family, not the first.)')

            if m.config is not config:
                raise RuntimeError(f'VarianceMultiMap: maps[{gamma}] holds a different'
                                   ' DedispersionConfig object than the multimap. The two'
                                   ' must be the same object, so that they cannot drift.')
            if m.detrender is not detrender:
                raise RuntimeError(f'VarianceMultiMap: maps[{gamma}] holds a different'
                                   ' Detrender2dParams than the multimap')

        ntrees = int(plan.ntrees)

        object.__setattr__(self, 'config', config)
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'detrender', detrender)
        object.__setattr__(self, 'maps', maps)
        object.__setattr__(self, 'num_primary_trees', npri)
        object.__setattr__(self, 'ntrees', ntrees)
        object.__setattr__(self, 'provenance', dict(provenance) if provenance else {})


    def __setattr__(self, k, v):
        raise AttributeError(f'VarianceMultiMap is immutable (tried to set {k!r});'
                             ' use with_maps()')

    def __repr__(self):
        return (f'VarianceMultiMap(num_primary_trees={self.num_primary_trees},'
                f' ntrees={self.ntrees}, maps={list(self.maps)})')

    # NOTE: no __len__/__getitem__/__iter__, deliberately. They used to be indexed by itree,
    # and a silent change of meaning to gamma would have left every existing call site
    # working and wrong. Say which you mean: primary_map(gamma), or apply_fine() for
    # per-tree results.

    def primary_map(self, gamma):
        """The stored VarianceMap of primary tree 'gamma'.

        Its ``itree`` is ``plan.dedispersion_tree_index(gamma, 0)``, which is NOT gamma.
        """
        npri = self.num_primary_trees
        if not (0 <= gamma < npri):
            raise RuntimeError(f'VarianceMultiMap.primary_map: gamma={gamma} is out of range'
                               f' [0, {npri})')
        return self.maps[gamma]


    def with_maps(self, maps, *, provenance=None):
        """Return a new VarianceMultiMap with the same config and detrender and the given
        per-PRIMARY-tree maps, re-validating.

        This is the general per-primary-tree transformation::

            vmm2 = vmm.with_maps([m.coarse_grain(L[g]) for g, m in enumerate(vmm.maps)])

        coarse_grain() and apply_fine() are the two operations common enough to name;
        everything else goes through here. There is deliberately no ``map(fn)`` helper that hides the
        loop: the loop is one line, and it is where the caller sees which map gets which
        reference and which rank.
        """
        return type(self)(self.config, maps, detrender=self.detrender, plan=self.plan,
                          provenance=(self.provenance if provenance is None else provenance))


    def coarse_grain(self, L):
        """Per-PRIMARY-tree coarse_grain().

        'L' may be a single int (applied to every stored map) or a length-num_primary_trees
        sequence, since the primary trees have different ranks r and the legal range
        ``R <= L <= r`` therefore differs. A scalar L that is out of range for some map is an
        error, not something to clamp.

        Child trees never carry a coarse-graining rank of their own: apply_fine() lifts
        before it restricts, so a child's L (which would be ``L - early_trigger_level``, not
        ``L``) is never computed anywhere.
        """

        npri = self.num_primary_trees
        Ls = [int(L)] * npri if np.isscalar(L) else [int(x) for x in L]
        if len(Ls) != npri:
            raise RuntimeError(f'VarianceMultiMap.coarse_grain: got {len(Ls)} values of L for'
                               f' {npri} primary trees')
        return self.with_maps([m.coarse_grain(Ls[g]) for (g, m) in enumerate(self.maps)])


    def apply_fine(self, freq_variances):
        """Length-ntrees list of per-tree VarianceMap.apply_fine() results, indexed by ITREE.

        Each entry has shape ``(2^(r-R), M, P)`` for its own tree -- the (coarse DM, multiplet,
        profile) form, since a child's result differs from its parent's only along the
        multiplet axis and returning it flat would hide that.

        NAMED apply_fine(), NOT apply(), because every entry is FINE even when the stored maps
        are coarse-grained: a coarse result cannot be restricted to a child's rows without a
        coarse-graining rank for the child, which this deliberately never computes (see
        coarse_grain()). Per-map coarse results are available as
        ``[m.apply(v) for m in vmm.maps]``, which says plainly that they are per PRIMARY tree
        and of length nbeta.

        This is the one method that speaks in dedispersion trees rather than primary trees,
        and it is the only thing production needs per tree.

        A child tree's entry is computed from its parent's WITHOUT ever forming a child
        matrix: apply the parent's map, lift the result to fine granularity, and select the
        child's rows. At CHORD the (3,3) child's dense matrix is 12.0 GiB while this result is
        0.44 MiB. That last step is expand_fine_vectors(), which this shares with
        compute_detrender_free_varfine() -- see it for Proposition 1 and for the
        parent-is-the-last-tree trap.

        The lift comes BEFORE the restriction on purpose -- see coarse_grain().
        """

        return expand_fine_vectors(self.plan,
                                   [m.apply_fine(freq_variances) for m in self.maps])


    def measure_admissibility(self, ref, **kwargs):
        """Per-PRIMARY-tree measure_admissibility() against the matching map of 'ref'.

        CERTIFYING THE PRIMARY TREES CERTIFIES EVERY TREE, so this is not the weaker result it
        might look like. Admissibility is pointwise -- "no element underestimates" -- and a
        child's matrix is a subset of its parent's ROWS (Proposition 1), so a parent which
        never underestimates has no subset that does. Four certifications cover chord_sb2_et's
        ten trees.

        Unlike a distance, this DOES have a well-defined aggregate -- a multimap is admissible
        iff every map is, so the aggregate ``max_r`` is the max over them. Returns the list of
        per-primary-tree AdmissibilityResults; aggregate with
        ``all(r.admissible for r in results)``.

        ``max_diff`` does NOT aggregate that way: each map normalizes by its own ``max|ref|``,
        so the max over trees is a max of per-tree relative errors rather than a global one.
        Report it per tree.
        """
        npri = self.num_primary_trees
        ref_maps = ref.maps if isinstance(ref, VarianceMultiMap) else tuple(ref)
        if len(ref_maps) != npri:
            raise RuntimeError(f'VarianceMultiMap.measure_admissibility: ref has'
                               f' {len(ref_maps)} maps, expected {npri}')
        return [m.measure_admissibility(ref_maps[g], **kwargs)
                for (g, m) in enumerate(self.maps)]


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

        The file must cover every PRIMARY tree of its config, since a multimap does; read a
        single-map file with ``VarianceMap.from_asdf(filename, gamma)``.
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
    # [m.get_distance() for m in vmm.maps] and aggregate at the call site, where the choice is
    # visible.
