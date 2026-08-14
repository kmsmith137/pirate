"""Reading and writing variance maps (the dense matrix A) as ASDF files.

The file holds one float64 array per DedispersionTree, plus enough metadata to interpret them
without rebuilding a DedispersionPlan -- which matters because DedispersionPlan construction
calls cudaHostAlloc, so a plan cannot be built on a machine without a working GPU.

The DedispersionConfig and Detrender2dParams that produced the map are stored as yaml strings
(the same trick DedispersionPlan.make_incomplete_plan_from_yaml uses to move a plan between
processes), and are re-parsed on demand by VarianceMapFile.config / .detrender.

Per-tree metadata includes the tree's RESTRICTED subband_counts, which is what makes the file
self-describing: the config's toplevel subband vector is not the same thing, and recovering one
from the other means reapplying the restriction rule (early trigger, then pf_rank). Storing it
is what lets a reader decompose the multiplet index m into (subband, fine DM) with no plan.

Arrays are stored uncompressed. That is not an oversight: uncompressed blocks are what let
asdf memmap them, so a map far larger than RAM can be opened and sliced. Compressing would
trade that away for a poor ratio on dense float64 data.
"""

import time

import numpy as np


class VarianceMapTree:
    """Per-tree metadata from a variance-map file: enough to interpret A[itree] without a
    DedispersionPlan (and so without a GPU). Read-only.

    'subband_counts' is the tree's RESTRICTED subband vector (length R+1), not the toplevel
    one from the config: it is what decomposes the multiplet index m into (subband, fine DM),
    which is the one thing the other fields do not determine. Without it a reader has to
    re-derive it from the config yaml, which needs the restriction rule and is easy to get
    subtly wrong.
    """

    _INT_FIELDS = ('itree', 'r', 'R', 'M', 'P', 'ndm_out', 'gamma', 'early_trigger_level',
                   'nfreq')
    _FIELDS = _INT_FIELDS + ('subband_counts',)

    def __init__(self, d):
        for f in self._INT_FIELDS:
            object.__setattr__(self, f, int(d[f]))

        if 'subband_counts' not in d:
            raise RuntimeError(
                "variance-map file predates the 'subband_counts' field. Regenerate it with"
                " 'pirate_frb variance_map', or migrate it (see"
                " plans/varmap_subband_counts_migration.md).")

        sbc = tuple(int(c) for c in d['subband_counts'])
        if len(sbc) != self.R + 1:
            raise RuntimeError(f'variance-map file: tree {self.itree} has subband_counts'
                               f' {sbc} of length {len(sbc)}, expected R+1 = {self.R+1}')
        object.__setattr__(self, 'subband_counts', sbc)

    def __setattr__(self, k, v):
        raise AttributeError(f'VarianceMapTree is read-only (tried to set {k!r})')

    @property
    def N(self):
        """Number of frequency subbands (M counts multiplets, i.e. subband x fine DM)."""
        return sum(self.subband_counts)

    def to_dict(self):
        return {f: getattr(self, f) for f in self._FIELDS}

    def __repr__(self):
        inner = ', '.join(f'{f}={getattr(self, f)}' for f in self._FIELDS)
        return f'VarianceMapTree({inner})'


class VarianceMapFile:
    """A variance map read from disk; see read_variance_map(), which is how you get one.

    LAZY READS KEEP THE FILE OPEN. With lazy=True (the default) the arrays are memmapped
    views into an open file handle, so the object must outlive any use of them -- use it as a
    context manager, or call close() when done. Touching an array after close() raises. With
    lazy=False the arrays are ordinary in-memory numpy arrays and there is no handle to
    manage.
    """

    def __init__(self, af, tree, lazy):
        self._af = af          # the open AsdfFile, or None if not lazy
        self._tree = tree
        self._lazy = lazy
        self._config = None    # parsed on demand; see config/detrender below
        self._detrender = None

        self.created = tree.get('created')
        self.overrides = list(tree.get('overrides', []))
        self.device = tree.get('device')
        self.ntime = int(tree['ntime'])
        self.nphases = int(tree['nphases'])
        self.ndata_chunks = int(tree['ndata_chunks'])
        self.guard_chunk = bool(tree['guard_chunk'])
        self.nbeams = int(tree['nbeams'])
        self.trees = [VarianceMapTree(d) for d in tree['trees']]

    # ---- arrays ----

    @property
    def A(self):
        """List of (2^(r-R), M, P, nfreq) float64 arrays, one per tree."""
        self._check_open()
        return self._tree['A']

    def matrix(self, itree):
        """A[itree] reshaped to (2^(r-R) * M * P, nfreq): the matrix A of the tex notes.

        With a lazy read this is a view where possible, but numpy will copy if the memmap
        cannot be reshaped without one -- so for a large map, slice before reshaping.
        """
        self._check_open()
        a = self._tree['A'][itree]
        return a.reshape(-1, a.shape[-1])

    # ---- the inputs, re-parsed on demand ----

    @property
    def config(self):
        """The DedispersionConfig that produced this map, as run (i.e. after any CLI
        overrides; see the 'overrides' member for what changed)."""
        if self._config is None:
            from ..pirate_pybind11 import DedispersionConfig
            self._config = DedispersionConfig.from_yaml_string(self._tree['config_yaml'])
        return self._config

    @property
    def detrender(self):
        """The Detrender2dParams that produced this map, or None if no detrender was used."""
        y = self._tree.get('detrender_yaml')
        if y is None:
            return None
        if self._detrender is None:
            from ..pirate_pybind11 import Detrender2dParams
            self._detrender = Detrender2dParams.from_yaml_string(y)
        return self._detrender

    # ---- lifetime ----

    def close(self):
        if self._af is not None:
            self._af.close()
            self._af = None
        self._tree = None

    def _check_open(self):
        if self._tree is None:
            raise RuntimeError('VarianceMapFile has been closed; its memmapped arrays are no'
                               ' longer valid. Read with lazy=False, or keep the object alive'
                               ' (e.g. "with read_variance_map(...) as vm: ...").')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __repr__(self):
        shapes = ', '.join(str(t.ndm_out) + 'x' + str(t.M) + 'x' + str(t.P) + 'x' + str(t.nfreq)
                           for t in self.trees)
        return (f'VarianceMapFile(ntrees={len(self.trees)}, A=[{shapes}], '
                f'device={self.device!r}, {"lazy" if self._lazy else "in-memory"})')


def write_variance_map(filename, A, plan, detrender=None, device=None, overrides=None,
                       ntime=0, nphases=1, ndata_chunks=0, guard_chunk=True, nbeams=1):
    """Writes a variance map to an ASDF file.

    Args:
      filename: output path.
      A: list of (2^(r-R), M, P, nfreq) arrays, one per tree -- i.e. what
        BruteForceVarianceMap.run() returns. Cast to float64 if it is not already.
      plan: the DedispersionPlan the sweep used. The per-tree metadata and the embedded
        config yaml both come from it, which is why this takes a plan rather than a pile of
        scalars.
      detrender: the Detrender2dParams used, or None.
      device: 'cpu' or 'gpu', for the record.
      overrides: list of human-readable strings describing how the config was modified
        relative to the user's input file (see the variance_map CLI).
      ntime, nphases, ndata_chunks, guard_chunk, nbeams: sweep geometry, recorded so a
        reader can interpret A without rebuilding the plan.
    """

    import asdf

    if len(A) != int(plan.ntrees):
        raise RuntimeError(f'write_variance_map: got {len(A)} arrays but plan.ntrees ='
                           f' {int(plan.ntrees)}')

    trees, arrays = [], []
    for itree in range(int(plan.ntrees)):
        t = plan.trees[itree]
        r, R = int(t.total_rank()), int(t.frequency_subbands.pf_rank)
        info = dict(itree=itree, r=r, R=R, M=int(t.frequency_subbands.M),
                    P=int(t.nprofiles), ndm_out=int(t.ndm_out),
                    gamma=int(t.primary_tree_index),
                    early_trigger_level=int(t.early_trigger_level), nfreq=int(plan.nfreq),
                    subband_counts=[int(c) for c in t.frequency_subbands.subband_counts])

        a = np.ascontiguousarray(A[itree], dtype=np.float64)
        want = (info['ndm_out'], info['M'], info['P'], info['nfreq'])
        if a.shape != want:
            raise RuntimeError(f'write_variance_map: tree {itree} has shape {a.shape},'
                               f' expected {want}')
        trees.append(info)
        arrays.append(a)

    tree = {
        'variance_map': {
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'config_yaml': plan.config.to_yaml_string(),
            'detrender_yaml': detrender.to_yaml_string() if (detrender is not None) else None,
            'overrides': list(overrides or []),
            'ntime': int(ntime),
            'nphases': int(nphases),
            'ndata_chunks': int(ndata_chunks),
            'guard_chunk': bool(guard_chunk),
            'nbeams': int(nbeams),
            'device': device,
            'trees': trees,
            'A': arrays,
        }
    }

    asdf.AsdfFile(tree).write_to(str(filename))


def read_variance_map(filename, lazy=True):
    """Reads a variance map written by write_variance_map(), returning a VarianceMapFile.

    With lazy=True (the default) the arrays are memmapped and the returned object owns an
    open file handle -- use it as a context manager, or close() it. This is what makes a map
    larger than RAM usable: opening is O(1) and a slice reads only what it touches.

    With lazy=False everything is materialized and there is no handle to manage.
    """

    import asdf

    af = asdf.open(str(filename), lazy_load=lazy, memmap=lazy)
    try:
        tree = af['variance_map']
    except KeyError:
        af.close()
        raise RuntimeError(f"{filename}: not a variance-map file (no 'variance_map' key)")

    if lazy:
        return VarianceMapFile(af, tree, lazy=True)

    # Materialize before closing: np.asarray on an NDArrayType forces the read.
    tree = dict(tree)
    tree['A'] = [np.asarray(a) for a in tree['A']]
    tree['trees'] = [dict(d) for d in tree['trees']]
    af.close()
    return VarianceMapFile(None, tree, lazy=False)


def test_variance_map_io(toplevel_tree_rank=8, num_early_triggers=0, verbose=True):
    """Round-trips a real sweep through an ASDF file: arrays, config, detrender and
    per-tree metadata must all survive, and a closed lazy read must raise rather than
    hand back a dangling memmap.

    Also asserts that a lazy read really is memmapped. That is the cheap half of the
    large-file property measured by hand when the format was chosen -- an asdf upgrade
    that changed its lazy-load defaults would silently turn every large read into a full
    materialization, and nothing else here would notice.
    """

    import os
    import tempfile

    from ..pirate_pybind11 import DedispersionPlan
    from .brute_force import BruteForceVarianceMap, _make_test_config, _make_test_detrender
    from ..utils import atomic_print

    config = _make_test_config(toplevel_tree_rank, [2, 2, 1],
                               num_early_triggers=num_early_triggers)
    plan = DedispersionPlan(config, cdd2_kernel_required=False)
    dparams = _make_test_detrender(config)
    bf = BruteForceVarianceMap(plan, detrender=dparams)
    A = bf.run()

    path = os.path.join(tempfile.mkdtemp(), 'vm.asdf')
    try:
        write_variance_map(path, A, plan, detrender=dparams, device='cpu',
                           overrides=['test: none'], ntime=bf.ntime, nphases=bf.nphases,
                           ndata_chunks=bf.ndata_chunks, nbeams=bf.nbeams)

        for lazy in (True, False):
            with read_variance_map(path, lazy=lazy) as vm:
                assert len(vm.A) == len(A), (len(vm.A), len(A))
                for itree in range(len(A)):
                    assert np.array_equal(np.asarray(vm.A[itree]), A[itree]), itree
                    t = vm.trees[itree]
                    assert (t.r, t.R, t.M, t.P) == (bf.tree_r[itree], bf.tree_R[itree],
                                                    bf.tree_M[itree], bf.tree_P[itree])
                    assert vm.matrix(itree).shape == (t.ndm_out * t.M * t.P, t.nfreq)

                    # subband_counts is the tree's RESTRICTED vector, so it must match the
                    # plan's rather than the config's -- the two differ whenever an early
                    # trigger or a smaller pf_rank restricts it, which is exactly the case
                    # this field exists to record.
                    fs = plan.trees[itree].frequency_subbands
                    assert t.subband_counts == tuple(int(c) for c in fs.subband_counts), \
                        (itree, t.subband_counts, fs.subband_counts)
                    assert len(t.subband_counts) == t.R + 1
                    assert t.N == int(fs.N), (itree, t.N, fs.N)
                    assert sum(c << l for l, c in enumerate(t.subband_counts)) == t.M

                assert vm.ntime == bf.ntime and vm.nphases == bf.nphases
                assert vm.nbeams == bf.nbeams and vm.device == 'cpu'
                assert vm.overrides == ['test: none']

                # The inputs survive as yaml and re-parse into equal objects.
                assert int(vm.config.toplevel_tree_rank) == int(config.toplevel_tree_rank)
                assert int(vm.config.get_total_nfreq()) == int(config.get_total_nfreq())
                assert list(vm.config.frequency_subband_counts) == \
                    list(config.frequency_subband_counts)
                d = vm.detrender
                for field in ('nfreq', 'M', 'n_phi', 'n', 'W', 'T', 'eta', 'eps'):
                    assert getattr(d, field) == getattr(dparams, field), field
                assert list(d.knots) == list(dparams.knots)

                if lazy:
                    # Uncompressed blocks are what make this possible; see the module
                    # docstring on why compression is off.
                    a = vm.A[0]
                    assert isinstance(getattr(a, 'base', None), np.memmap), type(a)

        # Use-after-close must be an intelligible exception, not a dangling memmap.
        vm = read_variance_map(path, lazy=True)
        vm.close()
        try:
            vm.A
            raise AssertionError('use-after-close did not raise')
        except RuntimeError as e:
            assert 'closed' in str(e), str(e)

        # A file that is not a variance map is rejected by name.
        import asdf
        other = os.path.join(os.path.dirname(path), 'other.asdf')
        asdf.AsdfFile({'something_else': 1}).write_to(other)
        try:
            read_variance_map(other)
            raise AssertionError('read_variance_map accepted a non-variance-map file')
        except RuntimeError as e:
            assert 'variance_map' in str(e), str(e)

        # A pre-migration file (no subband_counts) is rejected with a message that says what
        # to do about it, rather than a bare KeyError from deep inside the reader.
        with asdf.open(path, lazy_load=False, memmap=False) as af:
            tree = {'variance_map': dict(af['variance_map'])}
            tree['variance_map']['trees'] = [{k: v for k, v in d.items()
                                              if k != 'subband_counts'}
                                             for d in tree['variance_map']['trees']]
            old = os.path.join(os.path.dirname(path), 'old_format.asdf')
            asdf.AsdfFile(tree).write_to(old)
        try:
            read_variance_map(old)
            raise AssertionError('read_variance_map accepted a pre-migration file')
        except RuntimeError as e:
            assert 'subband_counts' in str(e) and 'migrat' in str(e), str(e)

        nbytes = os.path.getsize(path)
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    if verbose:
        atomic_print(f"    test_variance_map_io(r={toplevel_tree_rank},"
                     f" et={num_early_triggers}): {len(A)} tree(s) round-tripped lazily and"
                     f" eagerly, {nbytes/2**20:.1f} MiB file, memmap + use-after-close checked")
