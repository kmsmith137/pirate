"""The variance-map file format: one ASDF file per VarianceMultiMap.

Reached through ``VarianceMap.write_asdf()`` / ``.from_asdf()`` and
``VarianceMultiMap.write_asdf()`` / ``.from_asdf()`` / ``.open_asdf()``; this module holds
the format itself and is not normally imported directly.

What the format has to do
-------------------------
1. Be readable with NO DedispersionPlan, hence on a machine with no GPU. That is why the
   config is stored as a yaml string and the per-tree geometry is stored explicitly rather
   than re-derived.
2. Support every representation: dense or factored, coarse or fine, certified or not.
3. Store y_true, at FINE granularity -- the only part of A_true that survives
   coarse-graining and low-rank approximation, and the one thing D cannot be computed
   without.
4. Store arrays UNCOMPRESSED. Compression buys little on dense float64 and costs CPU on
   every read; leaving it off is also what lets asdf memmap the blocks, which is the scale
   path for a map far larger than RAM (open_multimap() below).

The tree
--------
::

    variance_multimap:
      format_version:  1
      created:         ISO8601 UTC string
      config_yaml:     str                 # DedispersionConfig.to_yaml_string()
      detrender_yaml:  str or None         # Detrender2dParams.to_yaml_string()
      provenance:      dict                # free-form; how the SWEEP was run
      trees:
        - itree:              int
          tree_yaml:          str          # DedispersionTree.to_yaml_string(config, itree)
          m_to_n:             (M,) int64
          is_coarse_grained:  bool
          L:                  int or None
          nbeta:              int
          is_admissible:      bool
          is_factored:        bool
          history:            list of dicts
          A:                  (nbeta, nfreq) float          # if not is_factored
          Q:                  (nbeta, factor_rank) float    # if is_factored
          mid:                (factor_rank, factor_rank) float
          W:                  (nfreq, factor_rank) float
          factor_rank:        int
          Q_is_semiorthogonal:  bool
          W_is_semiorthogonal:  bool
          pinned_columns:     (npinned,) int64
          y_true:             (nalpha,) float64 or None

Exactly ONE of the two array groups is present in a tree block: ``A`` for a dense map, or
``Q`` / ``mid`` / ``W`` and their descriptors for a factored one. ``is_factored`` says which,
and the reader CHECKS it against the arrays rather than believing it -- a block carrying both
groups, or neither, is refused by name, because that is the case where trusting the flag
would silently reinterpret a matrix.

Arrays live inside their tree's dict rather than in a parallel top-level list -- asdf
memmaps nested ndarrays just as well, and it removes an index-alignment invariant that has
no upside. ``provenance`` is one free-form dict rather than a fixed set of named scalars,
because new algorithms will want to record things this format has not thought of and a
reader must never need a schema change to load a file; the keys a sweep writes are
documented by the sweep, not enforced here.

What is checked on read, and why
--------------------------------
The reader is deliberately paranoid, because the archived library is hundreds of GiB that
cannot be regenerated cheaply, and every failure mode below would otherwise surface as a
silently reinterpreted map rather than an error:

- ``DedispersionTree.check_consistency(config)``, which rebuilds the tree from the config
  and compares every geometry member. This catches a file written by a build whose
  dedispersion-tree geometry differs from this one's.
- ``m_to_n`` against the table rebuilt from the tree's ``frequency_subband_counts``. This
  is the one field with no independent witness -- the tree yaml stores the counts, and
  FrequencySubbands rebuilds m_to_n from them on read -- so a silent change to the
  multiplet ordering convention would reinterpret every archived map. Stored as a tripwire.
- ``is_coarse_grained`` against ``L``, ``nbeta`` against the array, and ``itree`` against
  the tree's own (primary_tree_index, early_trigger_level).

``format_version`` exists from the start, so the next format change gets a clean error
rather than a KeyError from deep inside the reader. There is no migration path: a file
without it is rejected by name.

'Dcore' in a variance-map file is NEVER authoritative
-----------------------------------------------------
The tree yaml carries a ``Dcore``, and it is always the placeholder
``pf.time_downsampling``: varmap builds trees with ``Dcore_from_cdd2_registry=False`` (see
VarianceMap.make_tree), and the brute-force sweep runs with the cdd2 kernel not required.
Do not decode peak-finder tokens with it -- the authoritative value is the producer's, and
it travels in the FrbGrouper handshake, not here.
"""

import contextlib
import dataclasses
import time

import numpy as np

from .VarianceMap import VarianceMap
from .VarianceMultiMap import VarianceMultiMap


FORMAT_VERSION = 1

# The top-level key. Deliberately NOT the old format's 'variance_map': the two are
# incompatible, and a name collision would turn "wrong format" into a confusing field error.
ROOT_KEY = 'variance_multimap'


####################################   helpers   ####################################


def _plain(x):
    """Recursively convert numpy scalars, tuples and dataclasses to plain python, for the
    free-form 'provenance' and 'history' records.

    Those are free-form on purpose, so the conversion is by TYPE rather than by schema.
    ndarrays are passed through untouched: asdf stores them as blocks, which is the right
    thing for anything large enough to be an array in the first place.

    THE DATACLASS CASE IS THE LOAD-BEARING ONE. A step's history record is meant to carry the
    config it ran under, and the natural way to write that is to stash the config OBJECT.
    asdf cannot represent one, and the failure is not local: it takes down the write of the
    whole map, so it surfaces when a long run tries to save its result. Converting here means
    a caller cannot lose a run that way.

    Note the round trip is deliberately ASYMMETRIC -- a dataclass goes out and a dict comes
    back. Reconstructing the original type would need the file to name it, which is a schema,
    and 'free-form' is the property that matters more.
    """

    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _plain(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _plain(v) for (k, v) in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _root(af, filename):
    """The 'variance_multimap' block of an open AsdfFile, with the format checks."""

    try:
        root = af[ROOT_KEY]
    except KeyError:
        old = ('variance_map' in af.tree)
        raise RuntimeError(
            f"{filename}: not a variance-map file (no {ROOT_KEY!r} key)."
            + (" It IS an old-format file (top-level key 'variance_map', written by"
               ' pirate_frb.slow_avar.variance_map_io). The two formats are incompatible and'
               ' there is no migration path in the reader: convert the file instead.'
               if old else '')) from None

    v = root.get('format_version')
    if v is None:
        raise RuntimeError(f'{filename}: has a {ROOT_KEY!r} block with no'
                           " 'format_version', so it predates this format entirely.")
    if int(v) != FORMAT_VERSION:
        raise RuntimeError(f'{filename}: format_version is {v}, but this build reads only'
                           f' version {FORMAT_VERSION}.')
    return root


def _read_inputs(root):
    """(config, detrender) from the root block's yaml strings.

    Both are re-parsed here rather than stored per tree: they are metadata of the whole
    file, and the multimap requires every per-tree map to hold the SAME objects.
    """

    from ..pirate_pybind11 import DedispersionConfig, Detrender2dParams

    config = DedispersionConfig.from_yaml_string(root['config_yaml'])

    dy = root.get('detrender_yaml')
    detrender = Detrender2dParams.from_yaml_string(dy) if (dy is not None) else None

    return config, detrender


####################################   writing   ####################################


def _tree_dict(m):
    """The per-tree block for one VarianceMap.

    Exactly one of the two array groups is written: `A` for a dense map, or
    `Q`/`mid`/`W` plus their descriptors for a factored one. The reader refuses a block that
    carries both or neither, so `is_factored` can never be believed over the arrays.
    """

    d = dict(itree=int(m.itree),
             tree_yaml=m.tree.to_yaml_string(m.config, m.itree),
             m_to_n=np.ascontiguousarray(m.m_to_n, dtype=np.int64),
             is_coarse_grained=bool(m.is_coarse_grained),
             L=(int(m.L) if (m.L is not None) else None),
             nbeta=int(m.nbeta),
             is_admissible=bool(m.is_admissible),
             is_factored=bool(m.is_factored),
             history=[_plain(dict(h)) for h in m.history],
             y_true=(np.ascontiguousarray(m.y_true, dtype=np.float64)
                     if (m.y_true is not None) else None))

    # Every array below goes out as a BASE-CLASS view, not the stored object: asdf refuses
    # ndarray subclasses outright, and the scale path hands us an np.memmap (a matrix
    # accumulated on disk, or one read back from open_multimap()). asarray() is free here --
    # same dtype and order, so it is a view, not a copy of 344 GiB.
    if not m.is_factored:
        d['A'] = np.asarray(m.A)
    else:
        d['Q'] = np.asarray(m.Q)
        d['mid'] = np.asarray(m.mid)
        d['W'] = np.asarray(m.W)
        d['factor_rank'] = int(m.factor_rank)
        d['Q_is_semiorthogonal'] = bool(m.Q_is_semiorthogonal)
        d['W_is_semiorthogonal'] = bool(m.W_is_semiorthogonal)
        d['pinned_columns'] = np.ascontiguousarray(m.pinned_columns, dtype=np.int64)

    return d


def _write(maps, config, detrender, provenance, filename):
    """Assemble the root block and write it.

    The matrices are handed to asdf as they are, so a memmapped or otherwise on-disk-backed
    one streams through rather than being materialized -- which is what makes writing a map
    far larger than RAM possible.
    """

    import asdf

    root = {'format_version': FORMAT_VERSION,
            'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'config_yaml': config.to_yaml_string(),
            'detrender_yaml': (detrender.to_yaml_string()
                               if (detrender is not None) else None),
            'provenance': _plain(dict(provenance or {})),
            'trees': [_tree_dict(m) for m in maps]}

    asdf.AsdfFile({ROOT_KEY: root}).write_to(str(filename))


def write_multimap(vmm, filename, *, provenance=None):
    """Write a VarianceMultiMap to 'filename'.

    Parameters
    ----------
    provenance : dict, optional
        Overrides ``vmm.provenance``. Free-form; see the module docstring.
    """

    prov = vmm.provenance if (provenance is None) else provenance
    _write(list(vmm), vmm.config, vmm.detrender, prov, filename)


def write_map(m, filename, *, provenance=None):
    """Write a single VarianceMap to 'filename', as a file holding one tree.

    The result is the same format as write_multimap()'s, with a one-element 'trees' list,
    and is read back with ``VarianceMap.from_asdf(filename, m.itree)``. It is a complete
    multimap file -- readable by ``VarianceMultiMap.from_asdf()`` -- only when the config
    has a single dedispersion tree; otherwise that reader refuses it, since a multimap
    covers every tree by definition.
    """

    _write([m], m.config, m.detrender, provenance, filename)


####################################   reading   ####################################


def _read_tree(d, config, detrender, filename):
    """One VarianceMap from a per-tree block. See the module docstring for what is checked."""

    from ..pirate_pybind11 import DedispersionTree

    itree = int(d['itree'])
    where = f'{filename}: tree {itree}'

    # ---- geometry, and the three checks on it ----

    tree = DedispersionTree.from_yaml_string(d['tree_yaml'], config)

    # Compares every geometry member against a tree rebuilt from the config, and raises
    # naming the member that disagrees.
    tree.check_consistency(config)

    ir = int(config.dedispersion_tree_index(tree.primary_tree_index,
                                            tree.early_trigger_level))
    if ir != itree:
        raise RuntimeError(f"{where}: the stored tree yaml is tree {ir} of this config"
                           f" (primary_tree_index={tree.primary_tree_index},"
                           f" early_trigger_level={tree.early_trigger_level}), not tree"
                           f" {itree} as the 'itree' field claims.")

    stored = np.asarray(d['m_to_n'], dtype=np.int64)
    rebuilt = np.asarray(tree.frequency_subbands.m_to_n, dtype=np.int64)
    if not np.array_equal(stored, rebuilt):
        i = int(np.flatnonzero(stored != rebuilt)[0]) if (stored.shape == rebuilt.shape) else -1
        raise RuntimeError(
            f'{where}: the stored m_to_n disagrees with the one FrequencySubbands rebuilds'
            f' from frequency_subband_counts'
            + (f' (first at m={i}: stored {stored[i]}, rebuilt {rebuilt[i]})' if i >= 0
               else f' (stored shape {stored.shape}, rebuilt {rebuilt.shape})')
            + '. The multiplet ordering convention has changed, which reinterprets every'
              ' archived map -- do not "fix" this by dropping the check.')

    # ---- representation ----

    L = d.get('L')
    L = int(L) if (L is not None) else None

    if bool(d['is_coarse_grained']) != (L is not None):
        raise RuntimeError(f"{where}: is_coarse_grained={d['is_coarse_grained']} but L={L}."
                           ' The two are one fact, and a file where they disagree was written'
                           ' by something that did not treat them that way.')

    # The flag is checked AGAINST the arrays rather than believed. A block that carries both
    # groups, or neither, is the case where trusting 'is_factored' would silently reinterpret
    # a matrix, so it is refused by name.
    is_factored = bool(d['is_factored'])
    has_dense = (d.get('A') is not None)
    has_factors = any(d.get(k) is not None for k in ('Q', 'mid', 'W'))

    if has_dense and has_factors:
        raise RuntimeError(f'{where}: carries BOTH a dense A and factors (Q/mid/W). Exactly'
                           ' one group is written, and which one is what is_factored means.')
    if is_factored and not has_factors:
        raise RuntimeError(f'{where}: is_factored is True but the block carries no'
                           ' Q/mid/W -- it holds'
                           + (' a dense A.' if has_dense else ' no matrix at all.'))
    if (not is_factored) and has_factors:
        raise RuntimeError(f'{where}: is_factored is False but the block carries Q/mid/W.')
    if (not is_factored) and (not has_dense):
        raise RuntimeError(f'{where}: is_factored is False but the block carries no A.')

    y_true = d.get('y_true')
    if y_true is not None:
        y_true = np.asarray(y_true)

    common = dict(y_true=y_true, L=L, is_admissible=bool(d['is_admissible']),
                  history=[dict(h) for h in d.get('history', [])], tree=tree)

    # np.asarray() below is a view into the memmapped block in the lazy case (no copy, and no
    # asdf object left in the map), and a materializing read in the eager one.
    if not is_factored:
        A = d['A']
        if int(d['nbeta']) != int(A.shape[0]):
            raise RuntimeError(f"{where}: the stored nbeta is {d['nbeta']}, but A has"
                               f' {A.shape[0]} rows.')
        return VarianceMap(config, itree, detrender, A=np.asarray(A), **common)

    Q, mid, W = np.asarray(d['Q']), np.asarray(d['mid']), np.asarray(d['W'])
    K = int(d['factor_rank'])

    if int(d['nbeta']) != int(Q.shape[0]):
        raise RuntimeError(f"{where}: the stored nbeta is {d['nbeta']}, but Q has"
                           f' {Q.shape[0]} rows.')
    if (Q.shape[1] != K) or (W.shape[1] != K) or (mid.shape != (K, K)):
        raise RuntimeError(f"{where}: the stored factor_rank is {K}, but Q is {Q.shape},"
                           f' mid is {mid.shape} and W is {W.shape}.')

    return VarianceMap(config, itree, detrender, Q=Q, mid=mid, W=W,
                       pinned_columns=np.asarray(d.get('pinned_columns'), dtype=np.int64),
                       Q_is_semiorthogonal=bool(d.get('Q_is_semiorthogonal', False)),
                       W_is_semiorthogonal=bool(d.get('W_is_semiorthogonal', False)),
                       **common)


def _multimap_from_root(root, filename):
    """A VarianceMultiMap from a root block, with every tree present and in order."""

    config, detrender = _read_inputs(root)
    entries = list(root['trees'])

    ntrees = int(config.num_dedispersion_trees)
    if len(entries) != ntrees:
        got = [int(e['itree']) for e in entries]
        raise RuntimeError(
            f'{filename}: holds {len(entries)} tree(s) {got}, but its config has {ntrees}'
            ' dedispersion trees. A VarianceMultiMap covers EVERY tree by definition; read a'
            ' single-tree file with VarianceMap.from_asdf(filename, itree) instead.')

    maps = [_read_tree(d, config, detrender, filename) for d in entries]
    return VarianceMultiMap(config, maps, detrender=detrender,
                            provenance=dict(root.get('provenance') or {}))


def read_multimap(filename):
    """Read a variance-map file EAGERLY: ordinary in-memory arrays, no open handle, no
    lifetime contract. This is the one to use unless you have a specific reason not to."""

    import asdf

    with asdf.open(str(filename), lazy_load=False, memmap=False) as af:
        return _multimap_from_root(_root(af, filename), filename)


@contextlib.contextmanager
def open_multimap(filename):
    """Read a variance-map file with its arrays MEMMAPPED, as a context manager::

        with VarianceMultiMap.open_asdf(path) as vmm:
            ...

    The arrays are views into an open file handle and are invalid once the with-block
    exits. Scoping it syntactically is the whole point: it keeps the lifetime contract off
    the common path, where read_multimap() has none at all.

    Only worth it for a file too large to hold. That is a real case -- Abar is 86 GiB at
    CHORD with L = 6 -- but it is much rarer than it was, since scoring reads no reference
    matrix and the dense fine map is only produced at subscale.
    """

    import asdf

    af = asdf.open(str(filename), lazy_load=True, memmap=True)
    try:
        yield _multimap_from_root(_root(af, filename), filename)
    finally:
        af.close()


def read_map(filename, itree=0):
    """Read ONE tree out of a variance-map file, eagerly.

    Unlike read_multimap() this does not require the file to cover every tree, so it is how
    a single-tree file (write_map()'s output) is read back.
    """

    import asdf

    itree = int(itree)
    with asdf.open(str(filename), lazy_load=False, memmap=False) as af:
        root = _root(af, filename)
        config, detrender = _read_inputs(root)

        entries = [d for d in root['trees'] if int(d['itree']) == itree]
        if len(entries) != 1:
            got = [int(d['itree']) for d in root['trees']]
            raise RuntimeError(f'{filename}: asked for tree {itree}, but the file holds'
                               f' {len(entries)} entries for it (trees present: {got}).')

        return _read_tree(entries[0], config, detrender, filename)
