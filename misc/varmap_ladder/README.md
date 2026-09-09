# The variance-map benchmark ladder

A **variance map** $A$ of a dedispersion tree answers: if input frequency channel $F$ has
variance $v_F$, what is the variance of peak-finding output $\alpha$? It is the nonnegative
matrix in

$$ y_\alpha \;=\; \sum_F A[\alpha, F]\, v_F . $$

For the CHORD base tree it is $5\,963\,776 \times 28\,160$, which is 1.2 TiB dense, so the
real-time search cannot store it. Instead we store a **low-rank approximation**
$A \approx Q\,W^{T}$ with $W$ of shape $(N_{\rm freq}, K)$ and $K \sim 128$, and the question
is how good an approximation a given amount of compute buys.

Approximation quality is measured by a **one-sided distance** $D$. It is $+\infty$ if the
approximation underestimates $y_\alpha$ anywhere, and otherwise the mean over outputs of
$f\!\left(y^{\rm approx}_\alpha / y^{\rm true}_\alpha\right)$ with $f(x) = (x-1)/(1 + x/10)$.
Overestimating the variance is safe and costs sensitivity; underestimating it is a hard
failure, because the search would then run with a threshold it believes is conservative and is
not. Everything below is arranged around that asymmetry.

The **ladder** is a family of progressively cheaper versions of the same problem, so that an
idea can be tried for tens of core-hours before it is paid for at full scale. It is defined by
coarse-graining, not by shrinking the instrument: **every rung has the same config, the same
detrender, the same 28160 frequency channels, and is scored against the same 5963776 true
outputs.**

**Two files in this directory are the definition of $A$, and both are load-bearing:**

| file | what it fixes |
|---|---|
| `config_chord_base_tree.yml` | the dedispersion tree: `chord_sb2_et.yml` with `primary_trees` cut to its first entry |
| `detrender_chord.yml` | the `DetrenderLps2dParams` in the linear operator ahead of it |

**A sweep run with `--no-detrender` produces a different matrix and does not reproduce
anything below.** The detrender is not a perturbation: it is what destroys the exact zero
structure of $A$ (a multiplet gets no contribution from channels outside its subband, until a
spline that straddles the boundary gives it one) and it is most of what makes $A$ numerically
hard. Concretely, at rung 0 the two references are $400 \times 28160$ matrices of the same
geometry, and their singular spectra are not comparable: the detrender-free one has **exact
rank 19**, so a rank-128 approximation of it is not an approximation at all and $D$ saturates
at $0.0872$ from $K = 19$ upward, whereas the detrended one has full numerical rank 400 (353
singular values above $10^{-6}$ of the largest) and gives $D = 0.161$ at $K = 128$.
Section 2.2 says what the detrender-free map is good for.

---

## 1. The rungs

Rung $j$ groups the tree's output DMs in blocks of $2^{\,12-j}$, so it has
$N_\beta = 2^{\,j} \times 400$ rows. Each rung has exactly twice the rows of the one below,
and since the work is one small optimization problem per row, **each rung costs twice the one
below**. The top rung is the production problem itself, not a model of it.

| rung | $L$ | $N_\beta$ | reference on disk | $D$ at $K=128$ | $D/D_{\rm prod}$ | one $K=128$ cell, core-hours |
|---|---|---|---|---|---|---|
| 0  | 16 | 400     | 0.13 GiB  | 0.161113 | 6.63 | 1.2 |
| 1  | 15 | 800     | 0.21 GiB  | 0.086993 | 3.58 | 2.8 |
| 2  | 14 | 1 600   | 0.38 GiB  | 0.051885 | 2.13 | 4.6 |
| 3  | 13 | 3 200   | 0.72 GiB  | 0.034953 | 1.44 | 6.3 |
| 4  | 12 | 6 400   | 1.39 GiB  | 0.028904 | 1.19 | 11.8 |
| 5  | 11 | 12 800  | 2.73 GiB  | 0.027588 | 1.14 | **20.7** |
| 6  | 10 | 25 600  | 5.42 GiB  | 0.027614 | 1.14 | 37.3 |
| 7  | 9  | 51 200  | 10.79 GiB | 0.025755 | 1.06 | **63.1** |
| 8  | 8  | 102 400 | 21.53 GiB | 0.025021 | 1.03 | 149 [a] |
| 9  | 7  | 204 800 | 43.01 GiB | *no reference value* | -- | ~290 [a] |
| 10 | 6  | 409 600 | 86 GiB    | **0.024309** | 1 | **514** [a] |

Rungs 0-7 were re-measured end to end against a freshly swept reference, and every one
reproduces the tabulated $D$ to better than $1.2\times10^{-5}$ relative. Their core-hours are
CPU time actually consumed, self plus children through `os.times()`, on 56 workers -- not
wall-clock times a worker count, which overstates a cell by up to 1.5x because the pool sits
idle through the serial repair. The three rows marked **[a]** are campaign-4 figures that were
not re-run; rung 9's has never been measured at all.

**Cost does not quite double per rung, and cost per row is not flat.** Measured over rungs 0-7
the step is 1.4x to 2.3x rather than a clean 2x, and the cost per row falls steadily, from
$3.1\times10^{-3}$ core-hours at rung 0 to $1.2\times10^{-3}$ at rung 7. Part of that is a
fixed per-cell term -- `recommended('q')` builds a sampled constraint pool whose cost does not
scale with $N_\beta$, so it dominates a small rung and amortizes away on a large one -- and
part is that these seven cells shared the machine with the GPU sweep that produced their
reference. Budget a rung at 1.7x to 2x the one below and treat any single figure as good to
tens of percent; the row count, unlike the cost, doubles exactly.

**Cost figures are only comparable when the solver settings match.** All of these use
`LpConfig.recommended('q')` (section 3, step 5). Changing one of its constraint-generation
settings can move the cost of a high-rank cell by more than a factor of two while leaving the
answer identical to fourteen digits, so a timing quoted without its settings means nothing --
and the [a] rows above were taken before `cuts_pool_sample` was added to the preset, which is
one such change.

$L$ is the coarse-graining level in the code's convention, $L = 16 - j$; the two are the same
thing counted from opposite ends. Rung 10 is the production grouping, and is exactly the
grouping the real-time peak-finding weights use ($1024$ DMs $\times$ 25 subbands $\times$ 16
profiles, i.e. the config's `wt_dm_downsampling: 64`). The elementwise floor -- the best any
rank could do at rung 10 -- is $D = 0.00151$.

**How to read the last two columns.** The ladder predicts **cost** faithfully: the row count is
a factor of two per rung by construction, and the extrapolation to production was verified to
within 3%. It does
**not** predict $D$ by extrapolation -- fitting a power law up the ladder and evaluating it at
rung 10 is wrong by a factor of two. What it does give is a **conservative bound**: from rung 5
upward each rung's $D$ is within 3-14% *above* the production value, because a coarser grouping
is a looser constraint and can only make the approximation worse. So rung 5 (21 core-hours) or
rung 7 (63 core-hours) is the place to compare two ideas, and the comparison transfers; rungs 0
and 1 do not transfer, because there the coarse grouping dominates the error and increasing the
rank buys almost nothing (an eightfold rank increase moves rung 0 by 1.35x, against 5.67x at
rung 10).

Rung 6 sits a hair *above* rung 5, which looks wrong and is not: $D$ at this geometry is only
reproducible to about 6% (see section 3, "Scoring"), and the gap is 0.1%.

---

## 2. How the reference matrices are precomputed and stored

Every rung needs its own **reference matrix** $\bar A$ -- the coarse-grained variance map at
that rung's grouping. These are computed once, written to disk, and thereafter only read.

**Coarse-graining is a maximum, not a mean.** Row $\beta$ of the reference is the elementwise
maximum over the fine rows it contains,

$$ \bar A[\beta, F] \;=\; \max_{\alpha \,\in\, \beta} A[\alpha, F] . $$

The maximum is what preserves the one-sided guarantee: an approximation that dominates the
envelope $\bar A$ automatically dominates every fine row inside it, so admissibility at a
coarse grouping implies admissibility at the fine one. A mean would not have this property, and
an approximation built against a mean could underestimate half the outputs it covers.

The groups are blocks of $2^{\,L-R}$ adjacent output DMs, with $R = 4$ here -- 4096 of them at
rung 0, four at rung 10 -- together with the merge of the 91 multiplets into their 25 subbands.
**The groups never cross a frequency-subband or peak-finding-profile boundary** -- rows that
differ in subband or profile describe different frequency ranges and different pulse widths,
and merging across those boundaries is not a coarser description of the same thing but a wrong
one. Measured, it costs factors of several hundred to several thousand in $D$.

**The chain, computed once.** The fine matrix $A$ is never formed; a GPU sweep produces the
grouped matrix directly, one frequency channel at a time. The sweep is told to group at $L=6$,
so its output is rung 10 -- both the production reference and the source every other rung is
derived from:

```
   GPU sweep at L = 6, config + detrender (~5.5 GPU-hours, once)
        |
        v
   rung 10 reference, L = 6            86 GiB     <-- production, AND the source below
        |
        |  nine successive grouped maxima, chained     ~8 min
        v
   rungs 9 .. 0, L = 7 .. 16           86.3 GiB total
```

Nothing in the ladder needs a grouping finer than $L=6$. A finer sweep is legal -- the code
accepts any $R \le L \le r$ -- but the sweep's cost is one pass per *input channel* and is
independent of $L$, so a finer grouping costs the same GPU time to produce a larger matrix
that then has to be coarsened anyway. For the same reason, a cheap partial ladder is a sweep
told to group at $L = 11$ directly: same GPU time, 2.7 GiB instead of 86, and it yields rungs
0 through 5.

Grouping inside the sweep is not an approximation to grouping afterwards. Checked at CHORD: a
sweep told to group at $L = 6$ and a sweep at $L = 4$ followed by `coarse_grain(6)` agree
**bitwise** -- `y_true` exactly, and every sampled 2048-row block of the $409\,600 \times
28\,160$ matrix, including the first and the last. That is what licenses picking $L$ by what
you can store rather than by what you can trust.

The lower rungs are derived from rung 10, which is sound because the maximum **nests**: a
maximum over a set of rung-10 rows equals the maximum over all the fine rows beneath them.
This is worth stating precisely because it is what makes the ladder cheap, and it is worth
checking rather than assuming -- each rung should reproduce *exactly* the elementwise maximum
over its two constituent rows in the rung below, including at the far corner of the index range
where an off-by-one in the row-ordering convention would show up and nowhere else.

**They are cached, not recomputed on demand.** All ten derived rungs together are 86.3 GiB --
about one extra copy of rung 10, because a geometric series sums to roughly twice its largest
term. Against that, each rung is read at least four times (once per rank $K$ that is tried),
and re-deriving rung 0 on demand would still mean streaming 86 GiB to produce 130 MiB. Caching
also pins the benchmark to fixed, verified inputs: a cell whose reference is re-derived at run
time is a cell whose input silently changes if the coarse-graining code ever changes.

**The true outputs travel with the matrix.** Each reference carries the vector $y^{\rm true}$
of the 5963776 fine output variances, unchanged by coarse-graining. That is why $D$ means the
same thing at every rung: the grouping changes how many rows the approximation has to work
with, never what it is scored against.

**A swept reference carries no admissibility certificate.** `varmap bf` returns $A_{\rm true}$
itself, and sets `is_admissible = False` -- not because it is suspect but because a measurement
is not a proof, and the class reserves the flag for something that has been established. This
has one consequence that will otherwise stop a run dead at the last step: the flag is
inherited, so a $Q$-step against a swept reference also comes back `is_admissible = False`, and
`get_distance()` refuses to score it. Step 7 says what to do about it, and it is one line.

### 2.1 Building these with the varmap code

Two operations produce everything above: a brute-force **sweep** that computes a variance map
from a config and a detrender, and a **coarse-graining** that turns one map into a coarser one.

**The sweep, from the command line.** This is the supported entry point:

```
pirate_frb varmap bf config_chord_base_tree.yml detrender_chord.yml \
        -o chord_t0_L6.asdf -L 6 -g 0
```

It prints every config field it overrides, and there are five: `beams_per_gpu`,
`beams_per_batch` and `num_active_batches` to 1 (the beam axis is a pure spectator, and
measurement found that batching does not speed up a full sweep), `max_gpu_clag` to 10000 (the
GPU sweep needs a MegaRingbuf with no host segments; this is a placement decision and cannot
change $A$), and `dtype` to float32. **The last of those is a conversion rather than a
choice**: `GpuSbDedispersionKernel` is float32-only, and the CPU sweep's
`ReferenceDedispersionKernel` "uses float32 regardless of what dtype is specified", so there is
no float16 sweep to offer -- but unlike the other four it is not provably $A$-preserving, so it
is announced and recorded in the file's provenance. Other flags worth knowing: `--channels`
sweeps a subset for timing and deliberately returns `y_true=None` so that nothing downstream
can score a partial map; `--scratch-dir` backs the accumulator with an on-disk memmap when it
does not fit in RAM; `--cpu` forces the CPU sweep; `--no-guard-chunk` turns off the one check
that matters (see below) and should be left alone.

**The sweep, from python.** `compute_variance_multimap(config, detrender, *, device='gpu',
L=6)` returns a `VarianceMultiMap`, one map per PRIMARY tree -- an early-trigger tree's map is
a row subset of its parent's, so it is derived rather than stored. Call
`ksgpu.set_cuda_device()` first, which is required even for `device='cpu'`, because
`DedispersionPlan` allocates pinned host memory. The library applies **none** of the CLI's
overrides: it checks `beams_per_gpu == beams_per_batch` and leaves the rest to the sweep's own
checks, so a config straight off disk will be refused. Either use the CLI or apply the
overrides yourself.
**Passing `L=` is what makes production scale possible**: the sweep max-reduces each column
into its groups as that column is produced, so the dense $A$ is never formed -- at CHORD that
is the difference between 86 GiB and 1.2 TiB. It requires `nphases == 1`, which for a
single-primary-tree config is automatic. `L` may also be a per-primary-tree sequence, whose
entries may individually be `None`, because the legal range $R \le L \le r$ differs per tree.
Leave `guard_chunk=True`: it runs one extra all-zero chunk and requires the peak-finding output
to be identically zero, and it is the only check that the impulse response was fully emitted --
an undersized sweep silently *underestimates* $A$, which is the one failure mode that matters.

**Writing and reading.** `mm.write_asdf(path, provenance={...})`. Budget for the checksum:
`asdf` checksums each block by default, single-threaded, which is several minutes for a block
the size of rung 10. **The file format is at version 3** -- one entry per primary tree plus one
top-level plan yaml. Versions 1 and 2 are refused by name rather than misread, and there is no
converter, so an archived file from an older build has to be re-swept. Read a whole multimap
back with `VarianceMultiMap.from_asdf(path)`, which is eager and is the one to use unless you
have a reason not to; at rung-9 and rung-10 sizes use `VarianceMultiMap.open_asdf(path)` as a
context manager instead, which is lazy and memory-maps the matrix. Then
**`mm.primary_map(0)`, not `mm[0]`**: `VarianceMultiMap` deliberately has no `__getitem__`,
because the old one was indexed by tree and a silent change of meaning to primary-tree index
would have left every call site working and wrong. `VarianceMap.from_asdf(path, gamma)` reads a
single map directly; there is no `VarianceMap.open_asdf`.

**Coarsening.** `m.coarse_grain(L)` on a single map, or `mm.coarse_grain(Ls)` on the multimap,
where `Ls` may be a per-primary-tree sequence because the legal range $R \le L \le r$ differs
per tree. Coarsening an already-coarse map to a coarser level is supported, and is how rungs 9
down to 0 are produced from rung 10.

**Verifying.** `check_ref_covers_y_true()` on each derived map, and `measure_admissibility()`
for the elementwise figures; `VarianceMultiMap.measure_admissibility(ref)` runs it per primary
tree and documents why certifying those certifies every tree. For the nesting check described
above -- that a rung's row is the elementwise maximum over its constituents in the rung below
-- `group_members(beta)` and `alpha_to_beta_block(start, stop, L)` give the index arithmetic.

### 2.2 The analytic map, and why it is not this ladder

There is a second way to get a reference, and it does not sweep anything:
`compute_detrender_free_base_map(config, *, L=6, epsilon=None, max_bytes=None,
svd_optimize=True)`, or `pirate_frb varmap df config.yml -o out.asdf -L 6`. It computes $A$ in
closed form -- no dedisperser, no `DedispersionPlan`, no GPU -- returns it **factored**, and
sets `is_admissible = True`, which it has earned: the coarse map *is* the max-envelope by
construction, up to a per-group singular-value threshold of order $10^{-11}$. Coarse-graining
is free on this route rather than a reduction afterwards, because with no detrender the
max-envelope is a *slice* (the monotonicity theorem in the tex notes), so the coarse map is
emitted directly. Measured at CHORD, rung 10: **99.5 s** on eight threads (35.4 s to build the
factorization, 64.1 s for the exact rank reduction that takes it from 1399 columns to 553) and
**1.85 GiB** on disk, against 5.5 GPU-hours and 86 GiB for the swept reference.

**Its only limitation is the one that matters here: it is detrender-free.** It is the map of
the tree with the detrender taken out, which is a different matrix -- see the note at the top
of this file. So it does not reproduce section 1, and a $D$ measured on it is not comparable
with anything in that table.

What it is good for:

- **Exercising the whole chain in seconds, with no GPU and no sweep.** Rung 2 at $K = 16$ off
  an analytic reference runs end to end -- reference, SVD, sign canonicalization, seed,
  $Q$-step, repair, score, write, read back -- in **10 s** on 24 workers once the reference
  exists, and the reference itself is 16 s. That is the right first thing to run.
- **Bounding what the detrender costs.** The rank-19 fact quoted at the top of this file is
  measured on it, and is the cleanest statement available of how much of the difficulty is the
  detrender's doing.
- **Everything downstream of the reference is identical**, so a change to the $Q$-step or the
  basis can be exercised on it before a sweep is paid for.

Two caveats. A factored reference is densified block by block inside a $Q$-step
(`VarianceMap._lp_reference`), so this route saves the *sweep*, not the $Q$-step's memory.
And `svd()` on a factored map is always exact and cheap and ignores `method=`, so a ladder that
mixes analytic and swept references is not comparing one algorithm.

---

## 3. The sequence of steps at one rung

The same seven steps run at every rung, in the same order, with only the input file and the
rank $K$ changing. Holding the sequence fixed is what makes the rungs comparable.

**Step 1. Check the reference.** Confirm that $\bar A$ really does dominate $y^{\rm true}$ -- that the
envelope covers what it claims to cover. This is cheap and it is the guard on everything after
it: every later guarantee is of the form "the approximation dominates the reference", which is
worth nothing if the reference itself is wrong.

*In code*: `ref.check_ref_covers_y_true()`. It raises rather than returning a flag, so it is
a guard and not a query, and whatever builds a reference is expected to call it once. It is
cheap -- one blocked pass -- and it is the cheapest place to catch a reference that was
coarsened with the wrong index convention.

**Step 2. Build a basis.** Take a truncated singular value decomposition of $\bar A$ and keep the top
$K$ right singular vectors as the columns of $W$. This is the best rank-$K$ subspace in the
least-squares sense, which is *not* the sense we are optimizing -- but it is an excellent
starting subspace and it is cheap. At production scale the matrix is far too large for a direct
decomposition, so a randomized range-finder is used: project the matrix onto a modest random
subspace, orthogonalize, and refine with a power iteration. The number of extra random
directions matters more than the number of power iterations, and the settings here were chosen
by measuring the resulting $D$ rather than the residual, which selects noticeably different
values.

*In code*: `approx = ref.svd(K, method='randomized')`, which returns a factored `VarianceMap`
with `mid` $= \mathrm{diag}(s)$ and $W = V$. Two things about it are not obvious. At
$K \ge 32$ the `shape_normalize` default flips on, so what is decomposed is the unit-row-sum
*shape* matrix $S[\beta,F] = \bar A[\beta,F] / \sum_F \bar A[\beta,F]$ with the row sums folded
back into $Q$; that is the better $W$-matrix above that measured crossover, but it means
$Q$ is not semiorthogonal and `truncate()` will refuse the result -- ask `svd()` for the rank
you want instead. And `method=` is *ignored* when `self` is factored, where the decomposition is
always exact and cheap; pass `'randomized'` explicitly anyway, so that a ladder run against
dense swept references uses one algorithm at every rung and the $D$ column carries no
algorithmic step. Leave `oversample` (default $\max(48, K)$) and `power_iters` (default 2)
alone: they were tuned against $D$ rather than against the residual, and the textbook values
give a basis 1.3x-1.45x worse in $D$ for a barely smaller residual. All four of
`method`, `shape_normalize`, `oversample` and `power_iters` are recorded in the returned map's
`history`, so the provenance of a basis does not have to be written down by hand.

The SVD is one basis constructor among several -- `varmap.basis` also has
`basis_envelope_column`, `basis_greedy_envelope`, `basis_pivoted_qr` and `basis_random` (the
control) -- and the ladder fixes it deliberately. The point of a rung is to compare two *other*
ideas at a fixed basis; a run that changes the basis is measuring something else and its $D$
does not belong in the section-1 column.

One transformation is deliberately *not* in the recipe, and the reason is a measurement rather
than an oversight. `rescale_columns()` normalizes the columns of $W$ to unit 2-norm and absorbs
the reciprocal into `mid`. It is provably inert -- $(W_{:,c},\, q_c) \to (\lambda W_{:,c},\,
q_c/\lambda)$ is an exact symmetry of both the objective and the feasible set -- and yet its
docstring records it as worth **up to 1.49x in $D$** on subscale maps, with the obvious
explanation (the solver's absolute feasibility tolerance) tested and falsified. **On this ladder
it is worth nothing.** Measured at rungs 2 and 4 at $K = 128$, same reference, same basis, with
and without: $D$ moves by $2\times10^{-9}$ and $3\times10^{-8}$ relative, and the cost is
unchanged. So the ladder's $D$ column is not leaving a free factor on the table, and whatever
the subscale effect is, it is not a property of this geometry.

**Step 3. Canonicalize the signs.** The singular vectors are determined only up to sign, and the steps
that follow need columns that are nonnegative wherever possible: the covering constraint and
the repair both work by adding multiples of a nonnegative column, and a column that is negative
everywhere is the same subspace direction but useless for that purpose. This step flips each
column into the orientation that makes it usable, and is otherwise a no-op.

*In code*: `approx = approx.canonicalize_signs()`, no arguments. `approx.n_nonneg_cols()`
reports how many columns came out usable and is worth logging, because the next two steps
need at least one and fail loudly rather than quietly if there is none. On this geometry it
comes out at 1, which is enough but leaves no margin; if a variant of the recipe ever returns
0, the deterministic fix is `approx.pin_column(basis_envelope_column(ref))` from
`varmap.basis`, which appends a column that is nonnegative by construction and holds it fixed
through the steps.

**Step 4. Seed with a feasible point.** Before optimizing, construct an approximation that already
satisfies the constraint: for each group, pick the single nonnegative column of $W$ that covers
it most cheaply and scale it just enough to dominate. This is a rescaled envelope -- a
respectable approximation in its own right, and the baseline the whole low-rank exercise is
trying to beat. It also matters for a subtler reason: it is the fallback for any group whose
optimization fails. Starting from zero instead would turn a solver failure into a group that
silently underestimates, and at these sizes one such group costs more than an entire doubling
of the rank.

*In code*: `approx = approx.seed_onehot(ref)`. It requires at least one nonnegative column of
the atom matrix $W\,\mathrm{mid}^{T}$ -- which straight after `svd()` is $W$ rescaled by the
positive singular values, hence the same columns -- and raises if some group is covered by no
column at all. A seed that is not feasible is not a seed.

**Step 5. The $Q$-step.** With $W$ held fixed, choose the coefficient vector $q_\beta$ for each group
independently, by solving

*In code*: `approx = approx.qstep(ref, cfg=LpConfig.recommended('q'), workers=N,
progress=True)`. Use the `recommended('q')` preset rather than the bare defaults: relative to
`LpConfig()` it sets exactly `cuts=True`, `cuts_min_rows=1500`, `cuts_tol=1e-12`,
`cuts_pool_sample=32`, `cuts_pool_window=4`, `additive_first=True`, `additive_last=True` and
`rescale='rows'`; `cuts_pool` is already 8192 by default and `rescue` already `'prefix'`, which
is why neither appears in the diff. `cuts_pool_window=4` is the one that matters at high rank --
at $K = 128$ on the production rung it is the difference between about 1030 and 480 core-hours,
for a change in the answer at the fourteenth digit. Leave `cfg.threads` at 1: `OMP_NUM_THREADS`
does not reach the LP solver's own thread pool, and at these worker counts letting each worker
spawn one can exhaust the container's process limit. `workers=N` forks a pool only if the work
left after the first chunk projects above two seconds, so a small rung may run serially however
large `N` is; that changes wall-clock, not the core-hours in section 1. To checkpoint a long
run, solve slices with `qstep(ref, groups=idx, repair=False)` and merge them with
`replace(Q=...)` -- note that clears `Q_is_semiorthogonal` and drops any dense `A`, both
correctly; `groups=` requires `repair=False` because per slice the repair loses the step's own
violation accounting, so it must run once after the merge. Two further keywords exist and are
the documented extension points: `q_lower`, a per-group lower bound on the coefficients, and
`solve_fn`, which replaces the solver itself.

$$ \min_{q_\beta} \; s \cdot q_\beta \quad \text{subject to} \quad W q_\beta \;\ge\; \bar A[\beta, :] , \qquad s_c = \sum_F W[F, c] , $$

with $q$ free in sign. The constraint is exactly "this group's approximation dominates its
reference row", and the objective is the group's total output variance, which is what $D$
rewards making small. Because $f$ is strictly increasing, minimizing $D$ over a group's
coefficients is *exactly* minimizing that group's row sum -- so this step is not a heuristic:
given $W$, no better $Q$ exists, and the result does not depend on the precise shape of $f$.
The groups are independent, so this parallelizes perfectly. This step is essentially the entire
cost of the ladder.

**Step 6. Repair.** The optimization is solved in floating point to a finite tolerance, so the result
can violate its own constraint by a small amount, and on a sign-indefinite basis the product
can even go negative where the reference is positive. Since a violation of any size forfeits
the one-sided guarantee, a repair pass lifts the approximation back above the reference --
by rescaling a row where that suffices, and by adding a multiple of a nonnegative column where
it does not (a positive rescale cannot fix a negative entry). At production scale the repair
changes $D$ by less than a tenth of a percent; it is not there to improve the answer but to
make the guarantee exact.

*In code*: normally nothing -- `qstep(..., repair=True)` is the default and the repair runs
inside the step. It becomes a separate call when several repairs are to be compared against
one expensive solve: `qstep(..., repair=False)` returns the raw LP point, and
`approx.repair(ref, cfg=cfg, axis='rows')` applies a repair to it with no re-solve. The
primitives underneath are `repair_rows`, `repair_cols`, `repair_additive` and `fix_nonneg`;
which of them run is chosen by `LpConfig` fields (`additive_first`, `rescale`,
`additive_last`) rather than by an argument. One outcome is worth recognizing: if a negative
product entry survives the repair, the step reports `n_neg_after > 0` in its history record and
deliberately does *not* claim admissibility, whatever the reference said. That is a real
failure, not a rounding artifact, and the fix is a basis with a usable nonnegative column
(step 3), not a looser tolerance.

**Step 7. Score, and verify.** Compute $D$ against $y^{\rm true}$, and separately *measure* that the
approximation dominates everywhere rather than trusting that it must. The two are deliberately
distinct: $D$ is a finite number only if the approximation is admissible, so reporting $D$
without checking admissibility would be assuming the conclusion. Note that $D$ at this geometry
is reproducible only to about 6% -- the per-group problem is degenerate, with many distinct
optima sharing the same objective but distributing their slack differently, and the repair then
charges each row for its own worst channel. Two correct implementations agreeing to 13 digits
on the objective can differ by 6% in $D$, so differences below that are not evidence of
anything.

*In code*, and **this is the step where a swept reference bites**:

```python
adm = approx.measure_admissibility(ref)     # the independent elementwise check
approx = adm.vmap                           # the copy carrying the MEASURED flag
D = approx.get_distance()
```

`get_distance()` raises unless the map is certified admissible and carries $y^{\rm true}$ --
deliberately, since a finite distance reported for a map that underestimates somewhere is the
error the whole construction exists to prevent. A $Q$-step *inherits* the flag from its
reference, and `varmap bf` does not set it (section 2), so against a swept reference the
`adm.vmap` line above is what earns it. Read `adm.admissible`, not `adm.max_r <= 1`: the two
coincide only when both matrices are nonnegative, and `admissible` is the exact elementwise
test. `adm` also carries `max_r`, `max_diff`, `nviol`, `viol_rows`, `nneg_self` and
`worst_rows`. `approx.estimate_distance(groups=idx)` is the subsampled alternative and returns
a `DistanceEstimate` carrying an error bar, never a bare float; pass the same explicit
`groups=` to both arms when comparing two ideas, which is what makes the comparison paired.
Note also that the $Q$-step already put $D$, the full `LpConfig`, `n_lp`, `n_failed`, the
solver `status` histogram and `lp_seconds` into `approx.history[-1]`, so most of what a report
needs is already on the map. Finally `approx.write_asdf(path, provenance={...})` stores the
result -- record the rung, the rank and the `LpConfig` there, because a stored map with no
recipe is not a reproducible one.

**What is deliberately absent: the $W$-step.** One can also hold $Q$ fixed and re-optimize $W$,
one problem per frequency channel, and alternate the two. It is not part of this recipe. At
production scale a single $W$-step costs about thirteen times a $Q$-step while improving $D$ by
around 17%, which is roughly nine times worse per core-hour than simply doubling the rank. The
reason is structural rather than incidental: the $Q$-step's problems have one constraint per
frequency channel and stay small, whereas the $W$-step's have one constraint per *group*, so
they grow with the rung and at the top rung the technique that keeps them tractable stops
working altogether. Worth knowing because at small rungs the $W$-step looks like a good deal,
and that impression does not survive the scale-up.

*In code*, if you want to try it anyway: `approx.wstep(ref, cfg=LpConfig.recommended('w'))`.
Use that preset and **not** `LpConfig.for_wstep()`: the shipped default selects no additive
repair stage, and a purely multiplicative repair cannot fix a product entry that has gone
negative, because a positive rescale cannot change a sign. Note the `'w'` preset is not the
`'q'` preset with a different direction: it sets `additive_last=True` but not `additive_first`,
`rescale='none'` rather than `'rows'`, `clip_rel=0`, `rescue=None`, and it leaves
`cuts_pool_window` at 0.

---

## 4. Reproducing this from scratch

Everything below assumes no previous output files -- only the two definition files in this
directory, a GPU for the one sweep, and the `pirate_frb.varmap` package.

### 4.1 A whole cell, end to end

The two "once" steps are command-line work, and should be, because `varmap bf` applies and
records the config overrides that section 2.1 lists:

```
# ONCE: sweep the base tree straight to the rung-10 grouping (~5.5 GPU-hours at CHORD).
pirate_frb varmap bf config_chord_base_tree.yml detrender_chord.yml \
        -o chord_t0_L6.asdf -L 6 -g 0
```

```python
# ONCE: derive rungs 9..0 by chaining L -> L+1.  Total 86.3 GiB, ~8 minutes.
from pirate_frb.varmap import VarianceMultiMap

src = 'chord_t0_L6.asdf'
for L in range(7, 17):
    with VarianceMultiMap.open_asdf(src) as mm:          # lazy: rung 10 is 86 GiB
        dst = f'chord_t0_L{L}.asdf'
        mm.coarse_grain(L).write_asdf(dst, provenance=dict(rung=16-L, L=L))
    src = dst
```

```python
# PER CELL: one rung, one rank.
from pirate_frb.varmap import VarianceMultiMap, LpConfig

L, K, workers = 11, 128, 32                                    # rung 5
with VarianceMultiMap.open_asdf(f'chord_t0_L{L}.asdf') as mm:
    ref = mm.primary_map(0)                                    # NOT mm[0]
    ref.check_ref_covers_y_true()

    approx = ref.svd(K, method='randomized').canonicalize_signs().seed_onehot(ref)
    approx = approx.qstep(ref, cfg=LpConfig.recommended('q'),
                          workers=workers, progress=True)

    adm = approx.measure_admissibility(ref)                    # earns the flag
    approx = adm.vmap
    print(approx.get_distance(), approx.nscored, adm.admissible, adm.max_r, adm.nviol)

    approx.write_asdf(f'chord_t0_L{L}_K{K}.asdf',
                      provenance=dict(rung=16-L, L=L, K=K))
```

Two notes on the environment. `cfg.threads` is 1 by default and should stay there -- that is
the LP solver's own thread pool, which `OMP_NUM_THREADS` does not reach, and at these worker
counts letting each worker spawn one can exhaust the container's process limit. The BLAS
variables need no attention: the fork pool already sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 in each child.

If you want numbers that go into a table, put them there through
`pirate_frb.varmap.report` (`row_dict`, `frontier`, `format_table`, `save_json`, `load_json`),
which exists so that results taken months apart are comparable. The table in section 1 is
exactly that artifact.

### 4.2 Did it work?

A cell is right if all three hold.

- **`measure_admissibility(ref).admissible` is True, and `nviol` is zero.** This is the hard
  requirement; a cell that fails it is wrong regardless of its $D$. (`nviol` counts violations
  above a relative `viol_tol` of 1e-12, which is not the `LpConfig.viol_tol` of 1e-6 that the
  step's own accounting uses.)
- **`nscored == 5963776`** at every rung. If it is smaller, the reference lost `y_true`
  somewhere in the coarsening chain, and the $D$ is not comparable to the table in section 1.
- **$D$ within about 6% of the value tabulated in section 1.** Do not expect more agreement
  than that, and do not chase it: $D$ at this geometry is reproducible only to a few percent,
  because the per-group problem is degenerate -- many optima share an objective while
  distributing their slack differently, and the repair then charges each row for its own worst
  channel. Two correct implementations agreeing to thirteen digits on the objective have been
  measured to differ by 6% in $D$. A cell landing 20% off is a bug; a cell landing 3% off is a
  cell.

That last tolerance is about *different implementations*, not about noise in your own run. The
randomized SVD's default draw is `np.random.default_rng(0)` -- seeded, despite what its
docstring's advice about passing an explicit `rng` suggests -- so a cell is deterministic on a
fixed numpy, and re-running this recipe against a fixed reference reproduces the table to
better than $2\times10^{-5}$ relative at every rung tried. If your $D$ moves by a percent
between two runs of the same recipe on the same file, something is wrong that the 6% figure
does not excuse.

Also expect **every LP to report `optimal`**, with no failures and no rescues:
`approx.history[-1]['status']` is the histogram and `['n_failed']` the count. That has been
true at every rung and rank measured; a nonzero failure count means something is wrong with the
reference or the basis, not with the solver.

Those three check the *approximation* against the reference. The reference itself is checked by
`check_ref_covers_y_true()` (cheap, structural) and, independently of the whole analytic
chain, by `pirate_frb varmap mc <map.asdf>`: it pushes Gaussian noise through the real
pipeline and reports `eps = MC/map - 1` per output, with `eps > 0` meaning the map
underestimates. It converges slowly -- the per-channel counts need to reach the tens before
`eps_min`/`eps_max` are meaningful -- so run it with a chunk budget you are willing to wait
for, and read it as a check on the sweep rather than as a number to quote.

### 4.3 What it costs

Per-cell CPU is in the table in section 1. The other three resources:

- **GPU.** One sweep, about 5.5 hours on an L40S, independent of $L$: the work is one pass per
  input channel and there are 28160 of them. It is the only GPU step in the whole ladder.
- **Memory.** Peak is set by the reference plus the solver's working set. Rung 10 peaked near
  835 GiB of anonymous memory at $K = 128$ with 60 workers; rungs 0-7 are comfortable on a
  workstation. Scale roughly with $N_\beta$. The sweep itself holds the accumulator, which is
  the output matrix: 86 GiB at $L = 6$, 2.7 GiB at $L = 11$, and `--scratch-dir` moves it to
  disk if that does not fit.
- **Disk.** 86 GiB for rung 10, 86.3 GiB for rungs 9-0 together, and about 300 MB for the whole
  set of low-rank outputs at all four ranks -- the approximations are tiny compared with the
  references they are built from.

The cheapest useful thing to run first is **rung 2 at $K = 16$ off an analytic reference**
(section 2.2): no GPU, no sweep, under a minute, and it exercises the entire chain including
the file format. Then the same cell against a swept reference, which adds the sweep and the
admissibility certification. Build up from there rather than starting at rung 10.
