"""The covering LPs behind the Q-step and the W-step, and everything that controls them.

Both steps of the alternating low-rank fit are FAMILIES OF COVERING LPs sharing one
constraint matrix -- ``min cost . x  s.t.  M x >= b_j``, one right-hand side per subproblem::

    Q-step, per coarse group beta:  min s . q   s.t.  W q >= Abar[beta,:]
                                    K variables, nfreq constraints, s_c = sum_F W[F,c]
    W-step, per frequency channel:  min g . w   s.t.  Qbar w >= Abar[:,F]
                                    K variables, nbeta constraints, g = Qbar^T wbar

so there is ONE computational primitive (solve_covering_lps) and two wrappers (q_step and
w_step) that assemble it. An agent exploring a variation on the Q-step almost never needs to
touch q_step(): it changes a field of LpConfig, or passes its own solver.

THE TWO DIRECTIONS ARE NOT INTERCHANGEABLE, and this is the single most important thing to
know before using this module. Three knobs are measured to want OPPOSITE values:

  knob                        Q direction                   W direction
  --------------------------  ----------------------------  ---------------------------------
  relative constraint floor   1e-8; without it 11% of LPs   0.0; the same idea is measured
                              report 'infeasible' at 25     400x WORSE here
                              subbands
  prefix rescue of failures   improves 180/180, 282/282     improves 0/387
  repair of violations        damage grows with RANK        damage grows with GROUP COUNT

The last row is the general statement and it explains the other two: the repair is one
function applied along axis 0 or axis 1, and the asymmetry is not in the code but in the
shapes -- the row form maxes over nfreq and the column form over nbeta, and those differ by
orders of magnitude and scale differently with the config. So build a config with
LpConfig.for_qstep() or LpConfig.for_wstep(), never by hand, and do not assume a result
measured on one axis transfers to the other.

A warning that shapes the whole interface: THE SOLVER'S REPORTED STATUS IS NOT EVIDENCE OF
ADMISSIBILITY. HiGHS returns points violating their own constraints while reporting success,
at roughly one decade per rank doubling -- a violated fraction of 2e-6 at K=32 rising to 1e-2
at K=256, with one measured case off by a factor 18.7. So the steps report MEASURED
violations (violation_stats()), not statuses, and a repair always runs.

Nothing here does any file I/O, and nothing here knows about VarianceMap except the two
building blocks at the bottom, which are conveniences for assembling the arrays. The entry
points take the reference matrix as an in-memory array and never learn where it came from.

The problem these two steps solve, the distance function they minimize, and the campaign the
measured numbers throughout this module come from are written up in notes/variance_map.tex.
"""

import dataclasses
import os
import time
import warnings

import numpy as np

# f and fprime are re-exported, not just used: several things built on top of the steps
# need the distance kernel and nothing should re-derive it. Do not 'clean up' the
# apparently-unused import of f.
from .distance import YTRUE_FLOOR, f, fprime


####################################   the configuration   ####################################


# The repair triples that have been measured, as (additive_first, additive_last, rescale).
# Keyed by the short name a results table wants; see LpConfig.repair_label.
_REPAIR_LABELS = {
    (True, True): 'additive_first',
    (False, True): 'shipped',
}


@dataclasses.dataclass(frozen=True)
class LpConfig:
    """Everything that controls how a family of covering LPs is solved.

    Frozen and serializable, so that a produced VarianceMap can record exactly how it was made
    by stashing one of these in its history.

    THE DEFAULTS ARE THE RESEARCH CODE'S, NOT THE BEST VALUES KNOWN, AND THAT IS DELIBERATE.
    A default-constructed LpConfig() reproduces the module globals of the LP this was ported
    from, exactly, so that the port can be checked BIT FOR BIT against the code whose numbers
    are already published. Choosing "better" defaults destroys that property in the most
    expensive way: a difference in output can no longer be told apart from a bug. The
    measured-better values are not lost -- they are recommended(), which is one line to adopt.

    Be clear about what it costs: this configuration is the WORST one measured. Constraint
    generation off is 5.6x-32x slower, and the additive repair off costs up to 2.5x in D.

    The field list is a TRANSCRIPTION of the research module's globals, not a curated
    selection, because a config missing even one knob cannot reproduce the old behaviour and
    the omission then shows up as an unexplainable numerical difference rather than as a
    missing field. Three families are per-direction there and are per-direction here too
    (clip_rel, rescue, and the repair triple), which is why for_qstep() / for_wstep() exist
    and why a single config with a direction flag would not do.

    Four of that module's globals are deliberately NOT fields, because they belong to code
    outside the solve path: COL_NORM (a conditioning rescale of W, which belongs to whatever
    writes the alternation schedule), and DIAG_EXACT_ELEMENTS / DIAG_SAMPLE_ELEMENTS /
    DIAG_POST_HARDEN (reporting). Nothing here reads them, so a field would be inert.

    Three fields name capabilities the research code does not have: ``equilibrate=False``,
    ``slack > 0`` and ``nnz_cap``. They are named rather than omitted so that the vocabulary
    is complete, and they RAISE rather than guessing at an implementation, because there is
    nothing to check such an implementation against.
    """

    # ---- the constraint set ----

    nonneg: bool = False
    #   Impose x >= 0. Admissibility constrains the PRODUCT M x >= b, not the coefficients, so
    #   this was never implied by the problem, and dropping it is worth 1.4x to 3.9x in D at
    #   fixed rank. It also unlocks a signed (SVD) M, since the problem is then exactly
    #   invariant under M[:,c] -> -M[:,c], x_c -> -x_c. Keeping it on top of a signed M costs
    #   8.8x, so this flag and the choice of M are coupled.
    #
    #   NOTE this is the one default that is NOT the research module's: there it is a function
    #   ARGUMENT defaulting True, while every campaign-2 cell passes False. A bit-identity
    #   comparison must therefore set it explicitly to whatever the cell being reproduced used.

    equilibrate: bool = True
    #   Rewrite rows with b_i > 0 as (M/b) x >= 1. The solver's feasibility tolerance is
    #   ABSOLUTE while the tolerance we need is RELATIVE: HiGHS's default ~1e-7 is meaningless
    #   on a matrix element of 1e-12, and a variance-map row spans ~1e14. Not optional --
    #   False raises.

    clip_rel: float = 1.0e-8
    #   Q direction only; the W direction ships 0.0 (for_wstep). Rows whose right-hand side is
    #   below clip_rel * (row max) are treated as b == 0. A detrender replaces a subband's
    #   exact zeros with values ~1e-11 of the row max, the equilibrated matrix then spans
    #   ~1e22, and 11% of LPs come back 'infeasible' at 25 subbands. This floor removes all of
    #   them and improves D. 1e-10 is too small (it makes a provably feasible LP report
    #   infeasible, and can invert a rank ladder); 1e-6 is too large (loses 1.22x).
    #
    #   IN THE W DIRECTION THE SAME IDEA IS MEASURED 400x WORSE -- median raw violation 1.00
    #   -> 102.6, worst 7.19 -> 2880 -- and the reason is structural rather than tuning. The
    #   Q-side floor is only safe because the repair afterwards measures against the UNCLIPPED
    #   map on the same axis it clipped; the W-step's repair is on the other axis, so a
    #   clipped W-step constraint is never re-checked and the error survives. Leave it at 0.

    zero_rhs_margin: float = 1.0e-7
    #   Every channel must be constrained, including the ones with a zero right-hand side. A
    #   row with b == 0 is NOT automatically satisfied once x is sign-free: the product can go
    #   negative where the reference is zero, and dropping such rows makes the LP genuinely
    #   unbounded (measured: 320 of 400 sampled LPs). So rows are classified, never dropped,
    #   and there are exactly TWO classes:
    #       b > 0    equilibrate to right-hand side 1
    #       b <= 0   normalize the row to unit max-abs and give it THIS margin rather than 0,
    #                because HiGHS's primal tolerance is absolute and a constraint '>= 0' is
    #                "satisfied" at -1e-7, which is a negative variance
    #   The margin must exceed primal_tol. Its cost is zero_rhs_margin per such channel added
    #   to the row sum, ~1e-5 relative at the defaults, and it is re-measured afterwards
    #   rather than trusted.
    #
    #   NOTE WHAT THE SECOND CLASS DOES TO A NEGATIVE b, because it is not the obvious
    #   treatment and it has a price. A negative right-hand side arises only from a FACTORED
    #   reference (a streamed max-envelope is nonnegative by construction). Such a row is not
    #   divided by its own b -- that would flip the inequality -- and it is not kept at its
    #   true, negative right-hand side either. It is asked for a POSITIVE product, which is
    #   STRICTER than b and therefore safe, and which is also what stops the product going
    #   negative in exactly the channels an additive repair would otherwise have to lift.
    #
    #   The price is sharp: a dictionary with NO nonnegative column cannot make the product
    #   positive everywhere, so against a signed reference every subproblem comes back
    #   infeasible. Sign-canonicalizing the basis, or pinning a nonnegative column, is what
    #   makes a factored reference usable at all.

    primal_tol: float = 1.0e-9
    #   HiGHS primal and dual feasibility tolerance.

    # ---- constraint generation ----

    cuts: bool = False
    #   Solve on a working set, evaluate ALL the rows, add the violated ones, repeat, and stop
    #   when nothing outside the set is violated by more than cuts_tol. Exact up to that
    #   tolerance: the exit test is against every constraint, so the working optimum is
    #   feasible for the full LP to within cuts_tol and, being a relaxation optimum, optimal
    #   for it. It needs BOTH ingredients below. recommended() turns it on.
    #
    #   WHAT IT IS WORTH DEPENDS STRONGLY ON THE SHAPE, and it can be a LOSS. Measured here at
    #   nbeta ~ 10^4, one LP per group, against the same solve with cuts off:
    #
    #       nfreq     K    speed-up   pooled working rows
    #        3200    16      1.6x       385 of  3200
    #        6400    16      1.9x       550 of  6400
    #       28160    16      2.3x      1076 of 28160
    #       16384    16      4.0x       212 of 16384    (an UNSUBBANDED map -- see below)
    #        3200    32      1.3x       652 of  3200
    #        3200   128      0.8x      1536 of  3200    <-- A NET LOSS
    #
    #   Two things drive it. RANK: the initial working set is cuts_init*K rows, so at K = 128
    #   it is already 1024 of 3200 before a single cut is added, and there is nothing left to
    #   save. SUBBANDING: a map with 25 subbands spreads its binding channels, so the shared
    #   pool accumulates 1076 rows where an unsubbanded map needs 212 -- which is why the
    #   16384-channel row above beats the 28160-channel one.
    #
    #   cuts_min_rows therefore guards the wrong quantity: it gates on nfreq alone, and what
    #   decides this is nfreq RELATIVE TO K. Measure before turning it on at high rank.
    cuts_min_rows: int = 2048       # below this many constraints, do not bother: 2.3x LOSS
    cuts_init: float = 8.0          # initial working set = max(cuts_init*K, 64) rows
    cuts_maxadd: float = 4.0        # add at most cuts_maxadd*K rows per round
    cuts_rounds: int = 200
    cuts_tol: float = 1.0e-9
    #   Exit test, RELATIVE (the rows are equilibrated to right-hand side 1). Must not be
    #   below HiGHS's own primal tolerance, or rows already in the working set would be
    #   re-added forever. Raising it stops the loop while the residual violation is small and
    #   lets the repair (which runs anyway) absorb it.
    #
    #   MEASURED, at nbeta = 12800, nfreq = 3200, K = 16, against the same solve with cuts off:
    #
    #       cuts_tol    D differs by    worst group's objective    wall time   rounds
    #         1e-6        1.7e-4        8.2e-2 WORSE                 17.7 s     2.0
    #         1e-9        2.1e-9        1.7e-9 worse                 17.5 s     2.1
    #        1e-12        2.7e-13       1.7e-9 worse                 17.8 s     2.1
    #
    #   So TIGHTENING IT IS FREE -- same wall time, same round count, and D agrees to 13
    #   digits instead of 9. The trade this knob was built for is the other direction, and
    #   that direction is expensive: at 1e-6 the loop leaves violations for the repair, and
    #   the repair charges whole groups for them.
    cuts_agg: bool = True
    #   The aggregate row: sum of the constraints NOT in the working set. It is a nonnegative
    #   combination of the omitted constraints, so it cuts off no feasible point, and it
    #   bounds the objective below by the sum of ALL the right-hand sides for ANY working set.
    #   WITHOUT IT 26 of 32 relaxations are UNBOUNDED -- dropping rows from a free-sign LP
    #   creates a recession direction, which is also why naive constraint SAMPLING (drop rows
    #   and never add them back) is structurally dead at 896/896 unbounded.
    cuts_nagg: int = 128
    #   How many aggregate rows. 1 is the single complement sum; G > 1 partitions the channels
    #   into G contiguous bands and aggregates within each, which leaves validity and the
    #   boundedness identity untouched (the G rows sum to the single row) but makes the
    #   relaxation far tighter. That decides the cost, because the working set is rebuilt from
    #   scratch every round, so the ROUND COUNT is the whole game.
    cuts_warm: bool = False         # carry the working set from the previous subproblem
    cuts_act_tol: float = 1.0e-7    # a row is "binding" when its equilibrated residual is below this
    cuts_warm_seed: str = 'auto'    # 'auto' adds the heuristic seed only while the warm set is small
    cuts_pool: int = 8192
    #   THE CHANNEL POOL -- the measurement that turned constraint generation from 1.2x into
    #   10x. At the optimum exactly K constraints bind (32 of 28160 at K=32), so the LP the
    #   step actually needs is tiny; the entire cost is the 20-30 rounds it takes to FIND
    #   those K rows from a cold start. All the subproblems share one M and one objective --
    #   only b changes -- so the binding rows are drawn over and over from a small pool. This
    #   caps the pool size in rows; 0 turns it off. Per-subproblem cut generation WITHOUT the
    #   pool is a 0.67-0.99x LOSS at K >= 32: the pool is the whole win.
    cuts_pool_sample: int = 0       # subproblems pre-solved to warm the pool before the main pass
    cuts_pool_window: int = 0
    #   0 = the pool is the union of every binding row ever seen, which does NOT saturate (32
    #   -> 1753 rows over 512 contiguous groups at K=32), so the working set leaks and the
    #   speed-up decays 19x -> 14.5x. A window of M keeps only the last M subproblems' binding
    #   rows, which tracks the drift of Abar along the DM axis instead of accumulating it: at
    #   M = 4 the pool sits at ~1.2*K rows and the speed-up is FLAT.

    # ---- what to do about the residual violations ----

    additive_first: bool = False
    additive_last: bool = False
    rescale: str = 'auto'           # 'auto' | 'rows' | 'cols' | 'none'
    #   THE REPAIR IS A THREE-STAGE PIPELINE, NOT A CHOICE BETWEEN TWO REPAIRS, and these are
    #   its three independent decisions:
    #
    #       1. additive_first   add a multiple of a nonnegative column of W
    #       2. rescale          multiplicative: scale each row (Q direction) or column (W
    #                           direction) by its own worst violation ratio
    #       3. additive_last    the same additive lift again
    #
    #   'auto' picks the axis the step runs along: rows for the Q-step, cols for the W-step.
    #   Naming both explicitly is what lets a W-step use the row form, which nothing has ruled
    #   out.
    #
    #   WHY BOTH ADDITIVE STAGES. They do different jobs, and stage 3 is not stage 1 run
    #   twice. Stage 1 absorbs the SOLVER's violation additively so that the multiplicative
    #   pass has almost nothing left to charge for -- worth up to 1.59x at high rank, and it
    #   removes an ordering that is provably impossible (a nested dictionary scoring worse at
    #   K=384 than at K=256). Stage 3 is the safety net for what a positive scale structurally
    #   CANNOT reach: a negative product element, and entries below their own float64
    #   evaluation noise. After stages 1 and 2 the product already dominates, so stage 3 is
    #   nearly a no-op -- but it is the stage that makes admissibility exact rather than
    #   nearly exact, and the two report under separate keys so a change to one stays visible.
    #
    #   WHY THE TWO KINDS DIFFER. The multiplicative repair is exact, local, and cannot break
    #   another subproblem -- but D is precisely a penalty on scaling things up, so it pays in
    #   the figure of merit. The additive lift raises the product in every channel at once for
    #   a relative change of ~1e-9 rather than percents. It requires a nonnegative column of W
    #   to exist.
    #
    #   THE ADDITIVE STAGE IS A DIFFERENT FUNCTION IN THE TWO DIRECTIONS, and this is a
    #   property of the code being reproduced rather than a choice made here. Along 'rows' it
    #   is fix_nonneg(), which lifts every element to max(Abar, its own rounding noise) using
    #   the cheapest nonnegative column PER CHANNEL. Along 'cols' it is repair_additive(),
    #   which lifts to Abar using the cheapest nonnegative column PER GROUP and falls back to
    #   a row scale for groups no column reaches. The default configs never run either, so
    #   only recommended('w') is affected; call the two by name if you care which you get.
    #
    #   The measured arms, as triples:
    #       (False, False, 'auto')  THE DEFAULT: the multiplicative stage only
    #       (False, True,  'auto')  multiplicative, then the lift
    #       (True,  True,  'auto')  what recommended() sets
    #       (False, False, 'none')  the raw LP point, i.e. q_step(repair=False)
    #   and (True, False, ...), which the research code cannot express -- there,
    #   additive_first without additive_last is rejected with a warning rather than dropping
    #   the trailing lift. It is reachable here, and has never been measured.

    repair_margin: float = 1.0e-12  # relative margin the multiplicative stage aims for
    repair_growth: float = 1000.0   # the margin ladder's growth factor per pass
    repair_max_iter: int = 10
    #   repair_max_iter caps the multiplicative stage, which loops with a growing margin until
    #   the RECOMPUTED product really dominates. The loop is not belt-and-braces: the repair
    #   is exact in exact arithmetic but not in floating point once the factors are signed,
    #   because each term q_c W_Fc rounds separately and the sum then cancels -- measured
    #   per-entry cancellation reaches ~1e11 in channels where A is ~1e-12 of its row maximum,
    #   which swamps a 1e-12 margin by seven orders of magnitude and leaves max_r = 1 + 1e-5,
    #   i.e. D = infinity. Exhausting the cap is an ERROR.

    repair_bisect: bool = True
    backoff_iters: int = 24
    backoff_trigger: float = 1.0 + 1.0e-9   # rows scaled by less than this are already tight
    #   The multiplicative stage grows its margin by repair_growth per pass, so a row needing
    #   a margin just above 1e-3 is handed 1e0 -- a factor two of pure overshoot, which is
    #   what a max_r of ~0.5000 in four measured cells turned out to be. This bisects PER ROW
    #   for the smallest margin that works, which is legitimate because rows scale
    #   independently and the admissibility test is per row. Worth 2.01x on the A/B pair that
    #   isolated it, and it touches only the handful of rows that needed a large margin, so
    #   the extra matmuls are negligible.

    single_shot_repair: bool = False
    #   A SEPARATE knob, not repair_max_iter=1: it applies one scale at the given margin and
    #   returns WITHOUT recomputing the product, whereas max_iter=1 verifies and raises. It
    #   exists so the cost of the iterative repair can be measured rather than asserted, and
    #   it produces knowingly inadmissible output. Carried because everything is carried; do
    #   not use it.

    noise_kappa: float = 64.0
    #   fix_nonneg lifts an element to max(Abar, noise_kappa * eps * sum_c |q_c W_Fc|) rather
    #   than to max(Abar, 0). Pushing to exactly 0 would leave the entry at the mercy of the
    #   next positive rescale's rounding, which is how a repaired product goes negative again.
    noise_cost_rel: float = 1.0e-6
    #   Cost cap on the OPTIONAL half of that lift, and load-bearing on a badly-fitting
    #   dictionary. Lifting an entry to Abar is never optional -- that is admissibility.
    #   Lifting one that is already admissible up to its own rounding noise is optional, and
    #   can be arbitrarily expensive when the cheapest usable column is nearly zero in that
    #   channel: measured, 21694 noise-floor lifts cost a factor ~2 on the whole
    #   approximation. So a noise-floor lift is skipped when it would cost more than this
    #   fraction of that group's own row sum.

    viol_tol: float = 1.0e-6
    #   Relative tolerance for COUNTING a violated constraint in violation_stats(). Purely a
    #   diagnostic threshold; the repair itself uses no tolerance.

    # ---- rescuing failed subproblems ----

    rescue: str = 'prefix'          # 'prefix' or None; the W direction ships None (for_wstep)
    rescue_ladder: tuple = (64, 32, 16, 8)
    #   When a subproblem fails, re-solve it on a PREFIX of the same M -- fewer columns, same
    #   admissibility argument, rank preserved -- and keep the result only if it improves. It
    #   matters because one failure per ~450 groups costs more D than an entire doubling of
    #   the rank. Measured to improve 180/180 and 282/282 in the Q direction and 0/387 in the
    #   W direction, so it is off there.

    # ---- the W direction's guard and diagnostics ----

    w_guard: bool = False
    #   Restore the majorize-minimize guarantee CHANNEL BY CHANNEL. The product is
    #   column-separable, so each channel can be accepted or rejected on its own: scale the
    #   new row up by its own raw violation ratio, keep it only if its objective is then still
    #   at or below the incumbent's, and revert outright if the product went non-positive
    #   anywhere in that channel (a sign the multiplicative repair is structurally blind to).
    #   The incumbent W0 is feasible with the incumbent objective, so the accepted W satisfies
    #   both properties the majorization needs, exactly.
    w_diag: bool = False
    #   The raw per-channel violation diagnostics. Off because they cost one extra blocked
    #   pass over the product; when w_guard is on the same numbers come out of the guard's own
    #   pass and are reported regardless.
    w_step_transpose: bool = False
    #   Materialize Abar.T so each channel's right-hand side is a contiguous read. Values are
    #   identical either way; the copy is 172 GiB at CHORD's L = 4, and worth having only when
    #   Abar is small and the LPs are fast.
    w_fail_warn: bool = True
    #   Print when W-step subproblems fail. A failed W-step LP silently keeps the previous row
    #   of W, and there is no rescue on that side; printing is not a fix, but it is the
    #   difference between a traceable number and an untraceable one.

    stash_raw: bool = False
    #   Keep the pre-repair LP point in the step's info dict, under 'Q_raw'. That point is
    #   what the solver actually returned -- the one it called 'optimal' and that the repair
    #   then silently rescales -- so it is what makes a repair-order change testable without
    #   re-solving. Largely superseded by repair=False, which returns it directly.

    # ---- named but not implemented ----

    slack: float = 0.0
    #   Accept a solution violating its constraints by a relative 'slack' and repair by
    #   rescaling by 1/(1-slack), trading a slightly suboptimal distance for speed -- useful
    #   in the early iterations of a schedule. 0 is exact and is the only implemented value.
    nnz_cap: int = None
    #   After solving, restrict to the best admissible subset of this size of the LP's own
    #   support and re-solve on that support. The LP is not asked to be sparse and returns 2-3
    #   nonzeros anyway; rank, not apply cost, is the agreed figure of merit, so this is off.
    #   Capping at 2 costs 14-34% in D and is worth it only when nbeta >~ K*nfreq/4.

    # ---- numerics and execution ----

    threads: int = 1
    #   HiGHS threads. NOT OPTIONAL, and not a performance knob: HiGHS sizes its task executor
    #   from hardware_concurrency(), i.e. ~63 worker threads PER PROCESS, in code that is not
    #   exception-safe, and those threads count against the container's shared pid limit -- so
    #   a pool of W workers asks for ~64*W tasks and the process dies with 'terminate called
    #   without an active exception'. OMP_NUM_THREADS does not control it; HiGHS has its own
    #   pool. This has cost one run outright.
    block_bytes: int = 1 << 28
    #   Row-block size for the repair and violation passes, as a byte budget rather than a row
    #   count, so a small map is still processed in ONE block and a large one is bounded. An
    #   unblocked dense (nbeta, nfreq) product with a same-sized temporary, formed up to ten
    #   times, is ~900 GiB of transient at CHORD's L = 4; blocked it is flat at ~0.4 GiB and
    #   1.4x faster.
    block_min_rows: int = 8
    #   Floor on the block size AND on the ragged tail. Below 8 rows numpy uses dgemv, which
    #   sums over K in a different order, so a short tail breaks the bit-identity that
    #   blocking otherwise has -- see blocking_is_exact() for the other condition.
    chunk_timeout: float = 2400.0
    #   Seconds without a single finished chunk before the worker pool is declared hung and
    #   the work is re-run serially. The guard is what makes a pid-exhaustion death
    #   recoverable; serial is 10-50x slower but always finishes.

    def __post_init__(self):
        # A file round trip turns the tuple into a list (asdf writes plain data), so a config
        # read back out of a map's history would otherwise not compare equal to the one that
        # produced it -- and a list in a frozen dataclass is a mutable field in an object that
        # claims to be immutable. Coerce, so that LpConfig(**record) reconstructs the config a
        # step actually ran under.
        if not isinstance(self.rescue_ladder, tuple):
            object.__setattr__(self, 'rescue_ladder',
                               tuple(int(k) for k in self.rescue_ladder))

    @property
    def repair_label(self):
        """The repair triple as a short name, for a results table.

        Three fields are a space to configure rather than a closed list to edit, but a table
        still wants one string per row -- so the name is DERIVED from the fields, and an
        unmeasured combination gets a name too instead of being unreachable.
        """
        key = (bool(self.additive_first), bool(self.additive_last))
        if key in _REPAIR_LABELS:
            return _REPAIR_LABELS[key]
        if not self.additive_first and self.rescale == 'none':
            return 'raw'
        return f'add{int(self.additive_first)}{int(self.additive_last)}_{self.rescale}'

    @classmethod
    def for_qstep(cls, **overrides):
        """The Q-step's settings: the research module's globals as the Q-step reads them.

        clip_rel = 1e-8, rescue = 'prefix', rescale = 'rows' (the multiplicative row repair,
        which that code runs unconditionally in the Q-step). NOT the best values known -- see
        recommended().
        """
        return cls(**dict(dict(clip_rel=1.0e-8, rescue='prefix', rescale='rows'), **overrides))

    @classmethod
    def for_wstep(cls, **overrides):
        """The W-step's settings, which are genuinely different settings and not the Q-step's
        with a flag: clip_rel = 0.0, rescue = None, and the repair triple
        (False, False, 'cols').

        The last is the transcription of a four-way choose-one knob whose shipped value is
        'cols'; the other three values are 'rows' -> (False, False, 'rows'), 'additive' ->
        (False, True, 'none'), 'none' -> (False, False, 'none').
        """
        return cls(**dict(dict(clip_rel=0.0, rescue=None, rescale='cols',
                               additive_first=False, additive_last=False), **overrides))

    @classmethod
    def recommended(cls, direction, **overrides):
        """The best values measured, as a preset rather than a default.

        Everything the field commentary says is worth something is turned on: constraint
        generation with a pool, and both additive repair stages (up to 2.5x). 'direction' is
        'q' or 'w', because several of these differ between the two.

        MEASURE THE CUTS HALF BEFORE ADOPTING IT. Its benefit is strongly shape-dependent and
        it is a NET LOSS at high rank on a narrow-band map -- 0.8x at nfreq = 3200, K = 128,
        which is one of the campaign's own published geometries. See the LpConfig.cuts
        commentary for the measured table. The additive half needs no such caution: it is
        what every published cell already ran.

        THE W DIRECTION IS THE ONE TO BE CAREFUL WITH. Its shipped repair is a known-wrong
        default that should be the additive one, so the preset follows that -- but 'additive'
        named a choose-one value, and the three-field encoding admits a pipeline that
        direction has never run (additive, rescale, additive). The preset uses the literal
        reading. Comparing it against the pipeline is a cheap experiment -- one W-step with
        repair off, then each candidate triple applied to the same stashed point, no re-solve
        -- and worth doing before treating either as settled.
        """
        direction = str(direction).lower()
        if direction not in ('q', 'w'):
            raise RuntimeError(f"LpConfig.recommended: direction must be 'q' or 'w', got"
                               f" {direction!r}")

        common = dict(cuts=True, cuts_pool=8192, cuts_agg=True, cuts_min_rows=1500,
                      cuts_rounds=200)
        if direction == 'q':
            base = cls.for_qstep(**common, additive_first=True, additive_last=True)
        else:
            base = cls.for_wstep(**common, additive_last=True, rescale='none')
        return dataclasses.replace(base, **overrides) if overrides else base

    def resolved_rescale(self, axis):
        """cfg.rescale with 'auto' resolved against 'axis' ('rows' or 'cols')."""
        if axis not in ('rows', 'cols'):
            raise RuntimeError(f"LpConfig: axis must be 'rows' or 'cols', got {axis!r}")
        return axis if (self.rescale == 'auto') else self.rescale

    def _check_implemented(self):
        """Raise on the fields that name a capability nothing here implements."""
        if not self.equilibrate:
            raise RuntimeError(
                'LpConfig(equilibrate=False): there is no unequilibrated solve path. The'
                " solver's feasibility tolerance is absolute while the tolerance a variance"
                ' map needs is relative, so this is named to make the property visible, not'
                ' to be turned off.')
        if self.slack != 0.0:
            raise RuntimeError(f'LpConfig(slack={self.slack!r}): not implemented. Nothing'
                               ' measures it, so there is nothing to check an implementation'
                               ' against; slack=0 is the exact solve.')
        if self.nnz_cap is not None:
            raise RuntimeError(f'LpConfig(nnz_cap={self.nnz_cap!r}): not implemented, for the'
                               ' same reason as slack. Rank, not apply cost, is the figure of'
                               ' merit.')


_DEFAULT_CFG = LpConfig()


def _cfg(cfg):
    """A config for the building blocks, which take one optionally."""
    return _DEFAULT_CFG if (cfg is None) else cfg


####################################   row blocking   ####################################
#
# Every routine below that touches the (nbeta, nfreq) product streams over ROW BLOCKS instead
# of forming it. At CHORD's L = 4 (nbeta = 819200, nfreq = 28160) one such array is 172 GiB
# and the transient is ~2.5x that, inside a repair loop that runs up to ten times; that is
# what fails first at production scale, not the LP.
#
# The blocking is TRANSPARENTLY IDENTICAL rather than flagged, under two measured conditions:
#
#   (a) the block has at least cfg.block_min_rows = 8 rows. A 1-row block is a dgemv, not a
#       dgemm, and sums over K in a different order. This bounds the ragged TAIL as well as
#       the block size -- a splitter that bounds only the block size ends a 1500-row pass with
#       a 4-row block, and that is exactly the kind of latent bug a regression suite misses
#       when every test cell happens to divide exactly.
#   (b) nfreq is a multiple of 8. The rule is sharp: at 296, 304, 392, 400, 256, 4096, 16384
#       and 28160 every block size from 8 up is bit-identical; at 297...303, 255, 257, 4095,
#       4097, 28159 and 28161 none is. OpenBLAS handles the ragged tail of the N axis with a
#       different kernel whose accumulation order depends on the M-blocking.
#
# Maxima and counts are selections, so a blocked reduction returns the identical float
# (np.fmax combines block maxima because, like np.nanmax, it ignores NaN). The one sum that
# could reorder -- fix_nonneg's reported dy -- is accumulated as a per-row vector and summed
# once at the end, so even that is bit-identical.


def blocking_is_exact(nfreq):
    """True iff a row-blocked pass over the (nbeta, nfreq) product is bit-identical to the
    unblocked one here, which holds iff nfreq is a multiple of 8.

    Report it rather than assuming it: a bit-identity harness needs to know which of its
    comparisons are exact. Every geometry in the existing map library qualifies (400, 1600,
    3200, 4096, 16384, 28160); 2049 would not.
    """
    return int(nfreq) % 8 == 0


_BLOCK_WARNED = set()


def _block_rows(nrow, ncol, cfg, arrays=1.5):
    """Rows per block so that 'arrays' float64 (nrow, ncol) temporaries fit in cfg.block_bytes.

    Returns nrow -- i.e. one block, the unblocked call -- whenever it fits, so that anything
    small enough to have run before blocking existed is untouched.
    """
    per = max(1.0, float(arrays) * 8.0 * max(int(ncol), 1))
    bs = int(max(cfg.block_min_rows, min(int(nrow), int(int(cfg.block_bytes) // per))))
    if (bs < int(nrow)) and (not blocking_is_exact(ncol)) and (ncol not in _BLOCK_WARNED):
        _BLOCK_WARNED.add(int(ncol))
        print(f'  varmap.lp: blocking a product with nfreq = {int(ncol)}, which is not a'
              f' multiple of 8; results will differ from the unblocked form in the last ulp.'
              f' The repair re-establishes admissibility, so this is safe, but it is not'
              f' bit-reproducible.', flush=True)
    return bs


def _blocks(nrow, block, min_rows):
    """Row blocks of 'block' rows, EXCEPT that a ragged tail shorter than min_rows is merged
    into the block before it (so the last block may be up to block + min_rows - 1)."""
    nrow, block = int(nrow), int(block)
    if nrow <= 0:
        return
    starts = list(range(0, nrow, block))
    if (len(starts) > 1) and ((nrow - starts[-1]) < int(min_rows)):
        starts.pop()
    for i, s0 in enumerate(starts):
        yield s0, (starts[i+1] if (i+1) < len(starts) else nrow)


def _prod(Qblock, mid, W):
    """One row block of the product ``Q mid W^T``.

    'mid' is None for the identity, and then this is exactly ``Qblock @ W.T`` -- which is the
    expression whose blocking properties are described above, so the None path is the one to
    stay on when bit-identity matters.
    """
    return (Qblock @ W.T) if (mid is None) else ((Qblock @ mid) @ W.T)


####################################   measuring the violation   ####################################


def _ratios_blocked(Abar, Q, W, mid, cfg, *, want_rows=True, want_cols=True, block_rows=None,
                    stats=False):
    """Blocked ``max over the other axis of Abar/(Q mid W^T)``, elementwise-safe.

    Returns (row_ratios or None, col_ratios or None, stats or None). Where Abar == 0 the ratio
    is 0, since 0 over anything is not an underestimate -- which is exactly why the
    multiplicative repair alone cannot see a NEGATIVE product entry, and why check_nonneg()
    exists.

    Only the reductions that are asked for are computed, which by itself removes one
    full-sized temporary.
    """
    Abar = np.asarray(Abar)
    nrow, ncol = Abar.shape
    # 1.5 arrays: the product (which the ratio overwrites in place) plus the boolean mask
    # (1/8) is ~1.125; the rest leaves room for the BLAS packing buffers.
    bs = _block_rows(nrow, ncol, cfg, arrays=1.5) if (block_rows is None) else int(block_rows)

    rr = np.empty(nrow) if want_rows else None
    rc = np.full(ncol, -np.inf) if want_cols else None
    n_pos = n_viol = n_rows_viol = 0
    rmax = 0.0
    tol = float(cfg.viol_tol)

    for s, e in _blocks(nrow, bs, cfg.block_min_rows):
        a = Abar[s:e]
        P = _prod(Q[s:e], mid, W)
        with np.errstate(divide='ignore', invalid='ignore'):
            pos = a > 0
            np.divide(a, P, out=P)                  # (Abar/P) elementwise, in place
            np.copyto(P, 0.0, where=~pos)           # == np.where(Abar > 0, Abar/P, 0.0)
            if want_rows:
                rr[s:e] = np.nanmax(P, axis=1)
            if want_cols:
                np.fmax(rc, np.nanmax(P, axis=0), out=rc)
            if stats:
                n_pos += int(pos.sum())
                # ratio > 1: the point is BELOW Abar there. ratio < 0: the product is negative
                # where Abar is positive, which is the same violation with a worse sign (the
                # ratio cannot be negative where Abar <= 0, it was zeroed above).
                n_viol += int(((P > 1.0 + tol) | (P < 0.0)).sum())
                m = float(np.nanmax(P)) if P.size else 0.0
                rmax = m if (m > rmax) else rmax
                if want_rows:
                    n_rows_viol += int((rr[s:e] > 1.0 + tol).sum())

    st = None
    if stats:
        st = dict(n_pos=int(n_pos), n_viol=int(n_viol),
                  frac_viol=float(n_viol) / max(1, n_pos), max_ratio=float(rmax),
                  tol=float(tol))
        # frac_viol is a fraction of (group, channel) CONSTRAINTS, so it falls like 1/nfreq at
        # fixed K while the number of groups the repair charges does not. On a 400-channel map
        # frac_viol = 1e-3 already means 14-20% of groups are being rescaled; at CHORD's 28160
        # channels the same situation reports 1e-5 and looks clean. The row figures are the
        # ones that map onto D.
        if want_rows:
            st.update(n_rows_viol=int(n_rows_viol), n_rows=int(nrow),
                      frac_rows_viol=float(n_rows_viol) / max(1, nrow))
    return rr, rc, st


def violation_stats(Q, W, mid, Abar, cfg=None, *, block_rows=None):
    """How badly ``Q mid W^T`` fails to dominate Abar: a dict with the count and fraction of
    violated positive-rhs constraints, the number of rows involved, and the worst ratio.

    This is the answer to "is the point the solver returned actually admissible", which its
    reported status is not evidence of. The steps get these figures for free from the first
    pass of the repair, so calling this separately is for a point that is not being repaired.
    """
    cfg = _cfg(cfg)
    _, _, st = _ratios_blocked(Abar, Q, W, mid, cfg, want_rows=True, want_cols=False,
                               block_rows=block_rows, stats=True)
    return st


def check_nonneg(Q, W, mid=None, cfg=None):
    """(n_negative, most_negative_entry) of the product ``Q mid W^T``.

    Checked explicitly after every step because the multiplicative repair CANNOT fix a
    negative entry -- it multiplies by a positive scalar, which makes a negative entry more
    negative -- and the ratio it works from cannot even see one, since Abar is 0 there.
    """
    cfg = _cfg(cfg)
    Q = np.asarray(Q)
    nrow = Q.shape[0]
    bs = _block_rows(nrow, W.shape[0], cfg, arrays=1.25)
    n, mn = 0, np.inf
    for s, e in _blocks(nrow, bs, cfg.block_min_rows):
        P = _prod(Q[s:e], mid, W)
        n += int((P < 0).sum())
        v = float(P.min())
        mn = v if (v < mn) else mn
    return int(n), float(mn if np.isfinite(mn) else 0.0)


####################################   the multiplicative repair   ####################################


def _row_ratio(Abar, Q, W, mid, cfg, axis, rows):
    """max over the other axis of Abar/(Q mid W^T), restricted to 'rows' of the repaired factor.

    Blocked over the long axis, which for axis=0 is 'rows' itself (usually a short subset of
    the groups) and for axis=1 is ALL the groups against a subset of channels.
    """
    if axis == 0:
        rows = np.asarray(rows)
        out = np.empty(rows.size)
        bs = _block_rows(rows.size, W.shape[0], cfg, arrays=1.5)
        for s, e in _blocks(rows.size, bs, cfg.block_min_rows):
            idx = rows[s:e]
            a = Abar[idx]
            P = _prod(Q[idx], mid, W)
            with np.errstate(divide='ignore', invalid='ignore'):
                np.divide(a, P, out=P)
                np.copyto(P, 0.0, where=~(a > 0))
                out[s:e] = np.nanmax(P, axis=1)
        return out

    Wr = np.ascontiguousarray(W[rows])
    out = np.full(Wr.shape[0], -np.inf)
    bs = _block_rows(Q.shape[0], Wr.shape[0], cfg, arrays=1.5)
    for s, e in _blocks(Q.shape[0], bs, cfg.block_min_rows):
        a = Abar[s:e][:, rows]
        P = _prod(Q[s:e], mid, Wr)
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(a, P, out=P)
            np.copyto(P, 0.0, where=~(a > 0))
            np.fmax(out, np.nanmax(P, axis=0), out=out)
    return out


def _backoff(Abar, Q, W, mid, cfg, axis, X_in, cum, m_used, info):
    """Shrink each over-scaled row back to the smallest factor that still dominates.

    On entry X = X_in * cum dominates Abar, and the previous iterate X_in * cum/(1+m_used) did
    not. Bisect log(factor) in that bracket, per row, re-measuring the product honestly each
    time -- the whole point is that the recomputed product is NOT exactly proportional to the
    scale once the factors are signed. In place on Q (axis 0) or W (axis 1).
    """
    X = Q if (axis == 0) else W
    rows = np.flatnonzero(cum > cfg.backoff_trigger)
    # The bracket is [cum/(1+m_used), cum]: the previous iterate, which the loop rejected. Its
    # lower end is legitimate even when it equals 1 (no scaling at all), so the test is >=.
    rows = rows[cum[rows] / (1.0 + m_used) >= 1.0]
    info.clear()
    info.update(axis=int(axis), m_used=float(m_used), n_rows=int(rows.size),
                n_scaled=int((cum > cfg.backoff_trigger).sum()), n_total=int(cum.size))
    if rows.size == 0:
        return 0

    lo = np.log(cum[rows] / (1.0 + m_used))                # known bad
    hi = np.log(cum[rows])                                 # known good
    best = cum[rows].copy()
    for _ in range(int(cfg.backoff_iters)):
        midpt = 0.5 * (lo + hi)
        X[rows] = X_in[rows] * np.exp(midpt)[:, None]
        r = _row_ratio(Abar, Q, W, mid, cfg, axis, rows)
        ok = np.isfinite(r) & (r <= 1.0)
        best = np.where(ok, np.exp(midpt), best)
        hi = np.where(ok, midpt, hi)
        lo = np.where(ok, lo, midpt)

    X[rows] = X_in[rows] * best[:, None]
    r = _row_ratio(Abar, Q, W, mid, cfg, axis, rows)
    bad = ~(np.isfinite(r) & (r <= 1.0))
    if np.any(bad):                                        # never trust the bisection blindly
        X[rows[bad]] = X_in[rows[bad]] * cum[rows[bad]][:, None]
        best[bad] = cum[rows[bad]]
    shrink = cum[rows] / best
    info.update(n_reverted=int(bad.sum()), shrink_max=float(shrink.max()),
                shrink_med=float(np.median(shrink)),
                n_shrunk=int((shrink > 1.0 + 1e-9).sum()))
    return int(rows.size)


def _repair(Abar, Q, W, mid, cfg, axis, rows=None):
    """Scale each row of Q (axis=0) or of W (axis=1) up to admissibility, ITERATING with a
    growing margin until the recomputed product really does dominate Abar. In place.

    Returns a stats dict. Raises if the iteration cap is exhausted, which means the signed
    cancellation in the product has swamped the margin rather than that the map is hopeless --
    see LpConfig.repair_max_iter.
    """
    margin = float(cfg.repair_margin)
    growth = float(cfg.repair_growth)
    X = Q if (axis == 0) else W
    sub = None if (rows is None) else np.asarray(rows, dtype=np.int64)

    if cfg.single_shot_repair:
        if sub is None:
            rr, rc, _ = _ratios_blocked(Abar, Q, W, mid, cfg, want_rows=(axis == 0),
                                        want_cols=(axis == 1))
            r = rr if (axis == 0) else rc
            X *= np.maximum(r, 1.0)[:, None] * (1.0 + margin)
        else:
            r = _row_ratio(Abar, Q, W, mid, cfg, axis, sub)
            X[sub] *= np.maximum(r, 1.0)[:, None] * (1.0 + margin)
        return dict(max_ratio=float(r.max()), margin=float(margin), iters=1, backoff={},
                    viol=None)

    m = margin
    r0 = None
    viol = None
    X_in = X.copy() if cfg.repair_bisect else None
    cum = np.ones(X.shape[0] if (sub is None) else sub.size)

    for it in range(int(cfg.repair_max_iter)):
        if sub is None:
            # Statistics on the FIRST pass only: that is the point the LP returned, before any
            # repair, which is what the violation diagnostic is about.
            rr, rc, st = _ratios_blocked(Abar, Q, W, mid, cfg, want_rows=(axis == 0),
                                         want_cols=(axis == 1), stats=(it == 0))
            r = rr if (axis == 0) else rc
        else:
            r, st = _row_ratio(Abar, Q, W, mid, cfg, axis, sub), None
        if st is not None:
            viol = st
        if not np.all(np.isfinite(r)):
            raise RuntimeError('varmap.lp repair: the approximation is zero where Abar is not')
        if r0 is None:
            r0 = float(r.max())
        if r.max() <= 1.0:
            bk = {}
            # m/growth is the margin the last applied step used. If it is still the initial
            # one there is nothing to win (the bracket is 1+margin wide) and the bisection's
            # matmuls would be pure cost, so only escalated ladders are backed off.
            if cfg.repair_bisect and (m / growth > margin):
                if sub is None:
                    _backoff(Abar, Q, W, mid, cfg, axis, X_in, cum, m/growth, bk)
                else:
                    _backoff_subset(Abar, Q, W, mid, cfg, axis, X_in, cum, m/growth, sub, bk)
            return dict(max_ratio=r0, margin=float(m / growth), iters=it, backoff=bk, viol=viol)
        step = np.maximum(r, 1.0) * (1.0 + m)
        if sub is None:
            X *= step[:, None]
        else:
            X[sub] *= step[:, None]
        cum *= step
        m *= growth

    raise RuntimeError(f'varmap.lp repair: did not converge (max ratio {float(r.max())!r})')


def _backoff_subset(Abar, Q, W, mid, cfg, axis, X_in, cum, m_used, sub, info):
    """_backoff() when only 'sub' rows of the factor were scaled: 'cum' is indexed by position
    within 'sub' rather than by row, so the bisection is run on the sub-array and mapped back."""
    X = Q if (axis == 0) else W
    sel = np.flatnonzero(cum > cfg.backoff_trigger)
    sel = sel[cum[sel] / (1.0 + m_used) >= 1.0]
    info.clear()
    info.update(axis=int(axis), m_used=float(m_used), n_rows=int(sel.size),
                n_scaled=int((cum > cfg.backoff_trigger).sum()), n_total=int(cum.size))
    if sel.size == 0:
        return 0

    rows = sub[sel]
    lo = np.log(cum[sel] / (1.0 + m_used))
    hi = np.log(cum[sel])
    best = cum[sel].copy()
    for _ in range(int(cfg.backoff_iters)):
        midpt = 0.5 * (lo + hi)
        X[rows] = X_in[rows] * np.exp(midpt)[:, None]
        r = _row_ratio(Abar, Q, W, mid, cfg, axis, rows)
        ok = np.isfinite(r) & (r <= 1.0)
        best = np.where(ok, np.exp(midpt), best)
        hi = np.where(ok, midpt, hi)
        lo = np.where(ok, lo, midpt)

    X[rows] = X_in[rows] * best[:, None]
    r = _row_ratio(Abar, Q, W, mid, cfg, axis, rows)
    bad = ~(np.isfinite(r) & (r <= 1.0))
    if np.any(bad):
        X[rows[bad]] = X_in[rows[bad]] * cum[sel[bad]][:, None]
        best[bad] = cum[sel[bad]]
    shrink = cum[sel] / best
    info.update(n_reverted=int(bad.sum()), shrink_max=float(shrink.max()),
                shrink_med=float(np.median(shrink)),
                n_shrunk=int((shrink > 1.0 + 1e-9).sum()))
    return int(rows.size)


def repair_rows(Q, W, mid, Abar, cfg=None, *, rows=None):
    """Scale the rows of Q up until ``Q mid W^T >= Abar``, making the map admissible exactly
    and locally. Returns (Q_new, stats); Q is not modified.

    Scaling one row of Q touches only that row of the product, so this cannot break anything
    else, and 'rows' restricts it to a subset for the same reason.

    This is the low-level primitive under the multiplicative stage of the Q direction's
    repair. Prefer the map-level repair unless you are working with bare arrays.
    """
    cfg = _cfg(cfg)
    Q = np.array(Q, dtype=np.float64, copy=True)
    W = np.asarray(W, dtype=np.float64)
    st = _repair(Abar, Q, W, mid, cfg, 0, rows=rows)
    return Q, st


def repair_cols(Q, W, mid, Abar, cfg=None, *, cols=None):
    """The column counterpart of repair_rows(): scale the rows of W -- i.e. the CHANNELS of
    the product -- instead of the rows of Q. Returns (W_new, stats); W is not modified.

    Same function, other axis, but do not assume a result measured on one axis transfers to
    the other: the row form maxes over nfreq and the column form over nbeta, and those differ
    by orders of magnitude and scale differently with the config. Inflating one channel by 7x
    inflates every group in it, and a frozen W means the next Q-step cannot undo that.
    """
    cfg = _cfg(cfg)
    Q = np.asarray(Q, dtype=np.float64)
    W = np.array(W, dtype=np.float64, copy=True)
    st = _repair(Abar, Q, W, mid, cfg, 1, rows=cols)
    return W, st


####################################   the additive repairs   ####################################
#
# Both of these raise the product by ADDING a nonnegative multiple of a nonnegative column of
# W to a row of Q:
#
#     (W (q + delta e_c))_F = (W q)_F + delta W[F,c] >= (W q)_F      for delta >= 0, W[F,c] >= 0
#
# so admissibility can only improve, the factorization keeps exactly the same K columns, and
# the only cost is delta * s_c added to that group's row sum -- a relative change of ~1e-9,
# against the percents a multiplicative rescale costs. Both therefore need a nonnegative
# column of W to exist, and both are exact only for an IDENTITY mid, since the lift is defined
# on the columns of W; a caller with a non-identity mid should fold it into Q first.


def _require_identity_mid(mid, what):
    if mid is not None:
        raise RuntimeError(f'varmap.lp {what}: the additive lift adds a multiple of a COLUMN'
                           " of W to a row of Q, which only raises the product when 'mid' is"
                           ' the identity. Fold mid into Q first (Q @ mid, mid=None).')


def repair_additive(Q, W, mid, Abar, cfg=None, *, rows=None):
    """The additive lift, per GROUP: pick the cheapest nonnegative column of W that reaches
    every deficit in that group and add the smallest multiple that makes the row dominate.
    Returns (Q_new, stats); Q is not modified.

    Groups that no nonnegative column can reach fall back to a multiplicative row scale, so
    this is a COMPLETE repair on its own rather than a partial one -- which is why
    (additive_first=False, additive_last=True, rescale='none') is a legitimate setting and not
    a hole. A second multiplicative pass always runs afterwards, because the recomputed
    product is not exactly the arithmetic one.

    This is the additive stage of the W direction. The Q direction's is fix_nonneg(), which
    targets a different thing; see LpConfig's repair commentary.
    """
    cfg = _cfg(cfg)
    _require_identity_mid(mid, 'repair_additive')
    Abar = np.asarray(Abar, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64, copy=True)
    W = np.asarray(W, dtype=np.float64)
    margin = float(cfg.repair_margin)
    nrow, ncol = Abar.shape

    okc = (W.min(axis=0) >= 0) & (W.max(axis=0) > 0)
    cols = np.flatnonzero(okc)
    sW = W.sum(axis=0)
    bs = _block_rows(nrow, ncol, cfg, arrays=3.0)
    want = None if (rows is None) else np.zeros(nrow, dtype=bool)
    if want is not None:
        want[np.asarray(rows, dtype=np.int64)] = True

    add_col = np.full(nrow, -1, dtype=np.int64)
    add_del = np.zeros(nrow)
    r0 = 0.0
    n_bad = n_fallback = n_pos_tot = n_viol_tot = 0

    for s0, s1 in _blocks(nrow, bs, cfg.block_min_rows):
        a = Abar[s0:s1]
        P = Q[s0:s1] @ W.T
        Dfc = a - P
        np.maximum(Dfc, 0.0, out=Dfc)
        pos = a > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(a, P, out=P)                        # the ratio, in place
            np.copyto(P, 0.0, where=~pos)
            m = float(np.nanmax(P)) if P.size else 0.0
            n_pos_tot += int(pos.sum())
            n_viol_tot += int(((P > 1.0 + cfg.viol_tol) | (P < 0.0)).sum())
        r0 = m if (m > r0) else r0
        del P, pos

        bad = Dfc.max(axis=1) > 0.0
        if want is not None:
            bad &= want[s0:s1]
        n_bad += int(bad.sum())
        if (not bad.any()) or (cols.size == 0):
            n_fallback += int(bad.sum())
            del Dfc
            continue

        best_cost = np.full(s1 - s0, np.inf)
        for cidx in cols:
            wc = W[:, cidx]
            posF = wc > 0.0
            # A column reaches this group iff it is positive wherever the deficit is.
            reach = ~(Dfc[:, ~posF] > 0.0).any(axis=1) if np.any(~posF) \
                else np.ones(s1-s0, bool)
            if not reach.any():
                continue
            delta = (Dfc[:, posF] / wc[posF][None, :]).max(axis=1) * (1.0 + margin)
            cost = delta * sW[cidx]
            take = reach & bad & (cost < best_cost)
            best_cost = np.where(take, cost, best_cost)
            add_col[s0:s1][take] = cidx
            add_del[s0:s1][take] = delta[take]
        n_fallback += int((bad & (add_col[s0:s1] < 0)).sum())
        del Dfc

    # The RAW (pre-repair) violation figures, so that a step's reported viol_* keep meaning
    # "what the LP returned" whichever repair is selected: the second pass below would
    # otherwise report the POST-lift numbers.
    raw = dict(n_pos=int(n_pos_tot), n_viol=int(n_viol_tot),
               frac_viol=float(n_viol_tot) / max(1, n_pos_tot),
               max_ratio=float(r0), tol=float(cfg.viol_tol))
    if r0 <= 1.0:
        return Q, dict(mode='additive', max_ratio=r0, margin=float(margin), dy_rel=0.0,
                       n_rows=0, n_fallback=0, backoff={}, viol=raw)

    hit = np.flatnonzero(add_col >= 0)
    dy = 0.0
    if hit.size:
        y0 = float(np.abs(Q @ sW).sum())
        np.add.at(Q, (hit, add_col[hit]), add_del[hit])
        dy = float((add_del[hit] * sW[add_col[hit]]).sum()) / max(y0, 1e-300)

    st = _repair(Abar, Q, W, None, cfg, 0, rows=rows)
    return Q, dict(mode='additive', max_ratio=r0, margin=st['margin'], dy_rel=float(dy),
                   n_rows=int(n_bad), n_fallback=int(n_fallback), backoff=st['backoff'],
                   second_pass_max_ratio=st['max_ratio'], viol=raw)


def fix_nonneg(Q, W, mid, Abar, cfg=None, *, rows=None, max_iter=8):
    """The additive lift, per ELEMENT: raise ``Q mid W^T`` to ``max(Abar, its own rounding
    noise)`` using the cheapest nonnegative column of W in each CHANNEL. Returns (Q_new,
    stats); Q is not modified.

    This is the additive stage of the Q direction, and it is a different job from
    repair_additive(): the target is not Abar but max(Abar, noise), which is what makes it the
    safety net for the two things a positive row scale structurally cannot reach -- a negative
    product element, and an entry below its own float64 evaluation noise.

    The Abar half of that target matters as much as the sign. The element that goes negative
    is one where the LP returned a point violating its OWN positive-rhs constraint (measured:
    a channel with Abar/rowmax ~ 8e-10, where the equilibrated row spans ~1e11), so the entry
    is not merely negative, it is inadmissible. The multiplicative repair cannot see either
    problem -- a negative ratio loses to the row's max -- and lifting only to zero+epsilon
    would then leave it seeing a ratio of ~1e5 and inflating the whole group by that factor.
    Lifting straight to Abar costs ~1e-11 of the group's row sum instead.

    Elements no nonnegative column reaches are counted in the stats rather than repaired.
    """
    cfg = _cfg(cfg)
    _require_identity_mid(mid, 'fix_nonneg')
    kappa = float(cfg.noise_kappa)
    eps = float(np.finfo(np.float64).eps)
    Abar = np.asarray(Abar, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64, copy=True)
    W = np.asarray(W, dtype=np.float64)

    s = W.sum(axis=0)
    okc = (W.min(axis=0) >= 0.0) & (W.max(axis=0) > 0.0)     # usable (nonnegative) columns
    if not np.any(okc):
        return Q, dict(cols=0, repaired=0, unreachable=-1, dy=0.0, skipped=0, used={})
    cols = np.flatnonzero(okc)

    # Cheapest column per channel: most product per unit of row sum spent.
    with np.errstate(divide='ignore', invalid='ignore'):
        eff = np.where(s[cols] > 0, W[:, cols] / np.maximum(s[cols], 1e-300), 0.0)
    best = cols[np.argmax(eff, axis=1)]                      # (nfreq,)
    best_w = W[np.arange(W.shape[0]), best]

    nrow, ncol = Abar.shape
    bs = _block_rows(nrow, ncol, cfg, arrays=4.0)
    absWt = np.abs(W).T
    used = np.zeros(W.shape[1], dtype=np.int64)
    want = None if (rows is None) else np.zeros(nrow, dtype=bool)
    if want is not None:
        want[np.asarray(rows, dtype=np.int64)] = True

    n_fixed = n_unreach = n_skipped = 0
    dy = 0.0
    for _ in range(int(max_iter)):
        y = np.abs(Q @ s)               # Q is unchanged until the end of the pass
        add = np.zeros_like(Q)
        n_unreach_it = n_bad = n_kept = 0
        for lo, hi in _blocks(nrow, bs, cfg.block_min_rows):
            Ab = Abar[lo:hi]
            Qb = Q[lo:hi]
            P = Qb @ W.T
            noise = kappa * eps * (np.abs(Qb) @ absWt)
            target = np.maximum(noise, Ab)
            del noise
            bad = P < target
            if want is not None:
                bad &= want[lo:hi][:, None]
            n_bad += int(bad.sum())
            bi, fi = np.nonzero(bad)
            del bad
            usable = best_w[fi] > 0.0
            n_unreach_it += int((~usable).sum())
            bi, fi = bi[usable], fi[usable]
            if bi.size == 0:
                continue
            delta = (target[bi, fi] - P[bi, fi]) / best_w[fi]
            # Skip an OPTIONAL (noise-only) lift whose cost exceeds noise_cost_rel of that
            # group's own row sum. An admissibility lift is never skipped.
            need = P[bi, fi] < Ab[bi, fi]
            keep = need | (delta * s[best[fi]] <= cfg.noise_cost_rel
                           * np.maximum(y[lo:hi][bi], 1e-300))
            n_skipped += int((~keep).sum())
            bi, fi, delta = bi[keep], fi[keep], delta[keep]
            if bi.size == 0:
                continue
            n_kept += int(bi.size)
            used += np.bincount(best[fi], minlength=W.shape[1])
            np.maximum.at(add, (bi + lo, best[fi]), delta)
        if n_bad:
            n_unreach = n_unreach_it
        if (n_bad == 0) or (n_kept == 0):
            break
        n_fixed += n_kept
        # Accumulated as a per-row vector and summed once, so that even this reordering-prone
        # sum is bit-identical under blocking.
        dy += float((add @ np.maximum(s, 0.0)).sum())
        Q += add

    return Q, dict(cols=int(cols.size), repaired=int(n_fixed), unreachable=int(n_unreach),
                   dy=float(dy), skipped=int(n_skipped),
                   used={int(c): int(v) for c, v in enumerate(used) if v})


def _w_channel_stats(Abar, Q, W, mid, cfg, *, block_rows=None, want_nonpos=True,
                     thresholds=()):
    """Blocked per-CHANNEL statistics of the raw product, for the W-step's guard.

    Returns a dict with 'rc' (per-channel max of Abar/(Q mid W^T)), 'nonpos' (the product is
    <= 0 where Abar > 0, or negative anywhere), the entry violation count, and one count per
    (threshold, tag) pair.
    """
    Abar = np.asarray(Abar)
    nrow, ncol = Abar.shape
    bs = _block_rows(nrow, ncol, cfg, arrays=2.0) if (block_rows is None) else int(block_rows)
    rc = np.full(ncol, -np.inf)
    nonpos = np.zeros(ncol, dtype=bool)
    n_viol = 0
    for s0, s1 in _blocks(nrow, bs, cfg.block_min_rows):
        a = Abar[s0:s1]
        P = _prod(Q[s0:s1], mid, W)
        pos = a > 0
        if want_nonpos:
            nonpos |= ((P <= 0.0) & pos).any(axis=0) | (P < 0.0).any(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(a, P, out=P)
            np.copyto(P, 0.0, where=~pos)
            np.fmax(rc, np.nanmax(P, axis=0), out=rc)
            n_viol += int(((P > 1.0 + cfg.viol_tol) | (P < 0.0)).sum())
    rc = np.where(np.isfinite(rc), rc, np.inf)
    out = dict(rc=rc, nonpos=nonpos, n_entry_viol=int(n_viol), n_entry=int(nrow)*int(ncol))
    for thr, tag in thresholds:
        out[f'w_chan_viol_{tag}'] = int((rc > thr).sum())
    return out


####################################   the LP kernel   ####################################


# scipy/HiGHS status codes.
_STATUS_NAMES = {0: 'optimal', 1: 'iter_limit', 2: 'infeasible', 3: 'unbounded',
                 4: 'numerical'}


def _linprog(cost, A_ub, b_ub, bounds, cfg):
    """scipy.optimize.linprog on HiGHS, with the thread count pinned -- see LpConfig.threads.

    scipy is imported here rather than at module scope, and the 'threads' option warning is
    silenced locally rather than by a module-level filter, so that importing this module
    changes nothing globally.
    """
    from scipy.optimize import linprog
    opts = dict(primal_feasibility_tolerance=float(cfg.primal_tol),
                dual_feasibility_tolerance=float(cfg.primal_tol),
                threads=int(cfg.threads))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected')
        return linprog(c=cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                       options=opts)


def _clip_rhs(b, clip):
    """Zero the entries of b below ``clip * max(b)`` -- see LpConfig.clip_rel.

    Returns a NEW array when it does anything, so it is safe to call on a read-only view.
    """
    if clip <= 0.0:
        return b
    m = float(b.max())
    return np.where(b < clip*m, 0.0, b) if (m > 0.0) else b


def solve_cover_lp(cost, M, b, cfg, *, x_feasible=None, live=None, status=None,
                   zero_rows=None, lower=None, offset=None):
    """``min cost . x  s.t.  M x >= b`` (and ``x >= 0`` if cfg.nonneg). Returns (x, ok).

    Rows with b_i > 0 are equilibrated to right-hand side 1, which is load-bearing rather than
    cosmetic; rows with b_i == 0 are kept as a separate block with cfg.zero_rhs_margin unless
    they are automatic. See LpConfig.equilibrate and LpConfig.zero_rhs_margin for both, and
    for why a zero-rhs row may not simply be dropped.

    Parameters
    ----------
    cost : ndarray
        (K,) objective.
    M : ndarray
        (n, K) constraint matrix.
    b : ndarray
        (n,) right-hand side. May contain zeros, and may be NEGATIVE -- from a factored
        reference, or from 'offset' -- in which case see LpConfig.zero_rhs_margin for what
        happens to that row, which is not what it first looks like.
    x_feasible : ndarray or None
        Returned with ``ok=False`` if the solve fails. None RAISES instead, so a caller that
        has no fallback finds out rather than getting a silent zero.
    live : ndarray or None
        (K,) bool: columns to optimize over, the rest pinned to 0. None computes it as "the
        column is not identically zero in M". A column that appears in no constraint would
        otherwise send a free variable to minus infinity, so dead columns are dropped from the
        LP rather than repriced.
    zero_rows : bool or None
        None keeps the zero-rhs rows unless they are automatic (which they are exactly when
        both the variable and the matrix are nonnegative); True always keeps them; False
        always drops them, which makes the free-sign LP genuinely unbounded and is only useful
        for demonstrating that.
    lower : float or None
        A finite negative floor on the variables, used only when cfg.nonneg is False, so that
        the LP may use a little cancellation but not an unbounded amount.
    offset : ndarray or None
        (n,) constant contribution of columns held fixed outside this LP, so the constraint is
        ``M x >= b - offset``. The equilibration is unchanged -- each positive-rhs row is
        still divided by its own b_i > 0 -- but the resulting right-hand side may then be
        negative, which is legitimate.

    Returns
    -------
    tuple
        (x, ok) with x of shape (K,).
    """
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    K = M.shape[1]
    nonneg = bool(cfg.nonneg)

    if live is None:
        live = np.abs(M).max(axis=0) > 0
    x = np.zeros(K)
    if not np.any(live):
        return x, True

    off = None if (offset is None) else np.asarray(offset, dtype=np.float64)

    pos = b > 0
    blocks = [M[pos][:, live] / b[pos, None]]
    rhs = [np.ones(int(pos.sum())) if (off is None) else (1.0 - off[pos] / b[pos])]

    keep_zero = (not (nonneg and (M.min() >= 0.0))) if (zero_rows is None) else bool(zero_rows)
    if keep_zero:
        Mz = M[~pos][:, live]
        if Mz.shape[0]:
            sc = np.abs(Mz).max(axis=1)
            k = sc > 0                       # a structurally zero row is 0 >= 0, always true
            if np.any(k):
                blocks.append(Mz[k] / sc[k, None])
                z = np.full(int(k.sum()), float(cfg.zero_rhs_margin))
                if off is not None:
                    # keep the TOTAL product (pinned + free) strictly positive here
                    z = z - off[~pos][k] / sc[k]
                rhs.append(z)

    Mu = np.vstack(blocks)
    if Mu.shape[0] == 0:
        return x, True
    bu = np.concatenate(rhs)

    bounds = (0, None) if nonneg else ((None, None) if (lower is None) else (float(lower), None))
    res = _linprog(cost[live], -Mu, -bu, bounds, cfg)
    if status is not None:
        status[int(res.status)] = status.get(int(res.status), 0) + 1
    if res.success and (res.x is not None):
        v = np.asarray(res.x, dtype=np.float64)
        x[live] = np.maximum(v, 0.0) if nonneg else v
        return x, True
    if x_feasible is None:
        raise RuntimeError(f'varmap.lp solve_cover_lp failed with no fallback: {res.message}')
    return np.asarray(x_feasible, dtype=np.float64).copy(), False


####################################   constraint generation   ####################################
#
# THE LEVER. The Q-step LP has one constraint per FREQUENCY CHANNEL -- 28160 of them at CHORD,
# for K <= 128 variables. An optimal basis uses at most K of them, so >99.5% are slack, and
# the cost law says the constraint count is the dominant axis.
#
# A version of this for a NONNEGATIVE LP cannot simply be reused, and the reason is the whole
# design of what follows. A nonnegative LP is bounded on any subset of the rows, because the
# cost is nonnegative and so is x. The free-sign LP is not: its boundedness argument uses
# EVERY channel, since a recession direction d has (Wd)_F >= 0 in every constrained channel
# and the objective change is sum_F (Wd)_F. Dropping channels destroys the bound, and
# constraint sampling was measured to make 896/896 subproblems unbounded.
#
# The fix is one extra row. Alongside the working set S we carry the AGGREGATE of the
# constraints not in S,
#
#       sum_{F not in S} (W q)_F  >=  sum_{F not in S} b_F ,
#
# which is a nonnegative combination of the omitted constraints and is therefore
#   (a) VALID: it cuts off no point of the full feasible set, so the working problem stays a
#       RELAXATION and its optimum is a lower bound on the full LP's;
#   (b) BOUNDING: s.q = sum_{F in S} (Wq)_F + sum_{F not in S} (Wq)_F is bounded below by the
#       sum of ALL the right-hand sides, for ANY working set including the empty one.
# (A dropped row -- b_F = 0 with a structurally zero row -- contributes 0 to both sides, so
# the identity used in (b) is exact.)
#
# Hence the method is EXACT, by the textbook argument: solve the relaxation, test ALL the
# constraints, add the violated ones, repeat. On exit no constraint is violated, so the
# working optimum is feasible for the full LP and, being a relaxation optimum, optimal for it.
# It terminates in at most (number of rows) rounds because the working set strictly grows.


_CUTS_M_CACHE = {}


def _cuts_seed(Mk, beff, sk, n_init):
    """Initial working set: the rows that DEMAND the most relative to what any single column
    can supply (these are the rows a covering LP binds on), plus, for each column, the rows
    that column serves best."""
    nk, K = Mk.shape
    n_init = int(min(nk, max(n_init, 8)))
    cap = np.abs(Mk).max(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        hard = np.where(cap > 0, beff / np.maximum(cap, 1e-300), np.inf)
    half = max(1, n_init // 2)
    sel = set(int(i) for i in np.argpartition(-hard, min(half, nk-1))[:half])
    take = max(1, (n_init - half) // max(K, 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        E = Mk / np.maximum(sk, 1e-300)[:, None]         # the equilibrated matrix
    for cidx in range(K):
        col = E[:, cidx]
        if take >= nk:
            sel.update(range(nk))
            break
        sel.update(int(i) for i in np.argpartition(-col, take)[:take])
    return np.fromiter(sorted(sel), dtype=np.int64, count=len(sel))


def solve_cover_lp_cuts(cost, M, b, cfg, *, x_feasible=None, live=None, status=None,
                        zero_rows=None, lower=None, offset=None, warm=None, stats=None):
    """solve_cover_lp() by constraint generation. Same LP, same equilibration, same answer.

    'warm', if given, is a working set in ORIGINAL row indices from a previous subproblem --
    a pure warm start, it changes no answer, only the round count. The BINDING rows of the
    solve are written back to ``stats['act']`` when 'stats' is a dict, and those (not the
    whole working set) are what should be handed on: at most K constraints are tight at a
    vertex, so the handoff is a seed of size ~K.

    Falls back to the exact solver, so that the answer is then exactly the default path's,
    whenever the loop does not converge, an inner solve fails, or the problem uses a feature
    this path does not implement.
    """
    if (offset is not None) or ((zero_rows is not None) and (not bool(zero_rows))):
        # The offset and the "drop the zero rows" demonstration path are not implemented here;
        # fall back rather than guess.
        if stats is not None:
            stats['fallback_unsupported'] = stats.get('fallback_unsupported', 0) + 1
        return solve_cover_lp(cost, M, b, cfg, x_feasible=x_feasible, live=live, status=status,
                              zero_rows=zero_rows, lower=lower, offset=offset)

    tol = float(cfg.cuts_tol)
    max_rounds = int(cfg.cuts_rounds)
    nonneg = bool(cfg.nonneg)

    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    K = M.shape[1]
    if live is None:
        live = np.abs(M).max(axis=0) > 0
    x = np.zeros(K)
    if not np.any(live):
        return x, True

    all_live = bool(np.all(live))
    Ml = M if all_live else np.ascontiguousarray(M[:, live])
    cl = cost if all_live else cost[live]
    nl = Ml.shape[1]

    # ---- the same row set, the same equilibration, as solve_cover_lp ----
    pos = b > 0
    keep_zero = (not (nonneg and (M.min() >= 0.0))) if (zero_rows is None) else bool(zero_rows)
    n_all = Ml.shape[0]
    beff = np.where(pos, b, 0.0)         # right-hand side in UNEQUILIBRATED units
    rhs = np.where(pos, 1.0, 0.0)        # right-hand side AFTER equilibration
    sca = np.where(pos, b, 0.0)          # the row scale
    keep = pos.copy()
    if keep_zero and not pos.all():
        zi = np.flatnonzero(~pos)
        sc = np.abs(Ml[zi]).max(axis=1)
        g = sc > 0
        zz = zi[g]
        sca[zz] = sc[g]
        rhs[zz] = float(cfg.zero_rhs_margin)
        beff[zz] = float(cfg.zero_rhs_margin) * sc[g]
        keep[zz] = True
    rows = np.flatnonzero(keep)
    nk = rows.size
    if nk == 0:
        return x, True
    if nk == n_all:
        Mk, bk, rk, sk = Ml, beff, rhs, sca
        tot_coef = cl                    # exactly cost[live]: both are M[:, live].sum(axis=0)
    else:
        Mk = np.ascontiguousarray(Ml[rows])
        bk, rk, sk = beff[rows], rhs[rows], sca[rows]
        tot_coef = Mk.sum(axis=0)
    tot_rhs = float(bk.sum())

    # The band partition and the band sums of M do not depend on the right-hand side, so they
    # are computed once per (matrix, band count) and cached -- otherwise they are a full pass
    # over M per LP, which at 28160 x 128 is a third of the cost once the loop is fast.
    nagg = max(1, int(cfg.cuts_nagg))
    if nagg > 1:
        ckey = (Mk.__array_interface__['data'][0], nk, nl, nagg)
        hit = _CUTS_M_CACHE.get(ckey)
        if hit is None:
            edges = np.unique(np.linspace(0, nk, nagg + 1).astype(np.int64))
            starts = edges[:-1]
            blk = np.repeat(np.arange(starts.size, dtype=np.int64), np.diff(edges))
            hit = (starts, blk, np.add.reduceat(Mk, starts, axis=0))
            if len(_CUTS_M_CACHE) > 8:
                _CUTS_M_CACHE.clear()
            _CUTS_M_CACHE[ckey] = hit
        starts, blk, tot_coef_b = hit
        tot_rhs_b = np.add.reduceat(bk, starts)

    if nk <= max(2*nl, 64):              # nothing to gain
        return solve_cover_lp(cost, M, b, cfg, x_feasible=x_feasible, live=live, status=status,
                              zero_rows=zero_rows, lower=lower, offset=offset)

    n_init = int(max(cfg.cuts_init * nl, 64))
    max_add = int(max(cfg.cuts_maxadd * nl, 8))

    inS = np.zeros(nk, dtype=bool)
    nwarm = 0
    if (warm is not None) and len(warm):
        # 'warm' is in original row indices so that a pool can be shared between subproblems
        # whose kept-row sets differ; with a dense matrix every row is kept and this is the
        # identity.
        w = np.asarray(warm, dtype=np.int64)
        if nk == n_all:
            w = w[(w >= 0) & (w < nk)]
        else:
            p = np.searchsorted(rows, w)
            good = (p < nk) & (rows[np.minimum(p, nk-1)] == w)
            w = p[good]
        inS[w] = True
        nwarm = int(inS.sum())
    if (nwarm < max(8, nl)) or (cfg.cuts_warm_seed == 'always'):
        inS[_cuts_seed(Mk, bk, sk, n_init)] = True

    bounds = (0, None) if nonneg else ((None, None) if (lower is None) else (float(lower), None))
    xl = np.zeros(nl)
    nrounds = 0
    st_last = None
    for _ in range(max_rounds):
        nrounds += 1
        sel = np.flatnonzero(inS)
        Mu = Mk[sel] / sk[sel, None]
        bu = rk[sel]
        if cfg.cuts_agg and (sel.size < nk):
            if nagg > 1:
                ac = np.zeros_like(tot_coef_b)
                ar = np.zeros_like(tot_rhs_b)
                if sel.size:
                    bs = blk[sel]                       # nondecreasing (sel is sorted)
                    u, first = np.unique(bs, return_index=True)
                    ac[u] = np.add.reduceat(Mk[sel], first, axis=0)
                    ar[u] = np.add.reduceat(bk[sel], first)
                agg_c = tot_coef_b - ac
                agg_b = tot_rhs_b - ar
            else:
                agg_c = (tot_coef - Mk[sel].sum(axis=0))[None, :]
                agg_b = np.array([tot_rhs - float(bk[sel].sum())])
            d = np.where(agg_b > 0, agg_b, np.abs(agg_c).max(axis=1))
            g = d > 0
            if np.any(g):
                Mu = np.vstack([Mu, agg_c[g] / d[g, None]])
                bu = np.concatenate([bu, agg_b[g] / d[g]])
        res = _linprog(cl, -Mu, -bu, bounds, cfg)
        st_last = int(res.status)
        if not (res.success and (res.x is not None)):
            break                        # -> the exact full solve below; never worse
        xl = np.asarray(res.x, dtype=np.float64)
        if nonneg:
            xl = np.maximum(xl, 0.0)
        v = Mk @ xl
        resid = v / sk - rk              # equilibrated residual; violated when < 0
        viol = (resid < -tol) & (~inS)
        nbad = int(viol.sum())
        if nbad == 0:
            if status is not None:
                status[st_last] = status.get(st_last, 0) + 1
            if stats is not None:
                stats['rounds'] = stats.get('rounds', 0) + nrounds
                stats['rows'] = stats.get('rows', 0) + int(sel.size)
                stats['nlp'] = stats.get('nlp', 0) + 1
                act = np.flatnonzero(resid <= cfg.cuts_act_tol)
                if act.size > 4*nl:
                    act = act[np.argsort(resid[act])[:4*nl]]
                stats['act'] = act if (nk == n_all) else rows[act]
                stats['nact'] = int(act.size)
                stats['worst'] = float(-resid.min()) if resid.size else 0.0
            xf = xl if all_live else np.zeros(K)
            if not all_live:
                xf[live] = xl
            return xf, True
        bad = np.flatnonzero(viol)
        if nbad > max_add:
            bad = bad[np.argsort(resid[bad])[:max_add]]
        inS[bad] = True

    if stats is not None:
        stats['fallback'] = stats.get('fallback', 0) + 1
        stats['rounds'] = stats.get('rounds', 0) + nrounds
        stats['nlp'] = stats.get('nlp', 0) + 1
    return solve_cover_lp(cost, M, b, cfg, x_feasible=x_feasible, live=live, status=status,
                          zero_rows=zero_rows, lower=lower, offset=offset)


####################################   parallel dispatch   ####################################
#
# The subproblems are independent, so they are farmed out to a FORK-based pool whose big
# arrays are inherited rather than pickled. Two consequences the caller must know:
#
#   - a pool is created only when workers > 1. Do not use one in a process that has
#     initialized CUDA; fork() after CUDA initialization is not supported by the driver.
#   - the pool falls back to SERIAL execution rather than hanging. When a container's pid
#     limit is exhausted, fork() or a pthread_create inside HiGHS fails, the worker dies, and
#     pool.map then blocks forever on a result that will never arrive. Serial is 10-50x
#     slower but always finishes, and no result is lost.


_SHARED = {}


def _init_worker(shared, set_env=True):
    _SHARED.clear()
    _SHARED.update(shared)
    # Each LP is tiny, so threads inside HiGHS or BLAS would fight the process pool. Only a
    # forked CHILD's environment is touched: doing this in the parent would change the
    # environment every later subprocess of the CALLER inherits, for no benefit here (the
    # parent's BLAS is long since initialized).
    if set_env:
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[v] = '1'


def _lp_chunk(bounds):
    """Solve subproblems [start, stop) of the shared family. The pool's work unit."""
    start, stop = bounds
    S = _SHARED
    cfg = S['cfg']
    M, cost, B, live = S['M'], S['cost'], S['B'], S['live']
    X0, lower = S.get('X0'), S.get('lower')
    Mpin, Xpin = S.get('Mpin'), S.get('Xpin')
    clip = float(S.get('clip', 0.0))
    cuts = bool(S.get('cuts', False))

    out = np.zeros((stop - start, M.shape[1]))
    nfail = 0
    failed = []
    st = {}
    cst = {} if cuts else None

    poolcap = int(cfg.cuts_pool) if cuts else 0
    poolwin = int(cfg.cuts_pool_window)
    pool = np.asarray(S.get('pool0', ()), dtype=np.int64)
    recent = []                                  # the last poolwin binding sets
    warm = pool if pool.size else None

    for j in range(start, stop):
        b = B[:, j]
        if clip > 0.0:
            b = _clip_rhs(np.ascontiguousarray(b), clip)
        off = None if (Mpin is None) else (Mpin @ Xpin[j])
        lo = None if (lower is None) else float(lower[j])
        seed = None if (X0 is None) else X0[j]
        if cuts:
            cst.pop('act', None)
            x, ok = solve_cover_lp_cuts(cost, M, b, cfg, x_feasible=seed, live=live, status=st,
                                        lower=lo, offset=off, warm=warm, stats=cst)
            act = cst.get('act')
            if poolcap and (act is not None):
                if poolwin > 0:
                    recent.append(np.asarray(act, dtype=np.int64))
                    if len(recent) > poolwin:
                        del recent[0]
                    pool = np.unique(np.concatenate(recent))
                elif pool.size:
                    pool = np.union1d(pool, act)
                else:
                    pool = np.asarray(act, dtype=np.int64)
                if pool.size > poolcap:
                    pool = pool[:poolcap]
                warm = pool
            elif cfg.cuts_warm:
                warm = act if (act is not None) else warm
        else:
            x, ok = solve_cover_lp(cost, M, b, cfg, x_feasible=seed, live=live, status=st,
                                   lower=lo, offset=off)
        out[j-start] = x
        if not ok:
            nfail += 1
            failed.append(j)

    if cuts:
        for k in ('rounds', 'rows', 'nlp', 'fallback', 'fallback_unsupported'):
            if k in cst:
                st['cuts_' + k] = st.get('cuts_' + k, 0) + int(cst[k])
    return start, out, nfail, st, failed


def _default_workers(workers):
    if workers is None:
        return max(1, min(32, (os.cpu_count() or 1)))
    return max(1, int(workers))


def _map_chunks(fn, n, shared, workers, cfg, chunk=None, progress=False):
    """Run fn over [0, n) in chunks, in a fork pool, falling back to serial execution in the
    parent if the pool cannot be created or stops producing results."""
    if chunk is None:
        chunk = max(1, min(256, n // (8*max(workers, 1)) or 1))
    bounds = [(i, min(i+chunk, n)) for i in range(0, n, chunk)]

    def report(done, t0):
        if progress:
            print(f'  varmap.lp: {done}/{len(bounds)} chunks, {time.time()-t0:.0f} s',
                  flush=True)

    def serial():
        _init_worker(shared, set_env=False)
        t0 = time.time()
        out = []
        for i, bd in enumerate(bounds):
            out.append(fn(bd))
            if progress and ((i+1) % max(1, len(bounds)//20) == 0):
                report(i+1, t0)
        return out

    if workers <= 1:
        return serial()

    import multiprocessing as mp
    try:
        ctx = mp.get_context('fork')
        pool = ctx.Pool(workers, initializer=_init_worker, initargs=(shared,))
    except (OSError, BlockingIOError, ValueError) as e:
        print(f'  varmap.lp: pool creation failed ({e}); running serially', flush=True)
        return serial()

    try:
        it = pool.imap_unordered(fn, bounds, chunksize=1)
        out = []
        t0 = time.time()
        for i in range(len(bounds)):
            out.append(it.next(timeout=float(cfg.chunk_timeout)))
            if progress and ((i+1) % max(1, len(bounds)//20) == 0):
                report(i+1, t0)
        pool.close()
        pool.join()
        return out
    except Exception as e:                      # a timeout, or a worker that died
        print(f'  varmap.lp: pool failed ({type(e).__name__}: {e}); running serially',
              flush=True)
        pool.terminate()
        pool.join()
        return serial()


def _gather(parts, ncol, n):
    """Reassemble chunk results into (X, n_failed, status, failed_subproblems)."""
    out = np.zeros((n, ncol))
    nfail = 0
    status = {}
    failed = []
    for start, blk, nf, st, bad in parts:
        out[start:start+blk.shape[0]] = blk
        nfail += nf
        failed.extend(bad)
        for k, v in st.items():
            status[k] = status.get(k, 0) + v
    return out, nfail, status, failed


####################################   the primitive   ####################################


def solve_covering_lps(M, B, cost, cfg, *, x_seed=None, live=None, lower=None,
                       offset_factors=None, workers=None, progress=False, solver='highs'):
    """Solve ``min cost . x  s.t.  M x >= B[:,j]`` independently for each column j of B.

    This is the single computational primitive under both steps. Every numerical lesson lives
    here and in LpConfig, in one place, rather than being duplicated in two step functions.

    THE SOLVER'S REPORTED STATUS IS NOT EVIDENCE OF ADMISSIBILITY -- see the module docstring
    -- so the returned 'info' counts failures and reports statuses, and the caller is expected
    to MEASURE the residual violation (violation_stats(), or the first pass of a repair) and
    to repair. q_step() and w_step() do both.

    Parameters
    ----------
    M : ndarray
        (n, K) constraint matrix, shared by every subproblem.
    B : ndarray
        (n, nsub) right-hand sides, one per COLUMN. Only ``B[:, j]`` is ever read, so passing
        a transposed view of a row-major array (or a memmap) is free and is what the Q-step
        does -- materializing the transpose would be a second copy of the whole map.
    cost : ndarray
        (K,) objective.
    x_seed : ndarray or None
        (nsub, K) fallback points, returned for the subproblems whose solve fails. None makes
        a failure raise.
    lower : ndarray or None
        (nsub,) per-subproblem lower bound on the variables; see solve_cover_lp.
    offset_factors : tuple or None
        (Mpin, Xpin), giving subproblem j the offset ``Mpin @ Xpin[j]`` for columns held fixed
        outside this LP. Passed as factors rather than as an (n, nsub) array because that
        array would be the size of the whole map.
    workers : int or None
        Processes in the fork pool; None picks min(32, cpu_count). See the dispatch note above
        before using more than 1.

    Returns
    -------
    tuple
        (X, info) with X of shape (nsub, K) -- X[j] is the solution of subproblem j, which is
        directly the Q or W a step returns.
    """
    cfg._check_implemented()
    if solver != 'highs':
        raise RuntimeError(f'varmap.lp solve_covering_lps: solver={solver!r} is not'
                           " supported; pass a whole replacement via a step's solve_fn"
                           ' instead of adding a backend here.')

    M = np.ascontiguousarray(np.asarray(M, dtype=np.float64))
    nsub = int(B.shape[1])
    if M.shape[0] != B.shape[0]:
        raise RuntimeError(f'varmap.lp solve_covering_lps: M has {M.shape[0]} rows but B has'
                           f' {B.shape[0]}')
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape != (M.shape[1],):
        raise RuntimeError(f'varmap.lp solve_covering_lps: cost has shape {cost.shape},'
                           f' expected ({M.shape[1]},)')

    workers = _default_workers(workers)
    live = (np.abs(M).max(axis=0) > 0) if (live is None) else np.asarray(live)
    Mpin, Xpin = (None, None) if (offset_factors is None) else offset_factors

    shared = dict(cfg=cfg, M=M, cost=cost, B=B, live=live, X0=x_seed, lower=lower,
                  Mpin=Mpin, Xpin=Xpin, clip=float(cfg.clip_rel))

    info = {}
    if cfg.cuts and (M.shape[0] >= int(cfg.cuts_min_rows)):
        shared['cuts'] = True
        if cfg.cuts_pool and cfg.cuts_pool_sample:
            # Warm the channel pool on a spread sample of subproblems BEFORE the parallel
            # pass, so that every chunk starts in the steady state instead of paying the cold
            # start. These are ordinary cut-loop solves and their answers are thrown away.
            t_pool = time.time()
            ns = int(min(cfg.cuts_pool_sample, nsub))
            pool = np.zeros(0, dtype=np.int64)
            cst = {}
            for j in np.linspace(0, nsub-1, ns).astype(np.int64):
                cst.pop('act', None)
                solve_cover_lp_cuts(cost, M,
                                    _clip_rhs(np.asarray(B[:, j], dtype=np.float64),
                                              float(cfg.clip_rel)),
                                    cfg, x_feasible=None if (x_seed is None) else x_seed[j],
                                    live=live, warm=(pool if pool.size else None), stats=cst)
                a = cst.get('act')
                if a is not None:
                    pool = np.union1d(pool, a) if pool.size else np.asarray(a, dtype=np.int64)
                    if pool.size > int(cfg.cuts_pool):
                        pool = pool[:int(cfg.cuts_pool)]
            shared['pool0'] = pool
            info.update(pool_seconds=time.time()-t_pool, pool_size=int(pool.size),
                        pool_sample=ns)

    t0 = time.time()
    X, nfail, status, failed = _gather(
        _map_chunks(_lp_chunk, nsub, shared, workers, cfg, progress=progress),
        M.shape[1], nsub)
    lp_seconds = time.time() - t0

    cutstats = {k[5:]: v for k, v in status.items()
                if isinstance(k, str) and k.startswith('cuts_')}
    status = {k: v for k, v in status.items()
              if not (isinstance(k, str) and k.startswith('cuts_'))}
    info.update(lp_seconds=lp_seconds, n_lp=nsub, n_failed=int(nfail), failed=failed,
                workers=workers, dead_cols=int((~live).sum()),
                status={_STATUS_NAMES.get(k, k): v for k, v in status.items()})
    if cutstats:
        n = max(1, int(cutstats.get('nlp', 0)))
        info.update(cuts=True, cuts_stats=cutstats,
                    cuts_rounds_mean=cutstats.get('rounds', 0)/n,
                    cuts_rows_mean=cutstats.get('rows', 0)/n,
                    cuts_fallback=cutstats.get('fallback', 0))
    return X, info


####################################   the repair pipeline   ####################################


def _apply_repair(Abar, Q, W, cfg, axis, info):
    """Run the three-stage repair along 'axis' ('rows' or 'cols'), returning (Q, W).

    The stages and what each is worth are documented once, on LpConfig, rather than restated
    here: a caller comparing a step's built-in repair against a re-application of it must not
    have two vocabularies to reconcile. The two report under separate keys (``repair_pre_*``
    and ``repair_post_*``) so that a change to one stays visible.
    """
    rescale = cfg.resolved_rescale(axis)
    additive = fix_nonneg if (axis == 'rows') else repair_additive
    viol_raw = None
    max_r = margin = None
    backoff = {}
    viol = None

    if cfg.additive_first:
        # The multiplicative pass is no longer the first thing to touch the LP's output, so
        # its first pass would report a post-lift violation and the diagnostic would go blind
        # exactly where it is needed. Take the solver's own figures explicitly instead; it
        # costs one blocked pass, and only when this stage runs.
        viol_raw = violation_stats(Q, W, None, Abar, cfg)
        Q, st = additive(Q, W, None, Abar, cfg)
        info.update({f'repair_pre_{k}': v for k, v in st.items() if k != 'viol'})

    if rescale == 'rows':
        Q, st = repair_rows(Q, W, None, Abar, cfg)
    elif rescale == 'cols':
        W, st = repair_cols(Q, W, None, Abar, cfg)
    elif rescale == 'none':
        st = {}                     # measured by the additive stage, or below
    else:
        raise RuntimeError(f'varmap.lp: unknown rescale {cfg.rescale!r}')
    max_r, margin = st.get('max_ratio'), st.get('margin')
    backoff, viol = st.get('backoff', {}), st.get('viol')

    if cfg.additive_last:
        Q, st3 = additive(Q, W, None, Abar, cfg)
        info.update({f'repair_post_{k}': v for k, v in st3.items() if k != 'viol'})
        # Its own first pass measured the same point, so take the figures from it rather than
        # paying for a second full pass over the product -- which at CHORD scale is 172 GiB of
        # streaming for a number we are being handed.
        if max_r is None:
            max_r, margin = st3.get('max_ratio'), st3.get('margin')
            backoff = st3.get('backoff', {})
        if viol is None:
            viol = st3.get('viol')

    if max_r is None:
        # Nothing above measured it: the raw point, with no repair selected at all. It is
        # still worth one pass, because "the solver said optimal" is not evidence.
        rr, _, viol = _ratios_blocked(Abar, Q, W, None, cfg, want_rows=True, want_cols=False,
                                      stats=True)
        max_r, margin = float(np.nanmax(rr)), 0.0

    if viol_raw is not None:
        # Report the SOLVER's figures as max_r_raw and viol_*, which is what they mean
        # everywhere else, and keep the post-lift ratio the multiplicative pass actually saw
        # under its own key.
        info['max_r_after_pre_additive'] = max_r
        max_r, viol = viol_raw['max_ratio'], viol_raw

    info.update(max_r_raw=float(max_r), repair_margin=float(margin), backoff=backoff,
                repair_label=cfg.repair_label)
    if viol:
        info.update(viol_frac=viol.get('frac_viol'), viol_n=viol.get('n_viol'),
                    viol_n_pos=viol.get('n_pos'), viol_max=viol.get('max_ratio'),
                    viol_tol=viol.get('tol'), viol_rows=viol.get('n_rows_viol'),
                    viol_frac_rows=viol.get('frac_rows_viol'))
    return Q, W


####################################   rescuing failed subproblems   ####################################


def _rescue_q_failed(Abar, W, Q, failed, cfg, workers, info):
    """Re-solve the failed groups on PREFIXES of W, keeping a row only if its objective
    strictly improves and the result is admissible. In place on Q; returns the number improved.

    Legitimate rather than a new algorithm: any solution of the restricted LP is a feasible
    point of the full one (pad with zeros), so it is admissible by the same argument; it uses
    only columns of W, so the factorization still has exactly K columns and THE RANK IS
    UNCHANGED; and the comparison is by the LP's own objective, so a worse point is never
    accepted. It matters because one failure per ~450 groups costs more D than an entire
    doubling of the rank.
    """
    rows = np.asarray(sorted(set(int(i) for i in failed)), dtype=int)
    K = W.shape[1]
    if rows.size == 0:
        return 0

    # The prefix solves run on the plain solver: a rescue is a handful of LPs, and the point
    # is to reproduce the default path's answer on them exactly.
    sub = dataclasses.replace(cfg, cuts=False)
    best = Q[rows] @ W.sum(axis=0)              # incumbent objective, per failed row
    n_improved = 0

    for Kp in cfg.rescue_ladder:
        if Kp >= K:
            continue
        Wp = np.ascontiguousarray(W[:, :Kp])
        # An explicit zero fallback, because the rescue only ever runs on rows whose LP has
        # already failed, so the prefix solve fails too on the hard ones -- and with no
        # fallback that raises, which would kill the whole step.
        Qp, _ = solve_covering_lps(Wp, np.ascontiguousarray(Abar[rows]).T, Wp.sum(axis=0), sub,
                                   x_seed=np.zeros((rows.size, Kp)),
                                   live=np.abs(Wp).max(axis=0) > 0, workers=workers)
        obj = Qp @ Wp.sum(axis=0)
        adm = (Qp @ Wp.T >= Abar[rows] - 1.0e-12).all(axis=1)
        take = adm & (obj < best)
        if np.any(take):
            idx = rows[take]
            Q[idx] = 0.0
            Q[np.ix_(idx, np.arange(Kp))] = Qp[take]
            best[take] = obj[take]
            n_improved += int(take.sum())

    info.update(rescue_rows=int(rows.size), rescue_improved=int(n_improved))
    return n_improved


def _rescue_w_failed(Abar, Q, g, W, W0, failed, cfg, workers, info):
    """The W-side mirror: re-solve the failed CHANNELS on prefixes of Q. In place on W.

    One asymmetry with the Q side: the incumbent (the previous row of W) may itself be
    INADMISSIBLE, so an admissible prefix solution is accepted even when its objective is
    worse. An admissible point beats an inadmissible one, and the multiplicative repair
    charges real row sum for the inadmissible one.
    """
    chans = np.asarray(sorted(set(int(i) for i in failed)), dtype=int)
    K = Q.shape[1]
    if chans.size == 0:
        return 0

    sub = dataclasses.replace(cfg, cuts=False)
    n_improved = 0
    # Chunk the failed channels so that Abar[:, chunk] stays inside the block budget: a W-step
    # that fails EVERY channel would otherwise copy all of Abar.
    per = _block_rows(chans.size, Q.shape[0], cfg, arrays=1.0)
    for c0 in range(0, chans.size, per):
        cc = chans[c0:c0+per]
        Ac = np.ascontiguousarray(Abar[:, cc])                 # (nbeta, len(cc))
        adm_in = (Q @ W[cc].T >= Ac - 1.0e-12).all(axis=0)     # is the incumbent admissible?
        best = W[cc] @ g
        for Kp in cfg.rescue_ladder:
            if Kp >= K:
                continue
            Qp = np.ascontiguousarray(Q[:, :Kp])
            Wp, _ = solve_covering_lps(Qp, Ac, np.ascontiguousarray(g[:Kp]), sub,
                                       x_seed=np.zeros((cc.size, Kp)),
                                       live=np.abs(Qp).max(axis=0) > 0, workers=workers)
            obj = Wp @ g[:Kp]
            adm = (Qp @ Wp.T >= Ac - 1.0e-12).all(axis=0)
            take = adm & (~adm_in | (obj < best))
            if np.any(take):
                idx = cc[take]
                W[idx] = 0.0
                W[np.ix_(idx, np.arange(Kp))] = Wp[take]
                best[take] = obj[take]
                adm_in[take] = True
                n_improved += int(take.sum())
        del Ac

    info.update(w_rescue_chans=int(chans.size), w_rescue_improved=int(n_improved))
    return n_improved


####################################   the two steps   ####################################


def q_step(Abar, W, cfg=None, *, Q0=None, q_lower=None, workers=None, progress=False,
           repair=True, solve_fn=None, groups=None):
    """One Q-step: hold W fixed and choose the rows of Q optimally, one covering LP per group.

    The step is EXACT, not a heuristic: given W, no better Q exists. f is strictly increasing,
    so minimizing D over q_beta is exactly minimizing that group's row sum, and the groups are
    independent -- which also means the step is insensitive to the precise shape of f.

    'Abar' should already be scaled so that its maximum is ~1 (a power-of-two rescale is
    exact, and the equilibration's conditioning wants it).

    Parameters
    ----------
    Abar : ndarray
        (nbeta, nfreq) right-hand sides, one row per group.
    W : ndarray
        (nfreq, K) dictionary, held fixed.
    cfg : LpConfig or None
        None means LpConfig.for_qstep(), which is NOT the best configuration known -- see
        LpConfig. The W-step's config is a different one, not this one with a flag.
    Q0 : ndarray or None
        (nbeta, K) seed, returned for groups whose LP fails. None makes a failure raise.
    repair : bool
        Apply the admissibility repair before returning. True is the right default; False
        returns the RAW LP point, which is usually slightly inadmissible and is what to store
        when several repairs may be tried on one expensive solve. It is exactly sugar for
        setting the repair triple to (False, False, 'none'), so the two cannot disagree.
    solve_fn : callable or None
        An alternative to solve_covering_lps with the same signature -- the intended extension
        point for a different solver, a heuristic or a warm-start scheme, without
        reimplementing the plumbing that assembles the LPs, applies the repair and reports.
        The prefix rescue is NOT routed through it: it re-solves a handful of already-failed
        subproblems, and its job is to reproduce the exact solver's answer on them.
    groups : ndarray or None
        Solve only this subset of beta, keeping the rest of Q0, for cheap experiments and for
        manual parallelization: the rows of Q are independent given W, so slices combined
        afterwards give exactly the Q one process would have produced. Requires Q0 and
        repair=False, because the repair must run AFTER merging -- run per slice it would
        lose the step's own violation accounting.

    Returns
    -------
    tuple
        (Q, W, info). W is returned as well because a repair arm may scale the factor the step
        did not solve for, and returning it is how that stays visible rather than happening in
        place behind the caller's back.
    """
    cfg = LpConfig.for_qstep() if (cfg is None) else cfg
    if not repair:
        cfg = dataclasses.replace(cfg, additive_first=False, additive_last=False,
                                  rescale='none')
    if cfg.resolved_rescale('rows') == 'cols':
        raise RuntimeError("varmap.lp q_step: rescale='cols' would scale W, which this step"
                           " holds fixed. Use 'rows' (or 'auto'), or repair separately.")

    Abar = np.ascontiguousarray(np.asarray(Abar, dtype=np.float64))
    W = np.ascontiguousarray(np.asarray(W, dtype=np.float64))
    nbeta = Abar.shape[0]
    live = np.abs(W).max(axis=0) > 0            # a dictionary column that is identically zero

    qlo = None
    if (not cfg.nonneg) and (q_lower is not None):
        # The bound is in the group's own units, and so scale-free: q_beta is bounded below by
        # -q_lower times the magnitude the admissible one-hot seed needs for that group.
        ref = np.abs(Q0).max(axis=1) if (Q0 is not None) else np.ones(nbeta)
        qlo = -float(q_lower) * np.maximum(ref, 1.0e-300)

    sel = None
    if groups is not None:
        if Q0 is None:
            raise RuntimeError('varmap.lp q_step: groups= keeps the rest of Q, so Q0 is'
                               ' required')
        if repair:
            raise RuntimeError('varmap.lp q_step: groups= requires repair=False. The repair'
                               ' must run after the slices are merged; per slice it loses the'
                               " step's own violation accounting.")
        sel = np.asarray(groups, dtype=np.int64)

    B = Abar.T if (sel is None) else np.ascontiguousarray(Abar[sel]).T
    seed = Q0 if (sel is None) else np.ascontiguousarray(Q0[sel])
    lower = qlo if ((sel is None) or (qlo is None)) else qlo[sel]

    t0 = time.time()
    solver = solve_covering_lps if (solve_fn is None) else solve_fn
    Qs, lpinfo = solver(W, B, W.sum(axis=0), cfg, x_seed=seed, live=live, lower=lower,
                        workers=workers, progress=progress)
    # The clock excludes the (serial, one-off, amortized) constraint-pool warm-up, which is
    # timed separately.
    t0 += float(lpinfo.get('pool_seconds', 0.0))

    if sel is None:
        Q = Qs
    else:
        Q = np.array(Q0, dtype=np.float64, copy=True)
        Q[sel] = Qs

    info = dict(step='Q')
    info.update({k: v for k, v in lpinfo.items() if k != 'failed'})
    if (cfg.rescue == 'prefix') and lpinfo['failed'] and (sel is None):
        _rescue_q_failed(Abar, W, Q, lpinfo['failed'], cfg, workers, info)

    nneg, min_prod = check_nonneg(Q, W, None, cfg)
    if cfg.stash_raw:
        info['Q_raw'] = Q.copy()
    if sel is None:
        Q, W = _apply_repair(Abar, Q, W, cfg, 'rows', info)
    info.update(n_neg=nneg, min_prod=min_prod, seconds=time.time()-t0)
    return Q, W, info


def _majorizer(Q, s, y_true, labels, nbeta):
    """The W-step objective ``g_c = sum_alpha w_alpha Q[labels[alpha],c]``, with
    ``w_alpha = f'(y_approx/y_true)/y_true`` and w = 0 on the rows D does not score.

    THE FLOOR IS NOT OPTIONAL. An output with genuinely zero variance -- a Detrender2d with
    time half-width 0 annihilates the DM = 0 output exactly -- has y_true ~ 1e-14 rather than
    0 in floating point, and 1/y_true is then ~1e14: such a row would dominate this objective
    entirely, while contributing NOTHING to the distance, which ignores it by definition. So
    its weight is zero. This can only differ from an unfloored version on rows that D does not
    score, and on rows with y_true exactly 0 the unfloored version is not merely different but
    infinite.
    """
    scored = y_true >= YTRUE_FLOOR
    y_app = (Q @ s)[labels]
    w = np.zeros_like(y_true)
    ys = y_true[scored]
    w[scored] = fprime(y_app[scored] / ys) / ys                # > 0
    return Q.T @ np.bincount(labels, weights=w, minlength=nbeta)


def w_step(Abar, Q, y_true, labels, W0, cfg=None, *, pinned=None, workers=None,
           progress=False, repair=True, solve_fn=None, channels=None):
    """One W-step: hold Q fixed and choose the rows of W, one covering LP per channel.

    Needs a majorization, because the objective depends on W through column sums while the
    constraint depends on it elementwise, so it does not decouple as written. f is concave, so
    its tangent at the current iterate is a global UPPER bound; minimizing the tangent is a
    majorize-minimize step and cannot increase the true objective, and the majorizer IS linear
    in W and does decouple over channels. The tangent is taken at the FINE rows, which is why
    y_true and labels are needed and why the objective is assembled per group -- getting that
    accumulation wrong silently weights every group equally.

    Parameters
    ----------
    Abar : ndarray
        (nbeta, nfreq) right-hand sides, one COLUMN per subproblem.
    Q : ndarray
        (nbeta, K), held fixed.
    y_true : ndarray
        (nalpha,) true row sums, at FINE granularity.
    labels : ndarray
        (nalpha,) group index of each fine row.
    W0 : ndarray
        (nfreq, K) incumbent, and the fallback for channels whose LP fails. A failed W-step LP
        keeps the previous row, and there is no rescue on this side by default.
    pinned : sequence or None
        Column indices held FIXED: the LP is solved over the free columns only, with the
        pinned columns' contribution moved to the right-hand side as an offset.
    channels : ndarray or None
        Solve only this subset of channels; requires repair=False, for the same reason
        q_step's groups= does.

    Returns
    -------
    tuple
        (Q, W, info). Q is returned as well because this direction's 'rows' repair charges the
        violation to Q rather than to W -- see q_step.
    """
    cfg = LpConfig.for_wstep() if (cfg is None) else cfg
    if not repair:
        cfg = dataclasses.replace(cfg, additive_first=False, additive_last=False,
                                  rescale='none')

    Abar = np.asarray(Abar, dtype=np.float64)
    Q = np.ascontiguousarray(np.asarray(Q, dtype=np.float64))
    W0 = np.ascontiguousarray(np.asarray(W0, dtype=np.float64))
    labels = np.asarray(labels, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.float64)
    nbeta, nfreq = Abar.shape

    g = _majorizer(Q, W0.sum(axis=0), y_true, labels, nbeta)

    # Split the dictionary into pinned (excluded from this step) and free columns. pinned=None
    # gives free = every column, and everything below is then the unpinned expression verbatim.
    K = W0.shape[1]
    pinmask = np.zeros(K, dtype=bool)
    if pinned is not None:
        pinmask[np.asarray(pinned, dtype=np.int64)] = True
    freec = np.flatnonzero(~pinmask)
    Qf = Q if (not pinmask.any()) else np.ascontiguousarray(Q[:, freec])
    live = np.abs(Qf).max(axis=0) > 0                      # a column no group uses

    sel = None
    if channels is not None:
        if repair:
            raise RuntimeError('varmap.lp w_step: channels= requires repair=False, for the'
                               " same reason q_step's groups= does.")
        sel = np.asarray(channels, dtype=np.int64)

    # A transposed COPY makes each channel's right-hand side a contiguous read; the strided
    # view has identical values and is 172 GiB cheaper at CHORD scale.
    src = np.ascontiguousarray(Abar.T).T if cfg.w_step_transpose else Abar
    B = src if (sel is None) else np.ascontiguousarray(Abar[:, sel])
    seed = W0[:, freec] if pinmask.any() else W0
    seed = np.ascontiguousarray(seed if (sel is None) else seed[sel])
    offs = None if (not pinmask.any()) else (np.ascontiguousarray(Q[:, pinmask]),
                                             np.ascontiguousarray(W0[:, pinmask])
                                             if (sel is None) else
                                             np.ascontiguousarray(W0[np.ix_(sel, pinmask)]))

    t0 = time.time()
    solver = solve_covering_lps if (solve_fn is None) else solve_fn
    Wfree, lpinfo = solver(Qf, B, (g if not pinmask.any() else np.ascontiguousarray(g[freec])),
                           cfg, x_seed=seed, live=live, offset_factors=offs, workers=workers,
                           progress=progress)
    t0 += float(lpinfo.get('pool_seconds', 0.0))

    W = np.array(W0, dtype=np.float64, copy=True)
    if sel is None:
        W[:, freec] = Wfree
    else:
        W[np.ix_(sel, freec)] = Wfree
    del Wfree

    info = dict(step='W')
    info.update({k: v for k, v in lpinfo.items() if k != 'failed'})
    wfailed = lpinfo['failed']
    if (cfg.rescue == 'prefix') and wfailed and (sel is None):
        if pinmask.any():
            # The prefix rescue re-solves on a PREFIX of Q's columns, which is not well
            # defined once some columns are pinned out of this LP -- and it is a measured dead
            # end on this side anyway (0/387 channels improved), so it declines rather than
            # growing an untested second path.
            if cfg.w_fail_warn:
                print('  varmap.lp w_step: the prefix rescue is not supported with pinned'
                      ' columns; skipping it (it improved 0/387 channels when it did run).',
                      flush=True)
        else:
            _rescue_w_failed(Abar, Q, g, W, W0, wfailed, cfg, workers, info)

    # The step's LP objective is sum_F g.W[F,:] = g.s, and W0 is a feasible point of every
    # channel LP, so a correctly solved W-step CANNOT increase g.s -- and the majorization
    # then gives a non-increasing D. When D goes up anyway, that chain was broken either by
    # the solver or by the repair, and it is measured to be the repair. These are O(K).
    info['w_obj_before'] = float(g @ W0.sum(axis=0))
    info['w_obj_raw'] = float(g @ W.sum(axis=0))

    guard = None
    if cfg.w_guard or cfg.w_diag:
        guard = _w_channel_stats(Abar, Q, W, None, cfg,
                                 thresholds=((1.0 + 1.0e-9, '1e-9'), (1.01, '1pc'), (2.0, '2x')))
        rc0 = guard['rc']
        info.update(w_raw_max_r=float(np.nanmax(rc0)),
                    w_chan_frac_viol=float((rc0 > 1.0 + 1.0e-9).mean()),
                    w_entry_frac_viol=guard['n_entry_viol'] / max(1, guard['n_entry']))
        for k in ('w_chan_viol_1e-9', 'w_chan_viol_1pc', 'w_chan_viol_2x'):
            info[k] = guard[k]

    if cfg.w_guard:
        # Restore the majorize-minimize guarantee channel by channel; see LpConfig.w_guard.
        rc, nonpos = guard['rc'], guard['nonpos']
        sc = np.maximum(rc, 1.0) * (1.0 + cfg.repair_margin)
        Wtry = W * sc[:, None]
        obj_new, obj_old = Wtry @ g, W0 @ g
        take = np.isfinite(obj_new) & (obj_new <= obj_old) & (~nonpos)
        W = np.ascontiguousarray(np.where(take[:, None], Wtry, W0))
        del Wtry
        info.update(w_guard_reverted=int((~take).sum()),
                    w_guard_reverted_nonpos=int(nonpos.sum()),
                    w_guard_reverted_obj=int((np.isfinite(obj_new) & (obj_new > obj_old)).sum()),
                    w_guard_frac_reverted=float((~take).sum() / max(1, nfreq)),
                    w_guard_scale_max=(float(np.nanmax(sc[take])) if np.any(take) else 1.0))

    nneg, min_prod = check_nonneg(Q, W, None, cfg)
    if sel is None:
        Q, W = _apply_repair(Abar, Q, W, cfg, 'cols', info)

    if lpinfo['n_failed'] and cfg.w_fail_warn:
        print(f'  varmap.lp w_step: {lpinfo["n_failed"]}/{nfreq} channel LPs FAILED and fell'
              f' back to the previous row of W; status {info.get("status")}', flush=True)

    info.update(n_neg=nneg, min_prod=min_prod, seconds=time.time()-t0,
                n_g_neg=int((g[freec] < 0).sum()), n_pinned=int(pinmask.sum()),
                w_obj_after=float(g @ W.sum(axis=0)),
                w_failed=int(lpinfo['n_failed']),
                w_failed_frac=float(lpinfo['n_failed'])/max(1, nfreq),
                w_failed_chans=[int(i) for i in wfailed[:64]])
    b0 = info['w_obj_before']
    info['w_obj_ratio'] = (info['w_obj_after'] / b0) if (b0 != 0.0) else float('nan')
    info['w_obj_ratio_raw'] = (info['w_obj_raw'] / b0) if (b0 != 0.0) else float('nan')
    if pinmask.any():
        # The 'cols' repair scales whole rows of W, i.e. channels, by a factor >= 1, so a
        # pinned column can only be scaled UP: it stays nonnegative and still dominates every
        # group, and every guarantee the pin buys survives. This is how far it moved.
        info['pin_drift'] = float(np.max(W[:, pinmask] / np.maximum(W0[:, pinmask], 1e-300)))
    return Q, W, info


####################################   assembling the arrays   ####################################
#
# Two conveniences for callers holding VarianceMaps rather than bare arrays. Everything above
# takes arrays and never learns where they came from, which is what keeps this module free of
# file I/O and of any dependency on how a map is stored.


def covering_lp_data(vmap, ref, ibeta, cfg=None):
    """(cost, M, b) for one group's Q-step LP -- the raw arrays, for inspection, for a
    hand-written solver, or for a unit test.

    'b' may be READ-ONLY: with no clipping it is a view into the reference map. Copy it before
    modifying it.
    """
    if not vmap.is_factored:
        raise RuntimeError('varmap.lp covering_lp_data: the Q-step needs a factored map (its'
                           ' W is the constraint matrix)')
    W = np.asarray(vmap.W, dtype=np.float64)
    b = ref.rows(int(ibeta), int(ibeta) + 1)[0]
    if cfg is not None:
        b = _clip_rhs(b, float(cfg.clip_rel))
    return W.sum(axis=0), W, b


def majorizer_weights(vmap, ref):
    """The length-K objective vector g of the W-step, ``g_c = sum_alpha w_alpha Q[alpha,c]``
    with ``w_alpha = f'(y_approx/y_true) / y_true`` at the current iterate.

    Note the sum is over FINE alpha with Q row-duplicated, so it is accumulated per group and
    then contracted with Q -- ``sum_beta (sum over alpha in beta of w_alpha) Q[beta,c]``. That
    per-group accumulation is the whole trick, and getting it wrong silently weights every
    group equally.
    """
    if not vmap.is_factored:
        raise RuntimeError('varmap.lp majorizer_weights: needs a factored map')
    if ref.y_true is None:
        raise RuntimeError('varmap.lp majorizer_weights: the reference map has no y_true, so'
                           ' the majorization has nothing to linearize about')

    Q = np.asarray(vmap.Q, dtype=np.float64)
    nalpha = vmap.nalpha
    y_true = np.asarray(ref.y_true, dtype=np.float64)

    # The labels are assembled blockwise to bound the temporaries, but the reduction is a
    # SINGLE bincount, so the accumulation order does not depend on the block size.
    labels = np.empty(nalpha, dtype=np.int64)
    step = vmap._ALPHA_BLOCK
    for lo in range(0, nalpha, step):
        hi = min(lo + step, nalpha)
        labels[lo:hi] = vmap.alpha_to_beta_block(lo, hi)

    mid = None if (vmap.mid is None) else np.asarray(vmap.mid, dtype=np.float64)
    s = np.asarray(vmap.W, dtype=np.float64).sum(axis=0) if (mid is None) \
        else (mid @ np.asarray(vmap.W, dtype=np.float64).sum(axis=0))
    return _majorizer(Q, s, y_true, labels, vmap.nbeta)
