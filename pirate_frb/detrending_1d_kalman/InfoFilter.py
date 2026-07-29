"""
The two information-filter recursions of the fixed-lag detrender (see
plans/detrend_1d_kalman.md, sections "Forward information filter" and "Backward
information filter", and notes/tree_dedispersion.tex,
section "Time detrending algorithm 2: Kalman filter", eq (kf_fwd) and (kf_bwd)).

A Gaussian is carried in *information* form: the pair (J, eta) represents

    p(x) ~ exp(-x^T J x / 2 + eta^T x),

so J is the precision and eta = J mu.  Three properties of this form are what the
whole design rests on, and none of them holds in covariance form:

  - the diffuse prior is exactly (J, eta) = (0, 0), with no large-kappa fudge, which
    is what makes polynomial reproduction exact rather than approximate;
  - the measurement update is a pure addition;
  - the backward factor is a likelihood rather than a distribution, so it is allowed
    to be rank-deficient -- which it is whenever the future window holds fewer than
    k valid samples.

The two recursions are mirror images: measure, rank-one downdate, then a congruence
by A^-T going forwards or A^T going backwards.  Both are division-safe: the only
divide is by beta = g^T (.) g + 1/q with the left term >= 0 by positive
semidefiniteness, so beta >= 1/q > 0 always.  There is no guarded divide and no
empty-set rule anywhere here, in contrast to the moment monoid of detrending_1d.

Both entry points are batched over arbitrary leading axes, which is what lets the
same code serve the forward pass (batch (S,), sequential in t) and the vectorized
backward pass (batch (S,T), sequential in lag).
"""

import numpy as np


def _measure(J, eta, m, d, k):
    """
    Absorb one observation: J += m e_0 e_0^T, eta += m d e_0.

    Masked samples are SELECTED away rather than multiplied by m.  A masked sample
    may hold anything at all -- a dropped packet can leave NaN or Inf behind -- and
    0*inf is NaN, which would poison the carried state permanently.  This is the
    same discipline as detrending_1d, and test_masked_data_unused() checks it by
    bit-identity under poisoning.
    """
    mf = (m != 0)
    Jn = J.copy()
    en = eta.copy()
    Jn[..., 0, 0] = Jn[..., 0, 0] + np.where(mf, 1, 0).astype(J.dtype)
    en[..., 0] = en[..., 0] + np.where(mf, d, 0).astype(eta.dtype)
    return Jn, en


def _downdate(J, eta, invq, k):
    """
    Marginalize out the process noise: the rank-one step common to both directions.

        beta = J_gg + 1/q,   J <- J - (Jg)(Jg)^T / beta,   eta <- eta - Jg (eta_g)/beta

    with g = e_(k-1).  In exact arithmetic this preserves positive semidefiniteness
    (it is a Schur complement of a PSD matrix) and preserves rank exactly, since
    J - Jgg^TJ/beta = J^(1/2) (I - vv^T/beta) J^(1/2) with |v|^2 = J_gg < beta.
    """
    beta = J[..., k-1, k-1] + invq                 # >= 1/q > 0
    Jg = J[..., :, k-1]                            # J g
    Jn = J - Jg[..., :, None] * Jg[..., None, :] / beta[..., None, None]
    en = eta - Jg * (eta[..., k-1] / beta)[..., None]
    return Jn, en


def forward_step(J, eta, m, d, model):
    """
    One (measurement, predict) pair of the forward filter.

    Returns (J_post, eta_post, J_next, eta_next): the state *after* the measurement
    at this sample, which is the factor the per-output combine needs, and the state
    propagated to the next sample, which is what the loop carries.

    J has shape (..., k, k) and eta (..., k); m and d have the batch shape.
    """
    k = model.k
    Jm, em = _measure(J, eta, m, d, k)

    # Predict.  x[t+1] = A x[t] + g w, so in information form the congruence is by
    # A^-T (M = A^-T J A^-1, zeta = A^-T eta) followed by the noise downdate.  Note
    # eta @ Ainv is A^-T eta, which avoids materializing a transpose.
    M = model.Ainv.T @ Jm @ model.Ainv
    zeta = em @ model.Ainv
    Jn, en = _downdate(M, zeta, model.invq, k)
    return Jm, em, Jn, en


def backward_step(J, eta, m, d, model):
    """
    One step of the backward information filter, absorbing the sample (m, d) at u+1
    into a likelihood on x[u].  (J, eta) in and out represent
    p(d[u+1 .. b] | x[u]) as an unnormalized Gaussian in x[u].

    The exact mirror of forward_step(): measure, downdate, then a congruence by A^T
    instead of A^-T.  Initialize with (0, 0) at u = b.
    """
    k = model.k
    Jt, et = _measure(J, eta, m, d, k)
    Jh, eh = _downdate(Jt, et, model.invq, k)
    return model.A.T @ Jh @ model.A, eh @ model.A
