"""Point identified estimation: GMM for the moment approach, least squares for the regression approach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2

from pymte.ivlike import MomentSet


@dataclass(frozen=True)
class GMMResult:
    """Two-step GMM estimates of the MTR coefficients.

    Attributes
    ----------
    theta : numpy.ndarray
        Stacked coefficients ``(theta0, theta1)``.
    moments : numpy.ndarray
        Sample moment conditions ``beta - Gamma @ theta`` after dropping
        redundant moments.
    j_stat, j_df, j_pvalue : float or None
        Hansen J statistic, its degrees of freedom and asymptotic p-value
        (``None`` when the model is exactly identified).
    redundant : tuple of int
        Positions of moments dropped because of collinearity.
    """

    theta: NDArray[np.float64]
    moments: NDArray[np.float64]
    j_stat: float | None
    j_df: int | None
    j_pvalue: float | None
    redundant: tuple[int, ...]


def _moment_matrix(mom: MomentSet, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-observation moments ``s_k Y`` and ``[g0, g1]`` zero-filled outside each subset.

    Following the R package, moments defined on a subset are averaged over
    the full sample size ``n``.
    """
    s = mom.n_moments
    j = mom.gamma0.shape[1] + mom.gamma1.shape[1]
    ys = np.zeros((n, s))
    gs = np.zeros((n, s, j))
    for k in range(s):
        rows = mom.rows[k]
        ys[rows, k] = mom.ys[k]
        gs[rows, k, :] = np.hstack([mom.gamma0_obs[k], mom.gamma1_obs[k]])
    return ys, gs


def gmm(
    mom: MomentSet,
    n: int,
    *,
    identity_weight: bool = False,
    center: NDArray[np.float64] | None = None,
    redundant: tuple[int, ...] | None = None,
) -> GMMResult:
    """Estimate the MTR coefficients by two-step GMM.

    Parameters
    ----------
    mom : MomentSet
        IV-like moment conditions.
    n : int
        Sample size.
    identity_weight : bool, default False
        Use the identity weighting matrix (one-step GMM). The default is the
        efficient two-step estimator.
    center : numpy.ndarray, optional
        Moment recentering used in the bootstrap (the original sample's
        moment residuals).
    redundant : tuple of int, optional
        Moments to drop; detected automatically when ``None``.

    Returns
    -------
    GMMResult
    """
    ys, gs = _moment_matrix(mom, n)
    s = mom.n_moments
    xmat = gs.mean(axis=0)
    ymat = ys.mean(axis=0)
    j = xmat.shape[1]
    if s < j:
        raise ValueError(
            f"GMM system is underidentified: {j} MTR coefficients but only {s} moment conditions"
        )
    if np.linalg.matrix_rank(xmat) < j:
        raise ValueError(
            f"GMM system is underidentified: the {s} moment conditions identify fewer than "
            f"{j} MTR coefficients. Adjust the IV-like specifications or m0 and m1."
        )
    if redundant is None:
        # Drop moments whose residuals are collinear, as the R package does.
        rng = np.random.default_rng(0)
        resid = ys - gs @ rng.normal(size=j)
        omega = resid.T @ resid / n
        drop: list[int] = []
        keep = list(range(s))
        while True:
            w, v = np.linalg.eigh(omega[np.ix_(keep, keep)])
            small = np.flatnonzero(np.abs(w) < 1e-8)
            if not small.size:
                break
            loading = np.abs(v[:, small[0]])
            worst = max(k for k in range(len(keep)) if loading[k] > 1e-8)
            drop.append(keep.pop(worst))
        redundant = tuple(sorted(drop))
    keep_idx = [k for k in range(s) if k not in redundant]
    xk, yk = xmat[keep_idx], ymat[keep_idx]
    cen = np.zeros(len(keep_idx)) if center is None else center
    theta = np.linalg.lstsq(xk, yk - cen, rcond=None)[0]
    omega_inv = None
    if not identity_weight:
        resid = ys[:, keep_idx] - gs[:, keep_idx, :] @ theta
        omega = resid.T @ resid / n
        omega_inv = np.linalg.pinv(omega)
        a = xk.T @ omega_inv @ xk
        theta = np.linalg.solve(a, xk.T @ omega_inv @ (yk - cen))
    moments = yk - xk @ theta
    j_stat = j_df = j_p = None
    if len(keep_idx) > j:
        m = moments - cen
        w_mat = omega_inv if omega_inv is not None else np.eye(len(keep_idx))
        j_stat = float(n * m @ w_mat @ m)
        j_df = len(keep_idx) - j
        j_p = float(chi2.sf(j_stat, j_df))
    return GMMResult(theta, moments, j_stat, j_df, j_p, tuple(redundant))


def least_squares(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    equal: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Ordinary or equality-constrained least squares.

    Parameters
    ----------
    x : numpy.ndarray
        Design matrix (full column rank).
    y : numpy.ndarray
        Outcome.
    equal : numpy.ndarray, optional
        Constraint rows ``equal @ theta == 0``.

    Returns
    -------
    numpy.ndarray
        Coefficients.
    """
    xtx, xty = x.T @ x, x.T @ y
    if equal is None or len(equal) == 0:
        return np.asarray(np.linalg.solve(xtx, xty), dtype=float)
    # Solve the KKT system of the constrained problem.
    e = len(equal)
    kkt = np.block([[xtx, equal.T], [equal, np.zeros((e, e))]])
    rhs = np.concatenate([xty, np.zeros(e)])
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return np.asarray(sol[: x.shape[1]], dtype=float)
