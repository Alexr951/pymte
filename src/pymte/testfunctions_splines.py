"""Closed-form quantities for the spline test population.

These functions mirror ``testfunctions_splines.R`` of the R package. They
compute, from the population distribution of :func:`pymte.testdata.gendist_splines`,
the IV-like weights and the ``Gamma`` moments of the specification
``m1 = ~ x + uSplines(degree=2, knots=(0.3, 0.6))`` and
``m0 = ~ 0 + x:uSplines(degree=0, knots=(0.2, 0.5, 0.8), intercept=True) +
uSplines(degree=1, knots=(0.4,), intercept=True) + I(u^2)`` by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline


def _bspline(knots: Sequence[float], degree: int) -> BSpline:
    # The basis of splines2::bSpline with boundary knots 0 and 1, built
    # directly from scipy so that these helpers do not depend on the package.
    t = np.r_[np.zeros(degree + 1), np.sort(knots), np.ones(degree + 1)]
    return BSpline(t, np.eye(len(t) - degree - 1), degree, extrapolate=False)


def spline_basis(
    u: ArrayLike, knots: Sequence[float], degree: int, intercept: bool = False
) -> NDArray[np.float64]:
    """B-spline basis with boundary knots 0 and 1 evaluated at ``u`` (``splines2::bSpline``)."""
    x = np.asarray(u, dtype=float)
    out = np.asarray(_bspline(knots, degree)(np.where(x >= 1.0, np.nextafter(1.0, 0.0), x)))
    return out if intercept else out[:, 1:]


def spline_int(
    ub: ArrayLike, lb: ArrayLike, knots: Sequence[float], degree: int, intercept: bool = False
) -> NDArray[np.float64]:
    """Integrals of the B-spline basis with boundary knots 0 and 1 from ``lb`` to ``ub``."""
    anti = _bspline(knots, degree).antiderivative()
    lo, hi = np.broadcast_arrays(np.atleast_1d(lb).astype(float), np.atleast_1d(ub).astype(float))
    out = np.asarray(anti(hi) - anti(lo))
    return out if intercept else out[:, 1:]


def s_ols_splines(x: ArrayLike | None, d: int, j: int, exx: NDArray[np.float64]) -> float:
    """OLS weight on coefficient ``j`` (0-based) in ``y ~ d`` (``x=None``) or ``y ~ d + x``."""
    reg = [1.0, float(d)] if x is None else [1.0, float(d), *np.ravel(x).astype(float)]
    return float(np.linalg.solve(exx, reg)[j])


def s_tsls_splines(
    z: Sequence[float], d: int, j: int, exz: NDArray[np.float64], pi: NDArray[np.float64]
) -> float:
    """TSLS weight on coefficient ``j`` (0-based) at instrument values ``z``."""
    return float(np.linalg.solve(pi @ exz.T, pi @ np.array([1.0, *map(float, z)]))[j])


def w_att_splines(z: Any, d: int, ed: float) -> float:
    """ATT weight ``1 / P(D = 1)``."""
    return 1.0 / ed


def gen_gamma_splines_tt(
    distr: pd.DataFrame,
    weight: Callable[..., float],
    u1s1: NDArray[np.float64],
    u0s1: NDArray[np.float64],
    u0s2: NDArray[np.float64],
    zvars: Sequence[str] | None = None,
    target: bool = False,
    **kwargs: Any,
) -> tuple[pd.Series, pd.Series]:
    """``Gamma`` moments of the spline specification for a weight function.

    Parameters
    ----------
    distr : pandas.DataFrame
        Population distribution with columns ``x``, ``p`` and ``f``.
    weight : callable
        ``weight(z, d, **kwargs)`` giving the weight of one cell; ``z`` is
        the row of ``zvars`` (``None`` when no ``zvars`` are given).
    u1s1, u0s1, u0s2 : numpy.ndarray
        Spline integrals per cell for the treated spline and the two
        untreated splines.
    zvars : sequence of str, optional
        Columns passed to ``weight``.
    target : bool, default False
        Integrate ``u^2`` over ``[0, p]`` (target) instead of ``[p, 1]``.
    **kwargs
        Passed on to ``weight``.

    Returns
    -------
    tuple of pandas.Series
        ``g0`` and ``g1`` named as the R package names the coefficients.
    """
    rows: list[Any]
    if zvars is None:
        rows = [None] * len(distr)
    else:
        rows = [np.asarray(r, dtype=float) for r in distr[list(zvars)].to_numpy()]
    s0 = np.array([weight(z, d=0, **kwargs) for z in rows])
    s1 = np.array([weight(z, d=1, **kwargs) for z in rows])
    x = distr["x"].to_numpy(dtype=float)
    p = distr["p"].to_numpy(dtype=float)
    f = distr["f"].to_numpy(dtype=float)
    mu0s1 = u0s1 * x[:, None] * s0[:, None]
    mu0s2 = u0s2 * s0[:, None]
    muu2 = (1 / 3) * ((p**3) if target else (1 - p**3)) * s0
    g0 = np.concatenate([[muu2 @ f], f @ mu0s2, f @ mu0s1])
    g0_names = ["I(u^2)", "u0S1.1:1", "u0S1.2:1", "u0S1.3:1"] + [
        f"u0S2.{b}:x" for b in range(1, 5)
    ]
    m1int = s1 * p
    m1x = x * s1 * p
    mu1s1 = u1s1 * s1[:, None]
    g1 = np.concatenate([[m1int @ f, m1x @ f], f @ mu1s1])
    g1_names = ["(Intercept)", "x"] + [f"u1S1.{b}:1" for b in range(1, 5)]
    return pd.Series(g0, index=g0_names), pd.Series(g1, index=g1_names)
