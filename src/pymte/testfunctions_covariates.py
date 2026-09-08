"""Closed-form quantities for the covariate test population.

These functions mirror ``testfunctions_covariates.R`` of the R package.
They compute, from the population distribution of :func:`pymte.testdata.gendist_covariates`,
the IV-like weights and the ``Gamma`` moments of the MTR specification
``m0 = ~ x1 + x2:u + x2:I(u^2)``, ``m1 = ~ x1 + x1:x2 + u + x1:u + x2:I(u^2)``
by hand, so that the test suite can compare them with the estimator.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def popmean(
    exprs: Sequence[str], distribution: pd.DataFrame, density: str = "f"
) -> dict[str, float]:
    """Compute population means of expressions, weighted by the cell probabilities.

    Parameters
    ----------
    exprs : sequence of str
        Column expressions for :meth:`pandas.DataFrame.eval`, e.g. ``"ey * p"``.
    distribution : pandas.DataFrame
        One row per cell with the probability column ``density``.
    density : str, default "f"
        Name of the probability column (normalised to sum to one).

    Returns
    -------
    dict
        Expression to its population mean.
    """
    w = distribution[density].to_numpy(dtype=float)
    w = w / w.sum()
    return {e: float(np.asarray(distribution.eval(e), dtype=float) @ w) for e in exprs}


def symat(values: Sequence[float]) -> NDArray[np.float64]:
    """Symmetric matrix from its lower triangle listed column by column (R's ``lower.tri``)."""
    k = len(values)
    n = int(round(0.5 * (-1 + np.sqrt(1 + 8 * k))))
    m = np.zeros((n, n))
    pos = 0
    for j in range(n):
        for i in range(j, n):
            m[i, j] = m[j, i] = values[pos]
            pos += 1
    return m


def m_int(ub: ArrayLike, lb: ArrayLike, coef: Sequence[float]) -> NDArray[np.float64]:
    """Integral of the quadratic ``coef[0] + coef[1] u + coef[2] u^2`` from ``lb`` to ``ub``."""
    ub_, lb_ = np.asarray(ub, dtype=float), np.asarray(lb, dtype=float)
    out = coef[0] * (ub_ - lb_) + coef[1] / 2 * (ub_**2 - lb_**2) + coef[2] / 3 * (ub_**3 - lb_**3)
    return np.asarray(out, dtype=float)


def gen_gamma_tt(
    data: pd.DataFrame,
    s0: str,
    s1: str,
    lb: ArrayLike | None = None,
    ub: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``Gamma`` moments of the covariate MTR specification for weight columns ``s0``, ``s1``.

    Without ``lb``/``ub`` the integrals run over ``[p, 1]`` for ``m0`` and
    ``[0, p]`` for ``m1`` (IV-like estimands); with them, over ``[lb, ub]``
    for both arms (target parameters).
    """
    p = data["p"].to_numpy(dtype=float)
    x1 = data["x1"].to_numpy(dtype=float)
    x2 = data["x2"].to_numpy(dtype=float)
    w0 = data[s0].to_numpy(dtype=float)
    w1 = data[s1].to_numpy(dtype=float)
    f = data["f"].to_numpy(dtype=float)
    f = f / f.sum()
    if lb is None or ub is None:
        lo0, hi0, lo1, hi1 = p, np.ones_like(p), np.zeros_like(p), p
    else:
        lo0 = lo1 = np.broadcast_to(np.asarray(lb, dtype=float), len(p))
        hi0 = hi1 = np.broadcast_to(np.asarray(ub, dtype=float), len(p))
    g0 = np.column_stack(
        [
            (hi0 - lo0) * w0,
            (hi0 - lo0) * x1 * w0,
            m_int(hi0, lo0, [0, 1, 0]) * x2 * w0,
            m_int(hi0, lo0, [0, 0, 1]) * x2 * w0,
        ]
    )
    g1 = np.column_stack(
        [
            (hi1 - lo1) * w1,
            (hi1 - lo1) * x1 * w1,
            m_int(hi1, lo1, [0, 1, 0]) * w1,
            (hi1 - lo1) * x1 * x2 * w1,
            m_int(hi1, lo1, [0, 1, 0]) * x1 * w1,
            m_int(hi1, lo1, [0, 0, 1]) * x2 * w1,
        ]
    )
    return f @ g0, f @ g1


def s_ols1d(d: int, exx: NDArray[np.float64]) -> float:
    """OLS weight on ``d`` in ``y ~ d`` (``exx`` is ``E[XX']`` for ``X = (1, d, ...)``)."""
    return float(np.linalg.solve(exx[:2, :2], [1.0, float(d)])[1])


def s_ols2d(x: float, d: int, exx: NDArray[np.float64]) -> float:
    """OLS weight on ``d`` in ``y ~ d + x1`` at covariate value ``x``."""
    return float(np.linalg.solve(exx[:3, :3], [1.0, float(d), float(x)])[1])


def s_ols3(x: Sequence[float], d: int, j: int, exx: NDArray[np.float64]) -> float:
    """OLS weight on coefficient ``j`` (0-based) in ``y ~ d + x1 + x2`` at covariates ``x``."""
    return float(np.linalg.solve(exx, [1.0, float(d), *map(float, x)])[j])


def s_tsls(z: Sequence[float], j: int, exz: NDArray[np.float64], pi: NDArray[np.float64]) -> float:
    """TSLS weight on coefficient ``j`` (0-based) at instrument values ``z``."""
    return float(np.linalg.solve(pi @ exz.T, pi @ np.array([1.0, *map(float, z)]))[j])


def s_wald(z: float, p_to: float, p_from: float, e_to: float, e_from: float) -> float:
    """Wald weight for the instrument moving from ``z2 = 2`` to ``z2 = 3``."""
    return ((z == 3) / p_to - (z == 2) / p_from) / (e_to - e_from)
