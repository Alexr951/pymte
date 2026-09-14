"""Target parameter weights.

Every conventional target parameter of the form
``E[ int_0^1 ( w_1(u, X, Z) m_1(u, X) + w_0(u, X, Z) m_0(u, X) ) du ]`` is
represented by a lower limit, an upper limit and a multiplier per
observation: the ``m1`` weight is ``multiplier * 1{lb <= u <= ub}`` and the
``m0`` weight is its negative. The ``w*1`` functions below return these
interval weights for the ATE, ATT, ATU, LATE and generalised LATE, following
the functions of the same names in the R package; :func:`gen_weight`
evaluates the pieces of a custom weight. Integrating the MTR bases against
the weights is done by :func:`pymte.mst.gen_target`.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pymte.propensity import Propensity

TARGETS = ("ate", "att", "atu", "late", "avglate", "genlate")


@dataclass(frozen=True)
class TargetWeights:
    """Weights of a conventional target parameter, one interval and multiplier per row.

    Attributes
    ----------
    lb, ub, mult : numpy.ndarray
        Lower limit, upper limit and multiplier per observation.
    rows : numpy.ndarray of bool
        Rows over which the target expectation is taken (all rows unless the
        target conditions on covariate values through ``late_x``).
    """

    lb: NDArray[np.float64]
    ub: NDArray[np.float64]
    mult: NDArray[np.float64]
    rows: NDArray[np.bool_]


def late_rows(data: pd.DataFrame, late_x: dict[str, Any] | None) -> NDArray[np.bool_]:
    """Rows matching the covariate values in ``late_x`` (all rows when it is empty)."""
    rows = np.ones(len(data), dtype=bool)
    if late_x:
        for name, value in late_x.items():
            rows &= np.asarray(data[name] == value)
        if not rows.any():
            raise ValueError(f"No observations satisfy the 'late_x' condition {late_x}")
    return rows


def wate1(n: int) -> TargetWeights:
    """Weights of the average treatment effect: the whole unit interval, multiplier 1."""
    return TargetWeights(np.zeros(n), np.ones(n), np.ones(n), np.ones(n, dtype=bool))


def watt1(p: NDArray[np.float64], d: NDArray[np.float64]) -> TargetWeights:
    """Weights of the ATT: ``[0, p_i]`` with multiplier ``1 / P(D = 1)``."""
    n = len(p)
    return TargetWeights(np.zeros(n), p, np.full(n, 1.0 / d.mean()), np.ones(n, dtype=bool))


def watu1(p: NDArray[np.float64], d: NDArray[np.float64]) -> TargetWeights:
    """Weights of the ATU: ``[p_i, 1]`` with multiplier ``1 / P(D = 0)``."""
    n = len(p)
    return TargetWeights(p, np.ones(n), np.full(n, 1.0 / (1.0 - d.mean())), np.ones(n, dtype=bool))


def wlate1(
    prop: Propensity,
    data: pd.DataFrame,
    late_from: dict[str, Any],
    late_to: dict[str, Any],
    late_x: dict[str, Any] | None = None,
    avglate: bool = False,
) -> TargetWeights:
    """Weights of the LATE between two instrument values.

    Parameters
    ----------
    prop : Propensity
        Fitted propensity score model (a supplied column cannot be used).
    data : pandas.DataFrame
        Estimation sample.
    late_from, late_to : dict
        Instrument values, e.g. ``{"z": 1}`` and ``{"z": 3}``.
    late_x : dict, optional
        Covariate values to condition on, e.g. ``{"x": 2}``.
    avglate : bool, default False
        Use the pointwise multiplier ``1 / |p_i(to) - p_i(from)|`` (the R
        target ``avglate``) instead of ``1 / |E[p(to)] - E[p(from)]|``.

    Returns
    -------
    TargetWeights
    """
    if not prop.fitted:
        raise ValueError("Target parameters 'late' and 'avglate' require a propensity score model")
    n = len(data)
    rows = late_rows(data, late_x)
    lb = prop.predict(data.assign(**late_from, **(late_x or {})), clip=False)
    ub = prop.predict(data.assign(**late_to, **(late_x or {})), clip=False)
    if (lb < 0).any() or (ub < 0).any():
        warnings.warn("Propensity scores below 0 set to 0.", stacklevel=2)
    if (lb > 1).any() or (ub > 1).any():
        warnings.warn("Propensity scores greater than 1 set to 1.", stacklevel=2)
    lb, ub = np.clip(lb, 0.0, 1.0), np.clip(ub, 0.0, 1.0)
    width = np.abs(ub - lb) if avglate else np.full(n, abs(ub[rows].mean() - lb[rows].mean()))
    if (width == 0).any():
        raise ValueError(
            "The propensity scores at 'late_from' and 'late_to' coincide after clipping to "
            "[0, 1], so the LATE interval is empty. Choose instrument values whose propensity "
            "scores differ or use a link that keeps the scores inside (0, 1)."
        )
    mult = 1.0 / width
    return TargetWeights(lb, ub, mult, rows)


def wgenlate1(
    data: pd.DataFrame, lb: float, ub: float, late_x: dict[str, Any] | None = None
) -> TargetWeights:
    """Weights of the generalised LATE over ``[lb, ub]`` in ``u``, optionally conditional on ``late_x``."""
    if not 0 <= lb < ub <= 1:
        raise ValueError("'genlate_lb' and 'genlate_ub' must satisfy 0 <= lb < ub <= 1")
    n = len(data)
    return TargetWeights(
        np.full(n, lb), np.full(n, ub), np.full(n, 1.0 / (ub - lb)), late_rows(data, late_x)
    )


Piece = float | int | Callable[..., float]


def gen_weight(data: pd.DataFrame, fun: Piece) -> NDArray[np.float64]:
    """Evaluate one piece of a custom weight or knot, row by row.

    Parameters
    ----------
    data : pandas.DataFrame
        Estimation sample.
    fun : number or callable
        A constant, or a function whose argument names are columns of
        ``data``.

    Returns
    -------
    numpy.ndarray
        One value per row.
    """
    n = len(data)
    if isinstance(fun, int | float):
        return np.full(n, float(fun))
    fn: Callable[..., float] = fun
    args = [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if not args:
        return np.full(n, float(fn()))
    missing = [a for a in args if a not in data.columns]
    if missing:
        raise ValueError(f"Weight/knot function arguments not found in the data: {missing}")
    # Evaluate once per distinct covariate combination, then broadcast.
    keys = data[args]
    uniq = keys.drop_duplicates()
    values = {
        tuple(r): float(fn(**dict(zip(args, r, strict=True))))
        for r in uniq.itertuples(index=False)
    }
    return np.fromiter(
        (values[tuple(r)] for r in keys.itertuples(index=False)), dtype=float, count=n
    )
