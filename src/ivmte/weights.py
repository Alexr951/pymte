"""Target parameter weights and their integrals against the MTR bases.

Every target parameter of the form
``E[ int_0^1 ( w_1(u, X, Z) m_1(u, X) + w_0(u, X, Z) m_0(u, X) ) du ]`` is
represented by, for each arm, a lower limit, an upper limit and a multiplier
per observation: the weight is ``multiplier * 1{lb <= u <= ub}``. Integrating
the MTR bases over these intervals and averaging gives the vectors
``gstar0`` and ``gstar1`` such that the target equals
``gstar0 @ theta0 + gstar1 @ theta1``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ivmte.mtr import MTRSpec
from ivmte.propensity import Propensity

TARGETS = ("ate", "att", "atu", "late", "avglate", "genlate")


@dataclass(frozen=True)
class TargetWeights:
    """Weights of a conventional target parameter, one interval and multiplier per row.

    The ``m1`` weight is ``mult * 1{lb <= u <= ub}`` and the ``m0`` weight is
    its negative.

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


@dataclass(frozen=True)
class TargetGammas:
    """Integrated target weights: ``target = gstar0 @ theta0 + gstar1 @ theta1``."""

    gstar0: NDArray[np.float64]
    gstar1: NDArray[np.float64]
    weights: TargetWeights | None
    n: int


def _rows_from_late_x(data: pd.DataFrame, late_x: dict[str, Any] | None) -> NDArray[np.bool_]:
    rows = np.ones(len(data), dtype=bool)
    if late_x:
        for name, value in late_x.items():
            rows &= np.asarray(data[name] == value)
        if not rows.any():
            raise ValueError(f"No observations satisfy the 'late_x' condition {late_x}")
    return rows


def _predict_at(
    prop: Propensity,
    data: pd.DataFrame,
    values: dict[str, Any],
    late_x: dict[str, Any] | None,
) -> NDArray[np.float64]:
    if not prop.fitted:
        raise ValueError("Target parameters 'late' and 'avglate' require a propensity score model")
    new = data.assign(**values)
    if late_x:
        new = new.assign(**late_x)
    return prop.predict(new)


def conventional_weights(
    target: str,
    data: pd.DataFrame,
    prop: Propensity,
    *,
    late_from: dict[str, Any] | None = None,
    late_to: dict[str, Any] | None = None,
    late_x: dict[str, Any] | None = None,
    genlate_lb: float | None = None,
    genlate_ub: float | None = None,
) -> TargetWeights:
    """Weights for the ATE, ATT, ATU, LATE, average LATE and generalised LATE.

    Parameters
    ----------
    target : {"ate", "att", "atu", "late", "avglate", "genlate"}
        Target parameter.
    data : pandas.DataFrame
        Estimation sample.
    prop : Propensity
        Fitted propensity score.
    late_from, late_to : dict, optional
        Instrument values defining the LATE, e.g. ``{"z": 1}`` and ``{"z": 3}``.
    late_x : dict, optional
        Covariate values to condition on, e.g. ``{"x": 2}``.
    genlate_lb, genlate_ub : float, optional
        Limits in ``u`` of the generalised LATE.

    Returns
    -------
    TargetWeights
    """
    target = target.lower()
    n = len(data)
    p = prop.phat
    d = np.asarray(data[prop.treat], dtype=float)
    rows = _rows_from_late_x(data, late_x)
    zeros, ones = np.zeros(n), np.ones(n)
    if target == "ate":
        lb, ub, mult = zeros, ones, ones
    elif target == "att":
        lb, ub, mult = zeros, p, np.full(n, 1.0 / d.mean())
    elif target == "atu":
        lb, ub, mult = p, ones, np.full(n, 1.0 / (1.0 - d.mean()))
    elif target in ("late", "avglate"):
        if not late_from or not late_to:
            raise ValueError("Targets 'late' and 'avglate' need 'late_from' and 'late_to'")
        lb = _predict_at(prop, data, late_from, late_x)
        ub = _predict_at(prop, data, late_to, late_x)
        if target == "late":
            mult = np.full(n, 1.0 / abs(ub[rows].mean() - lb[rows].mean()))
        else:
            mult = 1.0 / np.abs(ub - lb)
    elif target == "genlate":
        if genlate_lb is None or genlate_ub is None:
            raise ValueError("Target 'genlate' needs 'genlate_lb' and 'genlate_ub'")
        if not 0 <= genlate_lb < genlate_ub <= 1:
            raise ValueError("'genlate_lb' and 'genlate_ub' must satisfy 0 <= lb < ub <= 1")
        lb, ub = np.full(n, genlate_lb), np.full(n, genlate_ub)
        mult = np.full(n, 1.0 / (genlate_ub - genlate_lb))
    else:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    return TargetWeights(lb, ub, mult, rows)


def target_gammas_from_weights(
    spec0: MTRSpec, spec1: MTRSpec, data: pd.DataFrame, w: TargetWeights
) -> TargetGammas:
    """Integrate the MTR bases against interval weights and average over rows."""
    g0 = -spec0.gamma(data, w.lb, w.ub, w.mult, rows=w.rows).mean(axis=0)
    g1 = spec1.gamma(data, w.lb, w.ub, w.mult, rows=w.rows).mean(axis=0)
    return TargetGammas(g0, g1, w, int(w.rows.sum()))


Piece = float | int | Callable[..., float]


def _evaluate_piecewise(data: pd.DataFrame, fun: Piece) -> NDArray[np.float64]:
    """Evaluate a constant or a callable of named covariates row by row."""
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


def custom_target_gammas(
    spec0: MTRSpec,
    spec1: MTRSpec,
    data: pd.DataFrame,
    *,
    target_weight0: Sequence[Piece],
    target_weight1: Sequence[Piece],
    target_knots0: Sequence[Piece] = (),
    target_knots1: Sequence[Piece] = (),
) -> TargetGammas:
    """Target defined by piecewise-constant weights in ``u``.

    Parameters
    ----------
    spec0, spec1 : MTRSpec
        MTR specifications of the two arms.
    data : pandas.DataFrame
        Estimation sample.
    target_weight0, target_weight1 : sequence
        Weights on each piece, as numbers or callables of covariate columns
        (argument names must match column names). With ``K`` knots there
        must be ``K + 1`` weights. No sign convention is imposed: for a
        treatment effect the ``m0`` weights are typically negative.
    target_knots0, target_knots1 : sequence, optional
        Interior knots in ``u`` splitting [0, 1] into pieces, as numbers or
        callables of covariate columns.

    Returns
    -------
    TargetGammas
    """
    gammas = []
    for spec, weights, knots in (
        (spec0, target_weight0, target_knots0),
        (spec1, target_weight1, target_knots1),
    ):
        if len(weights) != len(knots) + 1:
            raise ValueError(
                f"Expected {len(knots) + 1} weights for {len(knots)} knots, got {len(weights)}"
            )
        edges = [np.zeros(len(data))] + [_evaluate_piecewise(data, k) for k in knots]
        edges.append(np.ones(len(data)))
        total = np.zeros(spec.n_coef)
        for i, w in enumerate(weights):
            mult = _evaluate_piecewise(data, w)
            total += spec.gamma(data, edges[i], edges[i + 1], mult).mean(axis=0)
        gammas.append(total)
    return TargetGammas(gammas[0], gammas[1], None, len(data))
