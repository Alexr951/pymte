"""Plots of MTR functions, the MTE and the weights (requires matplotlib)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ivmte.results import IVMTEResult


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("Plotting requires matplotlib: pip install 'ivmte[plots]'") from err
    return plt


def _covariate_row(result: IVMTEResult, at: Mapping[str, Any] | None, n: int) -> pd.DataFrame:
    s0, s1 = result.specs
    needed = sorted(set(s0.covariates) | set(s1.covariates))
    at = dict(at or {})
    missing = [c for c in needed if c not in at]
    if missing:
        raise ValueError(
            f"The MTRs depend on covariates; pass their values with 'at', e.g. "
            f"at={{{', '.join(f'{c!r}: ...' for c in missing)}}}"
        )
    return pd.DataFrame({c: [at[c]] * n for c in needed}, index=range(n))


def _mtr_curves(
    result: IVMTEResult, u: NDArray[np.float64], at: Mapping[str, Any] | None
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Return ``{label: (m0(u), m1(u))}`` for the coefficient vectors in the result."""
    s0, s1 = result.specs
    frame = _covariate_row(result, at, len(u))
    d0, d1 = s0.design(frame, u), s1.design(frame, u)
    j0 = s0.n_coef
    thetas: dict[str, NDArray[np.float64]] = {}
    if result.mtr_coef is not None:
        thetas["estimate"] = result.mtr_coef.to_numpy()
    elif result.gstar_coef is not None:
        thetas["lower bound"] = result.gstar_coef["min"].to_numpy()
        thetas["upper bound"] = result.gstar_coef["max"].to_numpy()
    return {k: (d0 @ t[:j0], d1 @ t[j0:]) for k, t in thetas.items()}


def plot_mtr(
    result: IVMTEResult,
    at: Mapping[str, Any] | None = None,
    n_points: int = 201,
    ax: Any = None,
) -> Any:
    """Plot the MTR functions ``m0(u)`` and ``m1(u)``.

    In the partially identified case the curves at the lower and upper bound
    are drawn; in the point identified case the estimated curves.

    Parameters
    ----------
    result : IVMTEResult
        Output of :func:`ivmte.ivmte`.
    at : mapping, optional
        Covariate values at which to evaluate the MTRs; required when the
        specification includes covariates.
    n_points : int, default 201
        Number of grid points in ``u``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _plt()
    ax = ax if ax is not None else plt.subplots()[1]
    u = np.linspace(0, 1, n_points)
    for label, (m0, m1) in _mtr_curves(result, u, at).items():
        (line,) = ax.plot(u, m1, label=f"m1, {label}")
        ax.plot(u, m0, linestyle="--", color=line.get_color(), label=f"m0, {label}")
    ax.set_xlabel("u")
    ax.set_ylabel("MTR")
    ax.legend()
    return ax


def plot_mte(
    result: IVMTEResult,
    at: Mapping[str, Any] | None = None,
    n_points: int = 201,
    ax: Any = None,
) -> Any:
    """Plot the marginal treatment effect ``m1(u) - m0(u)``.

    Parameters are as in :func:`plot_mtr`.
    """
    plt = _plt()
    ax = ax if ax is not None else plt.subplots()[1]
    u = np.linspace(0, 1, n_points)
    for label, (m0, m1) in _mtr_curves(result, u, at).items():
        ax.plot(u, m1 - m0, label=label)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("u")
    ax.set_ylabel("MTE")
    ax.legend()
    return ax


def plot_weights(result: IVMTEResult, n_points: int = 401, ax: Any = None) -> Any:
    """Plot the average weights on ``m1`` as functions of ``u``.

    The target weight and the weight of each IV-like estimand are averaged
    over the sample, as in the figures of Mogstad, Santos and Torgovitsky
    (2018). The weights on ``m0`` are the negatives for the target and the
    complements in ``u`` for the IV-like estimands.

    Parameters
    ----------
    result : IVMTEResult
        Output of :func:`ivmte.ivmte`; the target weights must be of a
        conventional target (not custom weights).
    n_points : int, default 401
        Number of grid points in ``u``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _plt()
    ax = ax if ax is not None else plt.subplots()[1]
    u = np.linspace(0, 1, n_points)
    w = result.target_gammas.weights
    if w is None:
        raise ValueError("Weight plots are available for conventional targets only")
    inside = (u[None, :] >= w.lb[:, None]) & (u[None, :] <= w.ub[:, None])
    target = (w.mult[:, None] * inside)[w.rows].mean(axis=0)
    ax.plot(u, target, label=f"target ({result.target})", linewidth=2)
    if result.ivlike is not None:
        p = result.propensity.phat
        for k, fit in enumerate(result.ivlike.fits):
            rows = fit.rows
            for j, name in enumerate(fit.components):
                s1 = fit.s1[:, j][:, None] * (u[None, :] <= p[rows][:, None])
                ax.plot(u, s1.mean(axis=0), linewidth=1, label=f"{k + 1}:{name}")
    ax.set_xlabel("u")
    ax.set_ylabel("weight on m1")
    ax.legend(fontsize="small")
    return ax
