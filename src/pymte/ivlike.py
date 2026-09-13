"""IV-like estimands and their representation as moments of the MTR coefficients.

An IV-like estimand is a coefficient ``beta_k`` from a linear regression
(OLS or two-stage least squares) of the outcome on treatment, covariates and
instruments. By Mogstad, Santos and Torgovitsky (2018) every such
coefficient equals ``E[s_k(D, Z) Y]`` for a known weight ``s_k``, and hence
is linear in the MTR coefficients:

``beta_k = Gamma_0[k] @ theta0 + Gamma_1[k] @ theta1``

with ``Gamma_d[k, j] = E[ s_k(d, Z) * int b_j(u, X) 1{u in I_d(p)} du ]``,
``I_0(p) = [p, 1]`` and ``I_1(p) = [0, p]``. :func:`piv` computes the
coefficients, :func:`iv_estimate` fits one specification and its weights;
the moments are assembled by :func:`pymte.mst.gen_s_set`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pymte.design import design
from pymte.sweights import olsj, tsls


@dataclass(frozen=True)
class IVLikeFit:
    """A fitted IV-like regression and the weights of its selected coefficients.

    Attributes
    ----------
    formula : str
        The regression formula.
    components : tuple of str
        Names of the selected coefficients.
    beta : numpy.ndarray
        The selected coefficient estimates.
    s0, s1 : numpy.ndarray
        Weights ``s_k(0, Z_i)`` and ``s_k(1, Z_i)``, shape ``(n_subset, K)``.
    rows : numpy.ndarray of bool
        Observations used (the ``subset``).
    y : numpy.ndarray
        Outcome on the subset.
    d : numpy.ndarray
        Treatment on the subset.
    """

    formula: str
    components: tuple[str, ...]
    beta: NDArray[np.float64]
    s0: NDArray[np.float64]
    s1: NDArray[np.float64]
    rows: NDArray[np.bool_]
    y: NDArray[np.float64]
    d: NDArray[np.float64]
    instrumented: bool


@dataclass(frozen=True)
class MomentSet:
    """The IV-like moment conditions ``beta = Gamma0 @ theta0 + Gamma1 @ theta1``.

    Attributes
    ----------
    fits : tuple of IVLikeFit
        The underlying regressions.
    names : tuple of str
        One name per moment, ``"{spec}:{coefficient}"``.
    beta : numpy.ndarray
        Estimated IV-like coefficients, shape ``(S,)``.
    gamma0, gamma1 : numpy.ndarray
        Averaged integrals, shapes ``(S, J0)`` and ``(S, J1)``.
    gamma0_obs, gamma1_obs : tuple of numpy.ndarray
        Per-observation integrals for each moment, on its subset.
    ys : tuple of numpy.ndarray
        Per-observation ``s_k(D_i, Z_i) Y_i`` for each moment, on its subset.
    rows : tuple of numpy.ndarray
        Subset mask of each moment.
    n_independent : int
        Rank of the stacked weight matrix, the number of linearly independent
        moment conditions.
    """

    fits: tuple[IVLikeFit, ...]
    names: tuple[str, ...]
    beta: NDArray[np.float64]
    gamma0: NDArray[np.float64]
    gamma1: NDArray[np.float64]
    gamma0_obs: tuple[NDArray[np.float64], ...]
    gamma1_obs: tuple[NDArray[np.float64], ...]
    ys: tuple[NDArray[np.float64], ...]
    rows: tuple[NDArray[np.bool_], ...]
    n_independent: int

    @property
    def n_moments(self) -> int:
        """Number of moment conditions."""
        return len(self.beta)


def independent_columns(x: NDArray[np.float64], tol: float = 1e-7) -> list[int]:
    """Columns R's ``lm.fit`` keeps: each column is dropped when it lies in the span of the earlier ones.

    The residual of a column after projecting on the columns kept so far is
    compared with ``tol`` times its norm, the rule of the pivoted QR
    decomposition behind ``lm.fit``.
    """
    q = np.empty((len(x), 0))
    keep = []
    for j in range(x.shape[1]):
        col = x[:, j]
        resid = col - q @ (q.T @ col)
        norm = np.linalg.norm(resid)
        if norm > tol * np.linalg.norm(col) > 0:
            q = np.column_stack([q, resid / norm])
            keep.append(j)
    return keep


def _match(part: str, name: str) -> bool:
    # A factor such as ``C(z)`` names every level column ``C(z)[T.1]``, ...
    return name == part or name.startswith(part + "[")


def _resolve_components(
    components: Sequence[str] | None, names: Sequence[str], formula: str
) -> list[int]:
    """Positions of the requested coefficients, expanding factors and reordering interactions.

    As in R, a component naming a factor selects all of its level columns,
    the factors of an interaction may be given in any order, and
    ``"intercept"`` names the constant.
    """
    if not components:
        return list(range(len(names)))
    split = [n.split(":") for n in names]
    out: list[int] = []
    for c in components:
        parts = ("Intercept" if c == "intercept" else c).split(":")
        found = [
            i
            for i, np_ in enumerate(split)
            if len(np_) == len(parts)
            and all(any(_match(p, q) for q in np_) for p in parts)
            and all(any(_match(p, q) for p in parts) for q in np_)
        ]
        if not found:
            raise ValueError(
                f"Component {c!r} is not a coefficient of {formula!r}; available: {list(names)}"
            )
        out += [i for i in found if i not in out]
    return out


def piv(
    y: NDArray[np.float64], x: NDArray[np.float64], z: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """OLS (``z=None``) or TSLS coefficients of ``y`` on ``x`` with instruments ``z``.

    As in R's ``lm.fit``, a regressor that is collinear with the earlier
    ones gets a ``nan`` coefficient instead of raising, and collinear
    instruments are dropped from the first stage.
    """
    n = len(y)
    keep = independent_columns(x)
    beta = np.full(x.shape[1], np.nan)
    xk = x[:, keep]
    if z is None:
        beta[keep] = np.linalg.solve(xk.T @ xk / n, xk.T @ y / n)
        return beta
    if z.shape[1] < x.shape[1]:
        raise ValueError("TSLS needs at least as many instruments as regressors")
    zk = z[:, independent_columns(z)]
    exz = xk.T @ zk / n
    pi = exz @ np.linalg.inv(zk.T @ zk / n)
    beta[keep] = np.linalg.solve(pi @ exz.T, pi @ (zk.T @ y / n))
    return beta


def iv_estimate(
    formula: str,
    data: pd.DataFrame,
    treat: str,
    components: Sequence[str] | None = None,
    subset: str | None = None,
) -> IVLikeFit:
    """Fit one IV-like regression and compute the weights of its coefficients.

    Parameters
    ----------
    formula : str
        ``"y ~ d + x"`` for OLS or ``"y ~ d + x | z + x"`` for TSLS, where the
        part after ``|`` lists the instruments (exogenous covariates must be
        repeated there, as in R).
    data : pandas.DataFrame
        Estimation sample.
    treat : str
        Name of the treatment variable.
    components : sequence of str, optional
        Coefficients to use as estimands; ``"intercept"`` names the constant,
        a factor such as ``"C(z)"`` selects all of its level columns, and the
        factors of an interaction may be given in any order. Default: all
        coefficients. A component that is dropped for collinearity is
        omitted silently, as in R.
    subset : str, optional
        A :meth:`pandas.DataFrame.eval` expression selecting the rows used.

    Returns
    -------
    IVLikeFit
    """
    dm = design(formula, data, subset, treat)
    beta = piv(dm.y, dm.x, dm.z)
    # Collinear regressors and instruments are dropped, together with the
    # components they carried, as the R package does.
    keep_x = [j for j in range(dm.x.shape[1]) if not np.isnan(beta[j])]
    names = tuple(dm.x_names[j] for j in keep_x)
    requested = _resolve_components(components, dm.x_names, formula)
    cpos = [keep_x.index(j) for j in requested if j in keep_x]
    x, x0, x1 = dm.x[:, keep_x], dm.x0[:, keep_x], dm.x1[:, keep_x]
    if dm.z is None:
        s0, s1 = olsj(x, x0, x1, cpos)
    else:
        assert dm.z0 is not None and dm.z1 is not None
        keep_z = independent_columns(dm.z)
        s0, s1 = tsls(x, dm.z[:, keep_z], dm.z0[:, keep_z], dm.z1[:, keep_z], cpos)
    return IVLikeFit(
        formula=formula,
        components=tuple(names[i] for i in cpos),
        beta=beta[keep_x][cpos],
        s0=np.asarray(s0, dtype=float),
        s1=np.asarray(s1, dtype=float),
        rows=dm.rows,
        y=dm.y,
        d=np.asarray(data.loc[dm.rows, treat], dtype=float),
        instrumented=dm.z is not None,
    )
