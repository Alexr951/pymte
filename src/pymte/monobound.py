"""Grids and shape restrictions on the MTR functions.

Shape restrictions (bounds on ``m0``, ``m1`` and ``m1 - m0``, monotonicity in
``u``) are imposed on a finite grid of ``(u, x)`` points as linear
inequalities on the MTR coefficients. :func:`gengrid` expands a covariate
support and a ``u`` vector into the grid, :func:`genbound_a` and
:func:`genmono_a` build the boundedness and monotonicity rows,
:func:`combinemonobound` stacks constraint sets and :func:`genmonobound_a`
does all of this for a set of restrictions. The names follow the R package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray

from pymte.mtr import MTRSpec

KINDS = (
    "m0.lb",
    "m1.lb",
    "mte.lb",
    "m0.ub",
    "m1.ub",
    "mte.ub",
    "m0.inc",
    "m0.dec",
    "m1.inc",
    "m1.dec",
    "mte.inc",
    "mte.dec",
)


@dataclass(frozen=True)
class Grids:
    """Audit and initial constraint grids.

    Attributes
    ----------
    support : pandas.DataFrame
        Covariate rows of the audit grid (empty frame with one row when the
        MTRs have no covariates).
    audit_u : numpy.ndarray
        Values of ``u`` in the audit grid.
    init_index : numpy.ndarray
        Row positions of ``support`` used for the initial constraint grid.
    init_u : numpy.ndarray
        Values of ``u`` in the initial constraint grid.
    """

    support: pd.DataFrame
    audit_u: NDArray[np.float64]
    init_index: NDArray[np.intp]
    init_u: NDArray[np.float64]

    @property
    def no_x(self) -> bool:
        """Whether the MTRs involve no covariates."""
        return self.support.shape[1] == 0


@dataclass(frozen=True)
class ShapeConstraints:
    """Linear inequalities ``A @ theta <= b`` on the stacked coefficients ``(theta0, theta1)``.

    Attributes
    ----------
    A : scipy.sparse.csr_matrix
        Constraint matrix with ``n_coef0 + n_coef1`` columns.
    b : numpy.ndarray
        Right-hand sides.
    kind : numpy.ndarray of str
        Restriction each row implements (one of :data:`KINDS`).
    x_index : numpy.ndarray of int
        Position in the covariate support of the grid point of each row.
    u : numpy.ndarray
        Value of ``u`` of each row (for monotonicity rows, the upper point of
        the pair).
    """

    A: sp.csr_matrix
    b: NDArray[np.float64]
    kind: NDArray[np.str_]
    x_index: NDArray[np.intp]
    u: NDArray[np.float64]

    @property
    def n(self) -> int:
        """Number of constraint rows."""
        return int(self.A.shape[0])

    def subset(self, rows: ArrayLike) -> ShapeConstraints:
        """Return the constraints at the given row positions."""
        idx = np.asarray(rows, dtype=np.intp)
        return ShapeConstraints(
            sp.csr_matrix(self.A[idx]), self.b[idx], self.kind[idx], self.x_index[idx], self.u[idx]
        )

    def residuals(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute ``A @ theta - b``; positive entries are violations."""
        return np.asarray(self.A @ theta - self.b, dtype=float)


def _empty(n_cols: int) -> ShapeConstraints:
    return ShapeConstraints(
        sp.csr_matrix((0, n_cols)),
        np.empty(0),
        np.empty(0, dtype=str),
        np.empty(0, dtype=np.intp),
        np.empty(0),
    )


def gengrid(
    support: pd.DataFrame, x_index: ArrayLike, uvec: ArrayLike
) -> tuple[pd.DataFrame, NDArray[np.float64], NDArray[np.intp]]:
    """Expand covariate rows and a ``u`` vector into the constraint grid.

    Parameters
    ----------
    support : pandas.DataFrame
        Covariate rows.
    x_index : array_like of int
        Rows of ``support`` forming the grid.
    uvec : array_like
        Grid in ``u`` (sorted on output).

    Returns
    -------
    tuple
        Covariate rows of the grid (one per point, ``u`` varying fastest),
        the ``u`` value of each point and the position in ``support`` of each
        point.
    """
    xi = np.asarray(x_index, dtype=np.intp)
    uu = np.sort(np.asarray(uvec, dtype=float))
    nu = len(uu)
    grid_x = support.iloc[np.repeat(xi, nu)].reset_index(drop=True)
    return grid_x, np.tile(uu, len(xi)), np.repeat(xi, nu)


def _dedupe(
    mat: NDArray[np.float64],
    b: NDArray[np.float64],
    kind: str,
    xidx: NDArray[np.intp],
    uval: NDArray[np.float64],
) -> ShapeConstraints:
    # Identical constraints at different grid points are kept once.
    _, first = np.unique(np.round(mat, 12), axis=0, return_index=True)
    first = np.sort(first)
    return ShapeConstraints(
        sp.csr_matrix(mat[first]), b[first], np.full(len(first), kind), xidx[first], uval[first]
    )


def combinemonobound(*parts: ShapeConstraints) -> ShapeConstraints:
    """Stack constraint sets (empty ones are skipped)."""
    parts = tuple(p for p in parts if p.n)
    if not parts:
        raise ValueError("combinemonobound needs at least one non-empty constraint set")
    if len(parts) == 1:
        return parts[0]
    return ShapeConstraints(
        sp.csr_matrix(sp.vstack([p.A for p in parts])),
        np.concatenate([p.b for p in parts]),
        np.concatenate([p.kind for p in parts]),
        np.concatenate([p.x_index for p in parts]),
        np.concatenate([p.u for p in parts]),
    )


def genbound_a(
    a0: NDArray[np.float64],
    a1: NDArray[np.float64],
    grid_xi: NDArray[np.intp],
    grid_u: NDArray[np.float64],
    restrictions: dict[str, Any],
) -> list[ShapeConstraints]:
    """Boundedness rows for ``m0``, ``m1`` and ``m1 - m0`` on the grid.

    Parameters
    ----------
    a0, a1 : numpy.ndarray
        MTR bases evaluated at the grid points, from :meth:`MTRSpec.design`.
    grid_xi, grid_u : numpy.ndarray
        Covariate position and ``u`` value of each grid point.
    restrictions : dict
        Bounds ``m0_lb, m0_ub, m1_lb, m1_ub, mte_lb, mte_ub`` (numbers or
        ``None``).

    Returns
    -------
    list of ShapeConstraints
        One entry per active bound, in ``A @ theta <= b`` form.
    """
    zero0, zero1 = np.zeros_like(a0), np.zeros_like(a1)
    r = restrictions
    level = {
        "m0.lb": (r.get("m0_lb"), np.hstack([-a0, zero1]), -1.0),
        "m1.lb": (r.get("m1_lb"), np.hstack([zero0, -a1]), -1.0),
        "mte.lb": (r.get("mte_lb"), np.hstack([a0, -a1]), -1.0),
        "m0.ub": (r.get("m0_ub"), np.hstack([a0, zero1]), 1.0),
        "m1.ub": (r.get("m1_ub"), np.hstack([zero0, a1]), 1.0),
        "mte.ub": (r.get("mte_ub"), np.hstack([-a0, a1]), 1.0),
    }
    return [
        _dedupe(mat, np.full(len(mat), sign * float(b)), kind, grid_xi, grid_u)
        for kind, (b, mat, sign) in level.items()
        if b is not None
    ]


def genmono_a(
    a0: NDArray[np.float64],
    a1: NDArray[np.float64],
    grid_xi: NDArray[np.intp],
    grid_u: NDArray[np.float64],
    nu: int,
    restrictions: dict[str, Any],
) -> list[ShapeConstraints]:
    """Monotonicity rows: differences between consecutive ``u`` values in each covariate cell.

    Parameters
    ----------
    a0, a1, grid_xi, grid_u
        As in :func:`genbound_a`; the grid must come from :func:`gengrid`
        (``u`` varying fastest).
    nu : int
        Number of ``u`` values per covariate cell.
    restrictions : dict
        Flags ``m0_inc, m0_dec, m1_inc, m1_dec, mte_inc, mte_dec``.

    Returns
    -------
    list of ShapeConstraints
        One entry per active restriction.
    """
    flags = ("m0_inc", "m0_dec", "m1_inc", "m1_dec", "mte_inc", "mte_dec")
    if nu < 2 or not any(restrictions.get(k) for k in flags):
        return []
    nx = len(a0) // nu
    hi = np.array([c * nu + k for c in range(nx) for k in range(1, nu)])
    lo = hi - 1
    d0, d1 = a0[hi] - a0[lo], a1[hi] - a1[lo]
    z0, z1 = np.zeros_like(d0), np.zeros_like(d1)
    r = restrictions
    mono = {
        "m0.inc": (r.get("m0_inc"), np.hstack([-d0, z1])),
        "m0.dec": (r.get("m0_dec"), np.hstack([d0, z1])),
        "m1.inc": (r.get("m1_inc"), np.hstack([z0, -d1])),
        "m1.dec": (r.get("m1_dec"), np.hstack([z0, d1])),
        "mte.inc": (r.get("mte_inc"), np.hstack([d0, -d1])),
        "mte.dec": (r.get("mte_dec"), np.hstack([-d0, d1])),
    }
    return [
        _dedupe(mat, np.zeros(len(mat)), kind, grid_xi[hi], grid_u[hi])
        for kind, (flag, mat) in mono.items()
        if flag
    ]


def genmonobound_a(
    spec0: MTRSpec,
    spec1: MTRSpec,
    support: pd.DataFrame,
    x_index: ArrayLike,
    uvec: ArrayLike,
    restrictions: dict[str, Any],
) -> ShapeConstraints:
    """Build all shape restrictions on the grid ``support[x_index] x uvec``.

    Parameters
    ----------
    spec0, spec1 : MTRSpec
        MTR specifications.
    support : pandas.DataFrame
        Covariate rows.
    x_index : array_like of int
        Rows of ``support`` forming the grid.
    uvec : array_like
        Grid in ``u``.
    restrictions : dict
        Keys among ``m0_lb, m0_ub, m1_lb, m1_ub, mte_lb, mte_ub`` (numbers
        or ``None``) and ``m0_inc, m0_dec, m1_inc, m1_dec, mte_inc, mte_dec``
        (booleans).

    Returns
    -------
    ShapeConstraints
        All rows in ``A @ theta <= b`` form; ``>=`` restrictions are negated.
    """
    n_cols = spec0.n_coef + spec1.n_coef
    xi = np.asarray(x_index, dtype=np.intp)
    nu = len(np.asarray(uvec, dtype=float))
    if len(xi) == 0 or nu == 0:
        return _empty(n_cols)
    grid_x, grid_u, grid_xi = gengrid(support, xi, uvec)
    a0 = spec0.design(grid_x, grid_u)
    a1 = spec1.design(grid_x, grid_u)
    parts = genbound_a(a0, a1, grid_xi, grid_u, restrictions)
    parts += genmono_a(a0, a1, grid_xi, grid_u, nu, restrictions)
    if not parts:
        return _empty(n_cols)
    return combinemonobound(*parts)
