"""Grids and shape restrictions on the MTR functions.

Shape restrictions (bounds on ``m0``, ``m1`` and ``m1 - m0``, monotonicity in
``u``) are imposed on a finite grid of ``(u, x)`` points as linear
inequalities on the MTR coefficients. Following the R package, the grid in
``u`` is a Halton sequence plus the end points and the grid in ``x`` is a
sample of the distinct covariate rows. The audit procedure checks the
constraints on a finer grid and adds violated points to the constraint set.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def halton(n: int, base: int = 2) -> NDArray[np.float64]:
    """First ``n`` points of the Halton (van der Corput) sequence, as in R ``rhalton``."""
    out = np.empty(n)
    for j in range(1, n + 1):
        f, r, i = 1.0, 0.0, j
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        out[j - 1] = r
    return out


def u_grid(n: int) -> NDArray[np.float64]:
    """Sorted grid ``{0, 1} + halton(n)`` rounded to 8 decimals, as in the R package."""
    if n <= 0:
        return np.array([0.0, 1.0])
    return np.sort(np.concatenate([[0.0, 1.0], np.round(halton(n), 8)]))


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


def build_grids(
    data: pd.DataFrame,
    xvars: Sequence[str],
    *,
    initgrid_nx: int,
    initgrid_nu: int,
    audit_nx: int,
    audit_nu: int,
    initgrid_x: pd.DataFrame | None = None,
    initgrid_u: ArrayLike | None = None,
    audit_x: pd.DataFrame | None = None,
    audit_u: ArrayLike | None = None,
    rng: np.random.Generator | None = None,
) -> Grids:
    """Construct the audit grid and the initial constraint grid.

    Parameters
    ----------
    data : pandas.DataFrame
        Estimation sample.
    xvars : sequence of str
        Covariates entering the MTRs.
    initgrid_nx, initgrid_nu : int
        Size of the initial grid in ``x`` (rows sampled from the audit
        support) and in ``u`` (Halton points, plus 0 and 1).
    audit_nx, audit_nu : int
        Size of the audit grid; ``audit_nx`` is capped by the number of
        distinct covariate rows.
    initgrid_x, audit_x : pandas.DataFrame, optional
        Explicit covariate grids, overriding the sampled ones.
    initgrid_u, audit_u : array_like, optional
        Explicit ``u`` grids; 0 and 1 are always included.
    rng : numpy.random.Generator, optional
        Source of randomness for sampling covariate rows.

    Returns
    -------
    Grids
    """
    rng = rng or np.random.default_rng()
    xvars = list(xvars)
    if not xvars:
        support = pd.DataFrame(index=pd.RangeIndex(1))
        init_index = np.array([0])
    else:
        if audit_x is None:
            full = data[xvars].drop_duplicates().reset_index(drop=True)
            take = min(audit_nx, len(full))
            support = full.iloc[np.sort(rng.choice(len(full), take, replace=False))]
            support = support.reset_index(drop=True)
        else:
            support = audit_x[xvars].reset_index(drop=True)
        if initgrid_x is None:
            take = min(initgrid_nx, len(support))
            init_index = np.sort(rng.choice(len(support), take, replace=False))
        else:
            init = initgrid_x[xvars].reset_index(drop=True)
            combined = pd.concat([init, support], ignore_index=True)
            is_init = np.arange(len(combined)) < len(init)
            keep = ~(combined.duplicated().to_numpy() & ~is_init)
            support = combined.loc[keep].reset_index(drop=True)
            init_index = np.flatnonzero(is_init[keep])
    if audit_u is None:
        a_u = u_grid(audit_nu)
    else:
        a_u = np.asarray(audit_u, dtype=float)
        if initgrid_u is not None:
            a_u = np.union1d(a_u, np.asarray(initgrid_u, dtype=float))
        a_u = np.union1d(a_u, [0.0, 1.0])
    if initgrid_u is not None:
        i_u = np.union1d(np.asarray(initgrid_u, dtype=float), [0.0, 1.0])
    elif initgrid_nu <= 0:
        i_u = np.array([0.0, 1.0])
    elif audit_u is None:
        i_u = u_grid(initgrid_nu)
    else:
        take = min(len(a_u), initgrid_nu)
        i_u = np.sort(rng.choice(a_u, take, replace=False))
    return Grids(support, a_u, np.asarray(init_index, dtype=np.intp), i_u)


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


def shape_constraints(
    spec0: MTRSpec,
    spec1: MTRSpec,
    support: pd.DataFrame,
    x_index: ArrayLike,
    uvec: ArrayLike,
    restrictions: dict[str, Any],
) -> ShapeConstraints:
    """Build the shape restrictions on the grid ``support[x_index] x uvec``.

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
    j0, j1 = spec0.n_coef, spec1.n_coef
    xi = np.asarray(x_index, dtype=np.intp)
    uu = np.sort(np.asarray(uvec, dtype=float))
    nx, nu = len(xi), len(uu)
    if nx == 0 or nu == 0:
        return _empty(j0 + j1)
    grid_x = support.iloc[np.repeat(xi, nu)].reset_index(drop=True)
    grid_u = np.tile(uu, nx)
    grid_xi = np.repeat(xi, nu)
    a0 = spec0.design(grid_x, grid_u)
    a1 = spec1.design(grid_x, grid_u)
    zero0, zero1 = np.zeros_like(a0), np.zeros_like(a1)

    blocks: list[NDArray[np.float64]] = []
    rhs: list[NDArray[np.float64]] = []
    kinds: list[NDArray[np.str_]] = []
    xis: list[NDArray[np.intp]] = []
    us: list[NDArray[np.float64]] = []

    def add(
        mat: NDArray[np.float64],
        b: NDArray[np.float64],
        kind: str,
        xidx: NDArray[np.intp],
        uval: NDArray[np.float64],
    ) -> None:
        # Drop duplicate rows (identical constraints at different grid points).
        _, first = np.unique(np.round(mat, 12), axis=0, return_index=True)
        first = np.sort(first)
        blocks.append(mat[first])
        rhs.append(b[first])
        kinds.append(np.full(len(first), kind))
        xis.append(xidx[first])
        us.append(uval[first])

    r = restrictions
    level = {
        "m0.lb": (r.get("m0_lb"), np.hstack([-a0, zero1]), -1.0),
        "m1.lb": (r.get("m1_lb"), np.hstack([zero0, -a1]), -1.0),
        "mte.lb": (r.get("mte_lb"), np.hstack([a0, -a1]), -1.0),
        "m0.ub": (r.get("m0_ub"), np.hstack([a0, zero1]), 1.0),
        "m1.ub": (r.get("m1_ub"), np.hstack([zero0, a1]), 1.0),
        "mte.ub": (r.get("mte_ub"), np.hstack([-a0, a1]), 1.0),
    }
    for kind, (bound, mat, sign) in level.items():
        if bound is not None:
            add(mat, np.full(len(mat), sign * float(bound)), kind, grid_xi, grid_u)

    if nu > 1 and any(
        r.get(k) for k in ("m0_inc", "m0_dec", "m1_inc", "m1_dec", "mte_inc", "mte_dec")
    ):
        # Differences between consecutive u values within each covariate cell.
        hi = np.array([c * nu + k for c in range(nx) for k in range(1, nu)])
        lo = hi - 1
        d0, d1 = a0[hi] - a0[lo], a1[hi] - a1[lo]
        z0, z1 = np.zeros_like(d0), np.zeros_like(d1)
        mono = {
            "m0.inc": (r.get("m0_inc"), np.hstack([-d0, z1])),
            "m0.dec": (r.get("m0_dec"), np.hstack([d0, z1])),
            "m1.inc": (r.get("m1_inc"), np.hstack([z0, -d1])),
            "m1.dec": (r.get("m1_dec"), np.hstack([z0, d1])),
            "mte.inc": (r.get("mte_inc"), np.hstack([d0, -d1])),
            "mte.dec": (r.get("mte_dec"), np.hstack([-d0, d1])),
        }
        for kind, (flag, mat) in mono.items():
            if flag:
                add(mat, np.zeros(len(mat)), kind, grid_xi[hi], grid_u[hi])

    if not blocks:
        return _empty(j0 + j1)
    return ShapeConstraints(
        sp.csr_matrix(np.vstack(blocks)),
        np.concatenate(rhs),
        np.concatenate(kinds),
        np.concatenate(xis),
        np.concatenate(us),
    )


def violations(
    cons: ShapeConstraints, thetas: Sequence[NDArray[np.float64]], tol: float
) -> pd.DataFrame:
    """Constraint rows violated by any of the candidate solutions.

    Parameters
    ----------
    cons : ShapeConstraints
        Constraints on the audit grid.
    thetas : sequence of numpy.ndarray
        Stacked coefficient vectors to check (typically the solutions at the
        lower and upper bound).
    tol : float
        Violations smaller than ``tol`` are ignored.

    Returns
    -------
    pandas.DataFrame
        Columns ``row`` (position in ``cons``), ``kind``, ``x_index``, ``u``
        and ``diff`` (largest residual across the solutions).
    """
    diff = np.max(np.column_stack([cons.residuals(t) for t in thetas]), axis=1)
    pos = np.flatnonzero(diff > tol)
    return pd.DataFrame(
        {
            "row": pos,
            "kind": cons.kind[pos],
            "x_index": cons.x_index[pos],
            "u": cons.u[pos],
            "diff": diff[pos],
        }
    )


def select_violations(viol: pd.DataFrame, audit_add: int) -> pd.DataFrame:
    """Choose which violated points to add, following the R package.

    All violations are added when there are at most ``audit_add``. Otherwise
    the worst violation of every (restriction, covariate cell) group is
    taken first, then the second worst of every group, and so on until at
    least ``audit_add`` points are selected.
    """
    if len(viol) <= audit_add:
        return viol
    kind_order = {k: i for i, k in enumerate(KINDS)}
    v = viol.assign(_k=viol["kind"].map(kind_order))
    v = v.sort_values(["_k", "x_index", "diff"], ascending=[True, True, False])
    v["rank"] = v.groupby(["_k", "x_index"]).cumcount() + 1
    counts = v["rank"].value_counts().sort_index().cumsum()
    if counts.iloc[0] >= audit_add:
        chosen = v[v["rank"] == 1]
    else:
        k = int(counts.index[int(np.searchsorted(counts.to_numpy(), audit_add))])
        full = v[v["rank"] <= k - 1]
        extra = v[v["rank"] == k].sort_values("diff", ascending=False)
        chosen = pd.concat([full, extra.head(audit_add - len(full))])
    chosen = chosen.sort_values(["_k", "x_index", "diff"], ascending=[True, True, False])
    return chosen.drop(columns=["_k", "rank"])
