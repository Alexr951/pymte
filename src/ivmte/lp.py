"""Optimisation problems for partial identification.

Two criteria are supported. The *moment approach* measures the fit of the
IV-like moments by the absolute deviation ``sum_s |Gamma_s @ theta - beta_s|``,
linearised with two non-negative slack variables per moment, so both the
criterion minimisation and the bound problems are linear programs. The
*regression approach* uses the least-squares criterion
``||X @ theta - y||^2 / n``; the criterion problem is then a quadratic
program and the bound problems are linear objectives under one quadratic
constraint.

In both cases the decision vector is ``theta = (theta0, theta1)``, the
stacked MTR coefficients, preceded by the slack variables in the moment
approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ivmte.mtr import MTRSpec
from ivmte.shape import ShapeConstraints
from ivmte.solvers import LinearConstraints, SolveResult, solve_lp, solve_qcqp


@dataclass(frozen=True)
class L1Criterion:
    """Absolute-deviation criterion of the moment approach.

    Attributes
    ----------
    gamma : numpy.ndarray
        Moment matrix ``[Gamma0, Gamma1]`` of shape ``(S, J)``.
    beta : numpy.ndarray
        IV-like estimands, shape ``(S,)``.
    """

    gamma: NDArray[np.float64]
    beta: NDArray[np.float64]

    @property
    def n_slack(self) -> int:
        """Number of slack variables (two per moment)."""
        return 2 * len(self.beta)

    @property
    def n_coef(self) -> int:
        """Number of MTR coefficients."""
        return int(self.gamma.shape[1])


@dataclass(frozen=True)
class LSCriterion:
    """Least-squares criterion of the regression approach, in compact form.

    ``||X theta - y||^2 / n`` is written as ``||G theta - h||^2 + const`` with
    ``G`` of shape ``(r, J)``, ``r`` the rank of ``X``.
    """

    G: NDArray[np.float64]
    h: NDArray[np.float64]
    const: float

    @classmethod
    def from_regression(cls, x: NDArray[np.float64], y: NDArray[np.float64]) -> LSCriterion:
        """Compress the design ``x`` and outcome ``y`` into the compact form."""
        n = len(y)
        m = x.T @ x / n
        b = x.T @ y / n
        w, v = np.linalg.eigh(m)
        keep = w > w.max() * 1e-12
        g = np.sqrt(w[keep])[:, None] * v[:, keep].T  # G'G = M
        h = np.linalg.pinv(g.T) @ b
        const = float(y @ y / n - h @ h)
        return cls(g, h, const)

    @property
    def n_slack(self) -> int:
        """No slack variables in the regression approach."""
        return 0

    @property
    def n_coef(self) -> int:
        """Number of MTR coefficients."""
        return int(self.G.shape[1])


Criterion = L1Criterion | LSCriterion


def equal_coef_matrix(
    spec0: MTRSpec, spec1: MTRSpec, names: tuple[str, ...]
) -> NDArray[np.float64]:
    """Rows enforcing ``theta0[name] == theta1[name]`` for each name.

    Parameters
    ----------
    spec0, spec1 : MTRSpec
        MTR specifications; each ``name`` must be a coefficient of both.
    names : tuple of str
        Coefficient names to equate across arms.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(len(names), n_coef0 + n_coef1)``.
    """
    j0 = spec0.n_coef
    rows = np.zeros((len(names), j0 + spec1.n_coef))
    for i, name in enumerate(names):
        if name not in spec0.names or name not in spec1.names:
            raise ValueError(
                f"'equal_coef' term {name!r} must appear in both m0 and m1; "
                f"m0 has {spec0.names}, m1 has {spec1.names}"
            )
        rows[i, spec0.names.index(name)] = 1.0
        rows[i, j0 + spec1.names.index(name)] = -1.0
    return rows


def _pad(mat: sp.csr_matrix | NDArray[np.float64], n_slack: int) -> sp.csr_matrix:
    m = sp.csr_matrix(mat)
    if n_slack == 0:
        return m
    return sp.csr_matrix(sp.hstack([sp.csr_matrix((m.shape[0], n_slack)), m]))


def build_constraints(
    crit: Criterion,
    shape: ShapeConstraints,
    equal: NDArray[np.float64] | None = None,
) -> LinearConstraints:
    """Assemble the linear constraints shared by the criterion and bound problems.

    Parameters
    ----------
    crit : L1Criterion or LSCriterion
        The criterion, which fixes the variable layout.
    shape : ShapeConstraints
        Shape restrictions ``A theta <= b`` (on the coefficients only).
    equal : numpy.ndarray, optional
        Equality rows on the coefficients (``equal @ theta == 0``).

    Returns
    -------
    LinearConstraints
    """
    ns, j = crit.n_slack, crit.n_coef
    n = ns + j
    a_ub = _pad(shape.A, ns) if shape.n else None
    b_ub = shape.b if shape.n else None
    eq_blocks, eq_rhs = [], []
    if isinstance(crit, L1Criterion):
        s = len(crit.beta)
        eq_blocks.append(sp.hstack([-sp.identity(s), sp.identity(s), sp.csr_matrix(crit.gamma)]))
        eq_rhs.append(crit.beta)
    if equal is not None and len(equal):
        eq_blocks.append(_pad(equal, ns))
        eq_rhs.append(np.zeros(len(equal)))
    a_eq = sp.csr_matrix(sp.vstack(eq_blocks)) if eq_blocks else None
    b_eq = np.concatenate(eq_rhs) if eq_blocks else None
    lb = np.concatenate([np.zeros(ns), np.full(j, -np.inf)])
    return LinearConstraints(n, a_ub, b_ub, a_eq, b_eq, lb, None)


def _theta(crit: Criterion, x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x[crit.n_slack :]


def solve_criterion(
    crit: Criterion,
    cons: LinearConstraints,
    solver: str | None,
    options: dict[str, Any] | None,
) -> tuple[SolveResult, NDArray[np.float64] | None, float | None]:
    """Minimise the criterion subject to the constraints.

    Returns
    -------
    tuple
        Solver result, coefficient vector (or ``None``) and criterion value.
    """
    if isinstance(crit, L1Criterion):
        c = np.concatenate([np.ones(crit.n_slack), np.zeros(crit.n_coef)])
        res = solve_lp(c, cons, "min", solver, options)
    else:
        res = solve_qcqp(None, cons, (crit.G, crit.h, crit.const, None), "min", solver, options)
    if res.x is None:
        return res, None, None
    return res, _theta(crit, res.x), res.obj


def solve_bound(
    crit: Criterion,
    cons: LinearConstraints,
    gstar: NDArray[np.float64],
    crit_min: float,
    criterion_tol: float,
    sense: str,
    solver: str | None,
    options: dict[str, Any] | None,
) -> tuple[SolveResult, NDArray[np.float64] | None]:
    """Minimise or maximise the target subject to ``criterion <= (1 + tol) * crit_min``.

    Parameters
    ----------
    crit : L1Criterion or LSCriterion
        The criterion.
    cons : LinearConstraints
        Constraints from :func:`build_constraints`.
    gstar : numpy.ndarray
        Target coefficients ``(gstar0, gstar1)``.
    crit_min : float
        Minimum criterion value from :func:`solve_criterion`.
    criterion_tol : float
        Relative slack on the criterion.
    sense : {"min", "max"}
        Which bound to compute.
    solver, options
        Passed to the solver.

    Returns
    -------
    tuple
        Solver result and the coefficient vector at the bound (or ``None``).
    """
    limit = crit_min * (1.0 + criterion_tol)
    c = np.concatenate([np.zeros(crit.n_slack), gstar])
    if isinstance(crit, L1Criterion):
        row = sp.csr_matrix(np.concatenate([np.ones(crit.n_slack), np.zeros(crit.n_coef)]))
        a_ub = sp.csr_matrix(sp.vstack([row, cons.A_ub])) if cons.A_ub is not None else row
        b_ub = np.concatenate([[limit], cons.b_ub]) if cons.b_ub is not None else np.array([limit])
        bounded = LinearConstraints(cons.n, a_ub, b_ub, cons.A_eq, cons.b_eq, cons.lb, cons.ub)
        res = solve_lp(c, bounded, sense, solver, options)
    else:
        res = solve_qcqp(c, cons, (crit.G, crit.h, crit.const, limit), sense, solver, options)
    if res.x is None:
        return res, None
    return res, _theta(crit, res.x)
