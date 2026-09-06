"""Thin interface to the optimisers.

Linear programs are solved by HiGHS through :func:`scipy.optimize.linprog`.
Quadratic and quadratically constrained programs go through CVXPY, with
Clarabel by default. Gurobi and MOSEK can be selected for either problem
class when their Python packages are installed; they are used automatically
when available, in the same order of preference as the R package.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

STATUS_STRINGS = {
    0: "unknown",
    1: "optimal",
    2: "infeasible",
    3: "infeasible or unbounded",
    4: "unbounded",
    5: "numerical error",
    6: "suboptimal",
}
LP_SOLVERS = ("highs", "gurobi", "mosek")
QCQP_SOLVERS = ("clarabel", "gurobi", "mosek")


@dataclass(frozen=True)
class SolveResult:
    """Outcome of one optimisation.

    Attributes
    ----------
    x : numpy.ndarray or None
        Solution vector, ``None`` when the solver returned none.
    obj : float or None
        Objective value at ``x``.
    status : int
        Canonical status code: 0 unknown, 1 optimal, 2 infeasible,
        3 infeasible or unbounded, 4 unbounded, 5 numerical error,
        6 suboptimal. These are the codes used by the R package.
    runtime : float
        Wall-clock seconds spent in the solver.
    """

    x: NDArray[np.float64] | None
    obj: float | None
    status: int
    runtime: float

    @property
    def status_str(self) -> str:
        """Human-readable status."""
        return STATUS_STRINGS.get(self.status, "unknown")

    @property
    def ok(self) -> bool:
        """Whether a usable solution was returned (optimal or suboptimal)."""
        return self.x is not None and self.status in (1, 6)


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def default_solver(qcqp: bool = False) -> str:
    """Pick the solver: Gurobi, then MOSEK, then the open-source default."""
    if _has("gurobipy"):
        return "gurobi"
    if _has("mosek"):
        return "mosek"
    return "clarabel" if qcqp else "highs"


@dataclass
class LinearConstraints:
    """Linear constraints ``A_ub x <= b_ub``, ``A_eq x == b_eq``, ``lb <= x <= ub``."""

    n: int
    A_ub: sp.csr_matrix | None = None
    b_ub: NDArray[np.float64] | None = None
    A_eq: sp.csr_matrix | None = None
    b_eq: NDArray[np.float64] | None = None
    lb: NDArray[np.float64] | None = None
    ub: NDArray[np.float64] | None = None

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Variable bounds as two arrays (``-inf``/``inf`` when unbounded)."""
        lb = np.full(self.n, -np.inf) if self.lb is None else np.asarray(self.lb, dtype=float)
        ub = np.full(self.n, np.inf) if self.ub is None else np.asarray(self.ub, dtype=float)
        return lb, ub


_HIGHS_STATUS = {0: 1, 1: 6, 2: 2, 3: 4, 4: 5}


def solve_lp(
    c: ArrayLike,
    cons: LinearConstraints,
    sense: str = "min",
    solver: str | None = None,
    options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve ``min/max c @ x`` subject to linear constraints.

    Parameters
    ----------
    c : array_like
        Objective coefficients.
    cons : LinearConstraints
        Constraints and variable bounds.
    sense : {"min", "max"}
        Direction of optimisation.
    solver : {"highs", "gurobi", "mosek"}, optional
        Backend; default from :func:`default_solver`.
    options : dict, optional
        Passed to the backend unchanged (``linprog`` options for HiGHS,
        ``solver_opts`` for CVXPY backends).

    Returns
    -------
    SolveResult
    """
    solver = (solver or default_solver()).lower()
    if solver not in LP_SOLVERS:
        raise ValueError(f"solver must be one of {LP_SOLVERS}, got {solver!r}")
    cvec = np.asarray(c, dtype=float)
    sign = 1.0 if sense == "min" else -1.0
    if solver != "highs":
        return _solve_cvxpy(cvec, cons, sense, solver, options, quad=None)
    lb, ub = cons.bounds()
    start = time.perf_counter()
    res = linprog(
        sign * cvec,
        A_ub=cons.A_ub,
        b_ub=cons.b_ub,
        A_eq=cons.A_eq,
        b_eq=cons.b_eq,
        bounds=np.column_stack([lb, ub]),
        method="highs",
        options=options,
    )
    runtime = time.perf_counter() - start
    status = _HIGHS_STATUS.get(int(res.status), 0)
    x = np.asarray(res.x, dtype=float) if res.x is not None else None
    obj = float(sign * res.fun) if res.fun is not None and x is not None else None
    return SolveResult(x, obj, status, runtime)


def solve_qcqp(
    c: ArrayLike | None,
    cons: LinearConstraints,
    quad: tuple[NDArray[np.float64], NDArray[np.float64], float, float | None],
    sense: str = "min",
    solver: str | None = None,
    options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve a least-squares QP or a linear program with one quadratic constraint.

    The quadratic form is ``f(x) = ||G x - h||^2 + const``. With ``c=None``
    the problem is ``min f(x)`` subject to the linear constraints. Otherwise
    it is ``min/max c @ x`` subject to the linear constraints and
    ``f(x) <= rhs``.

    Parameters
    ----------
    c : array_like or None
        Linear objective, or ``None`` to minimise the quadratic form.
    cons : LinearConstraints
        Linear constraints and bounds.
    quad : tuple
        ``(G, h, const, rhs)``; ``rhs`` is ignored when ``c`` is ``None``.
    sense : {"min", "max"}
        Direction of optimisation for the linear objective.
    solver : {"clarabel", "gurobi", "mosek"}, optional
        Backend; default from :func:`default_solver`.
    options : dict, optional
        Passed to ``cvxpy.Problem.solve`` as solver options.

    Returns
    -------
    SolveResult
    """
    solver = (solver or default_solver(qcqp=True)).lower()
    if solver not in QCQP_SOLVERS:
        raise ValueError(f"solver must be one of {QCQP_SOLVERS}, got {solver!r}")
    cvec = None if c is None else np.asarray(c, dtype=float)
    return _solve_cvxpy(cvec, cons, sense, solver, options, quad=quad)


def _solve_cvxpy(
    c: NDArray[np.float64] | None,
    cons: LinearConstraints,
    sense: str,
    solver: str,
    options: dict[str, Any] | None,
    quad: tuple[NDArray[np.float64], NDArray[np.float64], float, float | None] | None,
) -> SolveResult:
    import cvxpy as cp

    x = cp.Variable(cons.n)
    constraints = []
    if cons.A_ub is not None and cons.A_ub.shape[0] > 0:
        constraints.append(cons.A_ub @ x <= cons.b_ub)
    if cons.A_eq is not None and cons.A_eq.shape[0] > 0:
        constraints.append(cons.A_eq @ x == cons.b_eq)
    lb, ub = cons.bounds()
    finite_lb, finite_ub = np.isfinite(lb), np.isfinite(ub)
    if finite_lb.any():
        constraints.append(x[np.flatnonzero(finite_lb)] >= lb[finite_lb])
    if finite_ub.any():
        constraints.append(x[np.flatnonzero(finite_ub)] <= ub[finite_ub])
    if c is None:
        if quad is None:
            raise ValueError("Either a linear objective or a quadratic form is required")
        g, h, const, _ = quad
        objective = cp.Minimize(cp.sum_squares(g @ x - h) + const)
    else:
        if quad is not None:
            g, h, const, rhs = quad
            constraints.append(cp.sum_squares(g @ x - h) + const <= rhs)
        objective = cp.Minimize(c @ x) if sense == "min" else cp.Maximize(c @ x)
    problem = cp.Problem(objective, constraints)
    start = time.perf_counter()
    try:
        problem.solve(solver=solver.upper(), **(options or {}))
    except cp.error.SolverError:
        return SolveResult(None, None, 5, time.perf_counter() - start)
    runtime = time.perf_counter() - start
    st = cp.settings
    status = {
        st.OPTIMAL: 1,
        st.OPTIMAL_INACCURATE: 6,
        st.USER_LIMIT: 6,
        st.INFEASIBLE: 2,
        st.INFEASIBLE_INACCURATE: 2,
        st.UNBOUNDED: 4,
        st.UNBOUNDED_INACCURATE: 4,
        st.INFEASIBLE_OR_UNBOUNDED: 3,
    }.get(problem.status, 0)
    if x.value is None:
        return SolveResult(None, None, status, runtime)
    xv = np.asarray(x.value, dtype=float).ravel()
    return SolveResult(xv, float(problem.value), status, runtime)
