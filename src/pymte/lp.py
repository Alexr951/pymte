"""Optimisation problems for partial identification and the solver interface.

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
approach. The functions follow the R package: ``lp_setup`` assembles the
constraints, ``lp_setup_criterion`` and ``lp_setup_bound`` add the objective
and the criterion constraint, ``criterion_min`` and ``bound`` solve, and
``lp_setup_criterion_boot`` builds the program of the specification test.
The ``qp_setup*`` functions are the regression-approach counterparts.

Linear programs are solved by HiGHS through :func:`scipy.optimize.linprog`.
Quadratic and quadratically constrained programs go through CVXPY, with
Clarabel by default. Gurobi and MOSEK can be selected for either problem
class when their Python packages are installed; they are used automatically
when available, in the same order of preference as the R package.
"""

from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

from pymte.monobound import ShapeConstraints
from pymte.mtr import MTRSpec

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


# -- solver interface ---------------------------------------------------------


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
    """Linear constraints ``A_ub x <= b_ub``, ``A_eq x == b_eq``, ``lb <= x <= ub``.

    This is the ``model`` object of the R package without its objective.
    """

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

Quad = tuple[NDArray[np.float64], NDArray[np.float64], float, float | None]


def run_highs(
    c: NDArray[np.float64],
    model: LinearConstraints,
    sense: str,
    options: dict[str, Any] | None,
) -> SolveResult:
    """Solve a linear program with HiGHS (:func:`scipy.optimize.linprog`)."""
    sign = 1.0 if sense == "min" else -1.0
    lb, ub = model.bounds()
    start = time.perf_counter()
    res = linprog(
        sign * c,
        A_ub=model.A_ub,
        b_ub=model.b_ub,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=np.column_stack([lb, ub]),
        method="highs",
        options=options,
    )
    runtime = time.perf_counter() - start
    status = _HIGHS_STATUS.get(int(res.status), 0)
    x = np.asarray(res.x, dtype=float) if res.x is not None else None
    obj = float(sign * res.fun) if res.fun is not None and x is not None else None
    return SolveResult(x, obj, status, runtime)


def run_cvxpy(
    c: NDArray[np.float64] | None,
    model: LinearConstraints,
    sense: str,
    solver: str,
    options: dict[str, Any] | None,
    quad: Quad | None = None,
) -> SolveResult:
    """Solve a linear, quadratic or quadratically constrained program through CVXPY.

    The quadratic form is ``f(x) = ||G x - h||^2 + const`` for
    ``quad = (G, h, const, rhs)``. With ``c=None`` the problem is ``min f(x)``;
    otherwise ``min/max c @ x`` under ``f(x) <= rhs`` when ``quad`` is given.
    """
    import cvxpy as cp

    x = cp.Variable(model.n)
    constraints = []
    if model.A_ub is not None and model.A_ub.shape[0] > 0:
        constraints.append(model.A_ub @ x <= model.b_ub)
    if model.A_eq is not None and model.A_eq.shape[0] > 0:
        constraints.append(model.A_eq @ x == model.b_eq)
    lb, ub = model.bounds()
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


def run_lp(
    c: ArrayLike,
    model: LinearConstraints,
    sense: str = "min",
    solver: str | None = None,
    options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve ``min/max c @ x`` subject to linear constraints.

    Parameters
    ----------
    c : array_like
        Objective coefficients.
    model : LinearConstraints
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
    if solver == "highs":
        return run_highs(cvec, model, sense, options)
    return run_cvxpy(cvec, model, sense, solver, options)


def run_qcqp(
    c: ArrayLike | None,
    model: LinearConstraints,
    quad: Quad,
    sense: str = "min",
    solver: str | None = None,
    options: dict[str, Any] | None = None,
) -> SolveResult:
    """Solve a least-squares QP or a linear program with one quadratic constraint.

    Parameters
    ----------
    c : array_like or None
        Linear objective, or ``None`` to minimise the quadratic form.
    model : LinearConstraints
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
    return run_cvxpy(cvec, model, sense, solver, options, quad=quad)


def magnitude(x: float) -> int:
    """Order of magnitude of ``x`` (``floor(log10 |x|)``), as in the R package."""
    return int(math.floor(math.log10(abs(x))))


# -- criteria -----------------------------------------------------------------


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
    ``G`` of shape ``(r, J)``, ``r`` the rank of ``X``. Build it with
    :func:`qp_setup`.
    """

    G: NDArray[np.float64]
    h: NDArray[np.float64]
    const: float

    @property
    def n_slack(self) -> int:
        """No slack variables in the regression approach."""
        return 0

    @property
    def n_coef(self) -> int:
        """Number of MTR coefficients."""
        return int(self.G.shape[1])


Criterion = L1Criterion | LSCriterion


def qp_setup(x: NDArray[np.float64], y: NDArray[np.float64]) -> LSCriterion:
    """Compress the regression design ``x`` and outcome ``y`` into an :class:`LSCriterion`."""
    n = len(y)
    m = x.T @ x / n
    b = x.T @ y / n
    w, v = np.linalg.eigh(m)
    keep = w > w.max() * 1e-12
    g = np.sqrt(w[keep])[:, None] * v[:, keep].T  # G'G = M
    h = np.linalg.pinv(g.T) @ b
    const = float(y @ y / n - h @ h)
    return LSCriterion(g, h, const)


def qp_setup_criterion(crit: LSCriterion) -> Quad:
    """Quadratic form of the criterion problem (no right-hand side)."""
    return crit.G, crit.h, crit.const, None


def qp_setup_bound(crit: LSCriterion, criterion_min: float, criterion_tol: float) -> Quad:
    """Quadratic constraint ``criterion <= (1 + tol) * criterion_min`` of the bound problems."""
    return crit.G, crit.h, crit.const, criterion_min * (1.0 + criterion_tol)


# -- problem construction -----------------------------------------------------


def lp_setup_equal_coef(
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


def lp_setup(
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


def lp_setup_criterion(crit: L1Criterion) -> NDArray[np.float64]:
    """Objective of the criterion problem: the sum of the slack variables."""
    return np.concatenate([np.ones(crit.n_slack), np.zeros(crit.n_coef)])


def lp_setup_bound(
    model: LinearConstraints, crit: L1Criterion, criterion_min: float, criterion_tol: float
) -> LinearConstraints:
    """Add the constraint ``criterion <= (1 + tol) * criterion_min`` to the LP."""
    limit = criterion_min * (1.0 + criterion_tol)
    row = sp.csr_matrix(lp_setup_criterion(crit))
    a_ub = sp.csr_matrix(sp.vstack([row, model.A_ub])) if model.A_ub is not None else row
    b_ub = np.concatenate([[limit], model.b_ub]) if model.b_ub is not None else np.array([limit])
    return LinearConstraints(model.n, a_ub, b_ub, model.A_eq, model.b_eq, model.lb, model.ub)


def _theta(crit: Criterion, x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x[crit.n_slack :]


def criterion_min(
    crit: Criterion,
    model: LinearConstraints,
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
        res = run_lp(lp_setup_criterion(crit), model, "min", solver, options)
    else:
        res = run_qcqp(None, model, qp_setup_criterion(crit), "min", solver, options)
    if res.x is None:
        return res, None, None
    return res, _theta(crit, res.x), res.obj


def bound(
    crit: Criterion,
    model: LinearConstraints,
    gstar: NDArray[np.float64],
    criterion_min: float,
    criterion_tol: float,
    solver: str | None,
    options: dict[str, Any] | None,
) -> tuple[
    tuple[SolveResult, NDArray[np.float64] | None], tuple[SolveResult, NDArray[np.float64] | None]
]:
    """Minimise and maximise the target subject to ``criterion <= (1 + tol) * criterion_min``.

    Parameters
    ----------
    crit : L1Criterion or LSCriterion
        The criterion.
    model : LinearConstraints
        Constraints from :func:`lp_setup`.
    gstar : numpy.ndarray
        Target coefficients ``(gstar0, gstar1)``.
    criterion_min : float
        Minimum criterion value from :func:`criterion_min`.
    criterion_tol : float
        Relative slack on the criterion.
    solver, options
        Passed to the solver.

    Returns
    -------
    tuple
        ``((min_result, theta_min), (max_result, theta_max))``; a coefficient
        vector is ``None`` when the solver returned no solution.
    """
    c = np.concatenate([np.zeros(crit.n_slack), gstar])
    out = []
    for sense in ("min", "max"):
        if isinstance(crit, L1Criterion):
            res = run_lp(
                c,
                lp_setup_bound(model, crit, criterion_min, criterion_tol),
                sense,
                solver,
                options,
            )
        else:
            res = run_qcqp(
                c,
                model,
                qp_setup_bound(crit, criterion_min, criterion_tol),
                sense,
                solver,
                options,
            )
        out.append((res, None if res.x is None else _theta(crit, res.x)))
    return out[0], out[1]


def lp_setup_criterion_boot(
    crit: L1Criterion,
    orig: L1Criterion,
    orig_min: float,
    criterion_tol: float,
    shape: ShapeConstraints,
    equal: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], LinearConstraints]:
    """Linear program of the bootstrap misspecification test.

    The absolute-deviation criterion of the resample is minimised over the
    coefficients whose original-sample criterion is at most
    ``(1 + criterion_tol) * orig_min``, under the shape restrictions. This
    is the statistic of the misspecification test of the R package.

    Parameters
    ----------
    crit : L1Criterion
        Moments of the resample.
    orig : L1Criterion
        Moments of the original sample.
    orig_min : float
        Minimum criterion in the original sample.
    criterion_tol : float
        Relative slack on the original criterion.
    shape : ShapeConstraints
        Shape restrictions in force (on the coefficients only).
    equal : numpy.ndarray, optional
        Equality rows on the coefficients.

    Returns
    -------
    tuple
        Objective vector and constraints; solve with :func:`run_lp`.
    """
    s, j = len(crit.beta), crit.n_coef
    n_slack = 4 * s
    eye = sp.identity(s)
    zero = sp.csr_matrix((s, 2 * s))
    a_eq = sp.vstack(
        [
            sp.hstack([-eye, eye, zero, sp.csr_matrix(crit.gamma)]),
            sp.hstack([zero, -eye, eye, sp.csr_matrix(orig.gamma)]),
        ]
    )
    b_eq = np.concatenate([crit.beta, orig.beta])
    if equal is not None and len(equal):
        a_eq = sp.vstack([a_eq, _pad(equal, n_slack)])
        b_eq = np.concatenate([b_eq, np.zeros(len(equal))])
    row = sp.csr_matrix(np.concatenate([np.zeros(2 * s), np.ones(2 * s), np.zeros(j)]))
    a_ub = sp.vstack([row, _pad(shape.A, n_slack)]) if shape.n else row
    b_ub = (
        np.concatenate([[orig_min * (1.0 + criterion_tol)], shape.b])
        if shape.n
        else np.array([orig_min * (1.0 + criterion_tol)])
    )
    model = LinearConstraints(
        n_slack + j,
        sp.csr_matrix(a_ub),
        b_ub,
        sp.csr_matrix(a_eq),
        b_eq,
        np.concatenate([np.zeros(n_slack), np.full(j, -np.inf)]),
        None,
    )
    c = np.concatenate([np.ones(2 * s), np.zeros(2 * s + j)])
    return c, model
