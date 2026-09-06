"""The audit procedure for partial identification.

Shape restrictions are first imposed on a small initial grid. After solving
for the bounds, the restrictions are checked on the finer audit grid; grid
points where either bounding solution violates a restriction are added to
the constraint set and the problem is solved again, until no violations
remain or ``audit_max`` rounds have been performed. This follows the R
package closely, including the rules for selecting which violations to add
and for terminating when the audit cannot make progress.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from ivmte.lp import Criterion, build_constraints, solve_bound, solve_criterion
from ivmte.mtr import MTRSpec
from ivmte.shape import Grids, ShapeConstraints, select_violations, shape_constraints, violations
from ivmte.solvers import SolveResult


class AuditError(RuntimeError):
    """The criterion or a bound problem could not be solved."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class AuditResult:
    """Outcome of the audit procedure.

    Attributes
    ----------
    lower, upper : float
        Bounds on the target parameter.
    theta_min, theta_max, theta_crit : numpy.ndarray
        Stacked MTR coefficients at the lower bound, the upper bound and the
        criterion minimum of the final round.
    criterion : float
        Minimum criterion in the final round.
    audit_count : int
        Number of rounds performed.
    violations : pandas.DataFrame
        Violations found on the audit grid in the final round (empty when
        the audit finished cleanly).
    constraints : ShapeConstraints
        Shape restrictions imposed in the final round.
    status : dict
        Solver status codes for ``"criterion"``, ``"min"`` and ``"max"``.
    runtime : dict
        Solver time in seconds for the same three problems (final round).
    messages : list of str
        Progress log, in the wording of the R package.
    """

    lower: float
    upper: float
    theta_min: NDArray[np.float64]
    theta_max: NDArray[np.float64]
    theta_crit: NDArray[np.float64]
    criterion: float
    audit_count: int
    violations: pd.DataFrame
    constraints: ShapeConstraints
    status: dict[str, int]
    runtime: dict[str, float]
    messages: list[str] = field(default_factory=list)


def _fmt(x: float) -> str:
    """Format numbers the way ``print.ivmte`` does."""
    if x == 0:
        return "0"
    if abs(x) < 1:
        return f"{x:.7g}"
    if abs(x) < 1e7:
        return f"{round(x, 4):.7g}"
    return f"{x:.7e}"


def _stack(a: ShapeConstraints, b: ShapeConstraints) -> ShapeConstraints:
    if a.n == 0:
        return b
    if b.n == 0:
        return a
    return ShapeConstraints(
        sp.csr_matrix(sp.vstack([a.A, b.A])),
        np.concatenate([a.b, b.b]),
        np.concatenate([a.kind, b.kind]),
        np.concatenate([a.x_index, b.x_index]),
        np.concatenate([a.u, b.u]),
    )


def _relaxed_tol(tol: float) -> float:
    # R: (tol / 10^magnitude) * 10^(magnitude / 2), e.g. 1e-6 -> 1e-3.
    mag = math.floor(math.log10(tol))
    return float((tol / 10**mag) * 10 ** (mag / 2))


def _check_criterion(res: SolveResult, alt: str) -> None:
    if res.status == 4:
        raise AuditError(
            "No solution to minimizing the criterion since the model is unbounded. " + alt,
            res.status,
        )
    if res.status == 5:
        raise AuditError(
            "No solution to minimizing the criterion due to numerical issues. " + alt, res.status
        )
    if res.status in (2, 3):
        raise AuditError(
            "The problem is infeasible: the shape restrictions cannot all be satisfied "
            "together. Relax the bounds or monotonicity restrictions on the MTRs. " + alt,
            res.status,
        )
    if res.x is None:
        raise AuditError(
            "No solution provided by the solver when minimizing the criterion. " + alt,
            res.status,
        )


def run_audit(
    spec0: MTRSpec,
    spec1: MTRSpec,
    crit: Criterion,
    gstar: NDArray[np.float64],
    grids: Grids,
    restrictions: dict[str, Any],
    *,
    equal: NDArray[np.float64] | None = None,
    criterion_tol: float = 1e-4,
    audit_tol: float = 1e-6,
    audit_add: int = 100,
    audit_max: int = 25,
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    log: list[str] | None = None,
) -> AuditResult:
    """Compute bounds on the target parameter with the audit procedure.

    Parameters
    ----------
    spec0, spec1 : MTRSpec
        MTR specifications.
    crit : L1Criterion or LSCriterion
        Criterion measuring the fit to the data.
    gstar : numpy.ndarray
        Stacked target coefficients ``(gstar0, gstar1)``.
    grids : Grids
        Initial and audit grids.
    restrictions : dict
        Shape restrictions, see :func:`ivmte.shape.shape_constraints`.
    equal : numpy.ndarray, optional
        Equality rows on the coefficients.
    criterion_tol : float
        Relative tolerance on the criterion in the bound problems.
    audit_tol : float
        Violations below this size are ignored.
    audit_add : int
        Maximum number of violated points added per round (see
        :func:`ivmte.shape.select_violations`).
    audit_max : int
        Maximum number of rounds.
    solver, solver_options
        Passed to the solver interface.
    log : list of str, optional
        Progress messages are appended here.

    Returns
    -------
    AuditResult

    Raises
    ------
    AuditError
        When the criterion or a bound problem has no usable solution.
    """
    log = log if log is not None else []
    alt = "Try relaxing 'criterion_tol' or the shape restrictions."
    support = grids.support
    all_x = np.arange(len(support))
    current = shape_constraints(
        spec0, spec1, support, grids.init_index, grids.init_u, restrictions
    )
    audit_cons = shape_constraints(spec0, spec1, support, all_x, grids.audit_u, restrictions)
    full_grid = len(grids.init_index) == len(support) and len(grids.init_u) == len(grids.audit_u)
    log.append("    Generating initial constraint grid...")

    prev: pd.DataFrame | None = None
    same = 0
    count = 1
    while True:
        log.append(f"\n    Audit count: {count}")
        cons = build_constraints(crit, current, equal)
        crit_res, theta_crit, crit_min = solve_criterion(crit, cons, solver, solver_options)
        _check_criterion(crit_res, alt)
        assert theta_crit is not None and crit_min is not None
        log.append(f"    Minimum criterion: {_fmt(crit_min)}")
        log.append("    Obtaining bounds...")
        min_res, theta_min = solve_bound(
            crit, cons, gstar, crit_min, criterion_tol, "min", solver, solver_options
        )
        max_res, theta_max = solve_bound(
            crit, cons, gstar, crit_min, criterion_tol, "max", solver, solver_options
        )
        if theta_min is None or theta_max is None or min_res.obj is None or max_res.obj is None:
            bad = min_res if theta_min is None else max_res
            raise AuditError(
                f"The {'minimization' if bad is min_res else 'maximization'} problem for the "
                f"bounds returned no solution (status: {bad.status_str}). {alt}",
                bad.status,
            )
        result = AuditResult(
            lower=float(min_res.obj),
            upper=float(max_res.obj),
            theta_min=theta_min,
            theta_max=theta_max,
            theta_crit=theta_crit,
            criterion=float(crit_min),
            audit_count=count,
            violations=pd.DataFrame(),
            constraints=current,
            status={"criterion": crit_res.status, "min": min_res.status, "max": max_res.status},
            runtime={
                "criterion": crit_res.runtime,
                "min": min_res.runtime,
                "max": max_res.runtime,
            },
            messages=log,
        )
        viol = violations(audit_cons, [theta_min, theta_max], audit_tol)
        if full_grid and len(viol):
            # Nothing can be added; the violations are solver precision.
            new_tol = _relaxed_tol(audit_tol)
            worst = viol["diff"].max()
            viol = viol[viol["diff"] > new_tol]
            warnings.warn(
                f"Violations of the shape constraints were found although the initial grid "
                f"equals the audit grid. The audit tolerance was raised from {audit_tol:g} "
                f"to {new_tol:g}; the largest violation was {worst:.3g}.",
                stacklevel=2,
            )
            result.violations = viol
            break
        if prev is not None and len(viol) and len(prev) == len(viol):
            same = (
                same + 1 if viol.reset_index(drop=True).equals(prev.reset_index(drop=True)) else 0
            )
        prev = viol
        if same >= 2:
            warnings.warn(
                "Audit is unable to resolve violations: the same set of violations have "
                "persisted for three iterations. This can occur if 'audit_tol' differs from "
                "the tolerance of the solver. Audit is terminated.",
                stacklevel=2,
            )
            result.violations = viol
            break
        if len(viol) == 0:
            log.append("    Violations: 0")
            log.append("    Audit finished.\n")
            break
        log.append(f"    Violations:  {len(viol)}")
        if count == audit_max:
            warnings.warn(
                f"Audit finished: maximum number of audits (audit_max = {audit_max}) reached. "
                "Try increasing audit_max.",
                stacklevel=2,
            )
            result.violations = viol
            break
        chosen = select_violations(viol, audit_add)
        log.append(f"    Expanding constraint grid to include {len(chosen)} additional points...")
        current = _stack(current, audit_cons.subset(chosen["row"].to_numpy()))
        count += 1
    log.append(f"Bounds on the target parameter: [{_fmt(result.lower)}, {_fmt(result.upper)}]")
    return result
