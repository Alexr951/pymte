"""The audit procedure for partial identification.

Shape restrictions are first imposed on a small initial grid. After solving
for the bounds, the restrictions are checked on the finer audit grid; grid
points where either bounding solution violates a restriction are added to
the constraint set and the problem is solved again, until no violations
remain or ``audit_max`` rounds have been performed. This follows the R
package closely, including the construction of the grids, the rules for
selecting which violations to add and for terminating when the audit cannot
make progress.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from pymte.lp import (
    STATUS_STRINGS,
    Criterion,
    SolveResult,
    bound,
    criterion_min,
    lp_setup,
    magnitude,
)
from pymte.monobound import KINDS, Grids, ShapeConstraints, combinemonobound, genmonobound_a
from pymte.mtr import MTRSpec


class AuditError(RuntimeError):
    """The criterion or a bound problem could not be solved.

    Attributes
    ----------
    status : int
        Canonical solver status code of the failed problem.
    stage : {"criterion", "bound"}
        Which problem failed.
    grids : Grids or None
        The grids in use, so that a retry can keep the audit grid.
    """

    def __init__(
        self, message: str, status: int, stage: str = "criterion", grids: Grids | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.stage = stage
        self.grids = grids


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
    grids : Grids
        The audit grid and the initial constraint grid that were used.
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
    grids: Grids
    status: dict[str, int]
    runtime: dict[str, float]
    messages: list[str] = field(default_factory=list)


def status_string(status: int) -> str:
    """Describe a canonical solver status code."""
    return STATUS_STRINGS.get(status, "unknown")


def fmt_result(x: float) -> str:
    """Format numbers the way ``print.ivmte`` does."""
    if x == 0:
        return "0"
    if abs(x) < 1:
        return f"{x:.7g}"
    if abs(x) < 1e7:
        return f"{round(x, 4):.7g}"
    return f"{x:.7e}"


# -- grids --------------------------------------------------------------------


def rhalton(n: int, base: int = 2) -> NDArray[np.float64]:
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


def _u_grid(n: int) -> NDArray[np.float64]:
    """Sorted grid ``{0, 1} + rhalton(n)`` rounded to 8 decimals, as in the R package."""
    if n <= 0:
        return np.array([0.0, 1.0])
    return np.sort(np.concatenate([[0.0, 1.0], np.round(rhalton(n), 8)]))


def _gen_grids(
    data: pd.DataFrame,
    xvars: Sequence[str],
    *,
    initgrid_nx: int,
    initgrid_nu: int,
    audit_nx: int,
    audit_nu: int,
    initgrid_x: pd.DataFrame | None,
    initgrid_u: ArrayLike | None,
    audit_x: pd.DataFrame | None,
    audit_u: ArrayLike | None,
    rng: np.random.Generator,
) -> Grids:
    """Sample the audit grid and the initial constraint grid as the R package does.

    The audit grid in ``x`` is a uniform sample (capped by the support size)
    of the distinct covariate rows, the initial grid a subsample of it; the
    ``u`` grids are Halton points plus the end points. Explicit grids
    override the sampled ones.
    """
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
        a_u = _u_grid(audit_nu)
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
        i_u = _u_grid(initgrid_nu)
    else:
        take = min(len(a_u), initgrid_nu)
        i_u = np.sort(rng.choice(a_u, take, replace=False))
    return Grids(support, a_u, np.asarray(init_index, dtype=np.intp), i_u)


# -- violations ---------------------------------------------------------------


def _violations(
    cons: ShapeConstraints, thetas: Sequence[NDArray[np.float64]], tol: float
) -> pd.DataFrame:
    """Constraint rows violated by any of the candidate solutions.

    Returns a frame with columns ``row`` (position in ``cons``), ``kind``,
    ``x_index``, ``u`` and ``diff`` (largest residual across the solutions).
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


# -- the audit loop -----------------------------------------------------------


def _relaxed_tol(tol: float) -> float:
    # R: (tol / 10^magnitude) * 10^(magnitude / 2), e.g. 1e-6 -> 1e-3.
    mag = magnitude(tol)
    if mag is None:
        return tol
    return float((tol / 10**mag) * 10 ** (mag / 2))


_DEFAULT_NOTE = {
    "m0.lb": "min. observed outcome by default",
    "m1.lb": "min. observed outcome by default",
    "m0.ub": "max. observed outcome by default",
    "m1.ub": "max. observed outcome by default",
}


def _describe_restriction(kind: str, restrictions: dict[str, Any], defaults: Sequence[str]) -> str:
    key = kind.replace(".", "_")
    value = restrictions[key]
    if isinstance(value, bool):
        return f"{key} = {value}"
    text = f"{key} = {round(float(value), 6)}"
    if key in defaults:
        text += f" ({_DEFAULT_NOTE[kind]})"
    return text


def _infeasibility_message(
    status: int,
    crit: Criterion,
    equal: NDArray[np.float64] | None,
    cons: ShapeConstraints,
    restrictions: dict[str, Any],
    defaults: Sequence[str],
    audit_tol: float,
    solver: str | None,
    options: dict[str, Any] | None,
) -> str:
    """Diagnose an infeasible criterion problem as the R package does.

    The criterion is minimised again without the shape restrictions and the
    restrictions that solution violates are named, since incoherent shape
    restrictions are the likely cause of an empty parameter space.
    """
    proved = "infeasible" if status == 2 else "infeasible or unbounded"
    message = f"No solution since the solver proved the model was {proved}."
    _, theta, _ = criterion_min(crit, lp_setup(crit, None, equal), solver, options)
    if theta is not None:
        violated = cons.residuals(theta) > audit_tol
        kinds = [k for k in KINDS if k in set(cons.kind[violated])]
        if kinds:
            named = ", ".join(_describe_restriction(k, restrictions, defaults) for k in kinds)
            message += (
                " The model should only be infeasible if the implied parameter space is "
                "empty. The likely cause of an empty parameter space is incoherent shape "
                f"restrictions. For example, {named} are all set simultaneously. Try "
                "changing the shape constraints on the MTR functions."
            )
    if status == 3:
        message += (
            " The model may be unbounded if the initial grid is too small. Try increasing "
            "the parameters 'initgrid_nx' and 'initgrid_nu'."
        )
    return message


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
    if res.x is None:
        raise AuditError(
            "No solution provided by the solver when minimizing the criterion. " + alt,
            res.status,
        )


def audit(
    spec0: MTRSpec,
    spec1: MTRSpec,
    crit: Criterion,
    gstar: NDArray[np.float64],
    data: pd.DataFrame,
    restrictions: dict[str, Any],
    *,
    defaults: Sequence[str] = (),
    equal: NDArray[np.float64] | None = None,
    criterion_tol: float = 1e-4,
    audit_tol: float = 1e-6,
    audit_add: int = 100,
    audit_max: int = 25,
    initgrid_nx: int = 20,
    initgrid_nu: int = 20,
    audit_nx: int = 2500,
    audit_nu: int = 25,
    initgrid_x: pd.DataFrame | None = None,
    initgrid_u: ArrayLike | None = None,
    audit_x: pd.DataFrame | None = None,
    audit_u: ArrayLike | None = None,
    audit_grid: Grids | None = None,
    rng: np.random.Generator | None = None,
    solver: str | None = None,
    solver_options_criterion: dict[str, Any] | None = None,
    solver_options_bounds: dict[str, Any] | None = None,
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
    data : pandas.DataFrame
        Estimation sample, from which the covariate grids are drawn.
    restrictions : dict
        Shape restrictions, see :func:`pymte.monobound.genmonobound_a`.
    defaults : sequence of str, optional
        Keys of ``restrictions`` that were not set by the user but taken
        from the range of the outcome; named as such in error messages.
    equal : numpy.ndarray, optional
        Equality rows on the coefficients.
    criterion_tol : float
        Relative tolerance on the criterion in the bound problems.
    audit_tol : float
        Violations below this size are ignored.
    audit_add : int
        Maximum number of violated points added per round (see
        :func:`select_violations`).
    audit_max : int
        Maximum number of rounds.
    initgrid_nx, initgrid_nu, audit_nx, audit_nu : int
        Sizes of the initial constraint grid and the audit grid.
    initgrid_x, initgrid_u, audit_x, audit_u : optional
        Explicit grids overriding the sampled ones; 0 and 1 are always part
        of the ``u`` grids.
    audit_grid : Grids, optional
        Reuse these grids instead of drawing new ones (bootstrap replicates).
    rng : numpy.random.Generator, optional
        Source of randomness for sampling covariate rows.
    solver : str, optional
        Solver name, see :mod:`pymte.lp`.
    solver_options_criterion, solver_options_bounds : dict, optional
        Options for the criterion and the bound problems.
    log : list of str, optional
        Progress messages are appended here.

    Returns
    -------
    AuditResult

    Raises
    ------
    AuditError
        When the criterion or a bound problem has no usable solution. An
        infeasible criterion problem is diagnosed by solving it again
        without the shape restrictions and naming the restrictions that
        solution violates.
    """
    log = log if log is not None else []
    alt = "Try relaxing 'criterion_tol' or the shape restrictions."
    if audit_grid is None:
        xvars = sorted(set(spec0.covariates) | set(spec1.covariates))
        audit_grid = _gen_grids(
            data, xvars, initgrid_nx=initgrid_nx, initgrid_nu=initgrid_nu, audit_nx=audit_nx,
            audit_nu=audit_nu, initgrid_x=initgrid_x, initgrid_u=initgrid_u, audit_x=audit_x,
            audit_u=audit_u, rng=rng or np.random.default_rng(),
        )  # fmt: skip
    grids = audit_grid
    support = grids.support
    all_x = np.arange(len(support))
    current = genmonobound_a(spec0, spec1, support, grids.init_index, grids.init_u, restrictions)
    audit_cons = genmonobound_a(spec0, spec1, support, all_x, grids.audit_u, restrictions)
    full_grid = len(grids.init_index) == len(support) and len(grids.init_u) == len(grids.audit_u)
    log.append("    Generating initial constraint grid...")

    prev: pd.DataFrame | None = None
    same = 0
    count = 1
    while True:
        log.append(f"\n    Audit count: {count}")
        model = lp_setup(crit, current, equal)
        crit_res, theta_crit, crit_min = criterion_min(
            crit, model, solver, solver_options_criterion
        )
        if crit_res.status in (2, 3):
            raise AuditError(
                _infeasibility_message(
                    crit_res.status, crit, equal, current, restrictions, defaults, audit_tol,
                    solver, solver_options_criterion,
                ),
                crit_res.status,
                "criterion",
                grids,
            )  # fmt: skip
        _check_criterion(crit_res, alt)
        assert theta_crit is not None and crit_min is not None
        log.append(f"    Minimum criterion: {fmt_result(crit_min)}")
        log.append("    Obtaining bounds...")
        (min_res, theta_min), (max_res, theta_max) = bound(
            crit, model, gstar, crit_min, criterion_tol, solver, solver_options_bounds
        )
        if theta_min is None or theta_max is None or min_res.obj is None or max_res.obj is None:
            bad = min_res if theta_min is None else max_res
            raise AuditError(
                f"The {'minimization' if bad is min_res else 'maximization'} problem for the "
                f"bounds returned no solution (status: {bad.status_str}). {alt}",
                bad.status,
                "bound",
                grids,
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
            grids=grids,
            status={"criterion": crit_res.status, "min": min_res.status, "max": max_res.status},
            runtime={
                "criterion": crit_res.runtime,
                "min": min_res.runtime,
                "max": max_res.runtime,
            },
            messages=log,
        )
        viol = _violations(audit_cons, [theta_min, theta_max], audit_tol)
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
        current = combinemonobound(current, audit_cons.subset(chosen["row"].to_numpy()))
        count += 1
    log.append(
        f"Bounds on the target parameter: [{fmt_result(result.lower)}, {fmt_result(result.upper)}]"
    )
    return result
