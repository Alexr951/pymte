"""The :func:`ivmte` estimator."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from formulaic import Formula

from ivmte.audit import AuditError, AuditResult, run_audit
from ivmte.ivlike import MomentSet, build_moments
from ivmte.lp import Criterion, L1Criterion, LSCriterion, equal_coef_matrix
from ivmte.mtr import MTRSpec
from ivmte.point import gmm, least_squares
from ivmte.propensity import Propensity, fit_propensity, propensity_from_column
from ivmte.results import IVMTEResult
from ivmte.shape import Grids, build_grids
from ivmte.solvers import default_solver
from ivmte.weights import (
    TARGETS,
    TargetGammas,
    conventional_weights,
    custom_target_gammas,
    target_gammas_from_weights,
)


def _formula_vars(formula: str) -> set[str]:
    vars_: set[str] = set()
    for part in formula.replace("|", "+").split("~"):
        part = part.strip()
        if part:
            vars_ |= set(Formula(part).required_variables)
    return vars_


def _required_columns(
    data: pd.DataFrame, formulas: Sequence[str], extra: Sequence[str]
) -> list[str]:
    """List the data columns referenced by the formulas; unknown names such as ``u`` are ignored."""
    names: set[str] = set(extra)
    for f in formulas:
        names |= _formula_vars(f)
    return [c for c in data.columns if c in names]


def _regression_design(
    data: pd.DataFrame, spec0: MTRSpec, spec1: MTRSpec, prop: Propensity
) -> np.ndarray:
    """Design matrix of the regression approach (inverse propensity weighted integrals)."""
    d = np.asarray(data[prop.treat], dtype=float)
    p = prop.phat
    with np.errstate(divide="ignore", invalid="ignore"):
        w0 = np.where(d == 0, 1.0 / (1.0 - p), 0.0)
        w1 = np.where(d == 1, 1.0 / p, 0.0)
    x0 = spec0.gamma(data, p, 1.0, w0)
    x1 = spec1.gamma(data, 0.0, p, w1)
    return np.hstack([x0, x1])


def ivmte(
    data: pd.DataFrame,
    *,
    m0: str,
    m1: str,
    target: str | None = None,
    late_from: dict[str, Any] | None = None,
    late_to: dict[str, Any] | None = None,
    late_x: dict[str, Any] | None = None,
    genlate_lb: float | None = None,
    genlate_ub: float | None = None,
    target_weight0: Sequence[float | Callable[..., float]] | None = None,
    target_weight1: Sequence[float | Callable[..., float]] | None = None,
    target_knots0: Sequence[float | Callable[..., float]] = (),
    target_knots1: Sequence[float | Callable[..., float]] = (),
    uname: str = "u",
    m0_lb: float | None = None,
    m0_ub: float | None = None,
    m1_lb: float | None = None,
    m1_ub: float | None = None,
    mte_lb: float | None = None,
    mte_ub: float | None = None,
    m0_inc: bool = False,
    m0_dec: bool = False,
    m1_inc: bool = False,
    m1_dec: bool = False,
    mte_inc: bool = False,
    mte_dec: bool = False,
    equal_coef: str | None = None,
    ivlike: str | Sequence[str] | None = None,
    components: Sequence[Sequence[str] | None] | None = None,
    subset: str | Sequence[str | None] | None = None,
    propensity: str,
    link: str = "logit",
    treat: str | None = None,
    outcome: str | None = None,
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    criterion_tol: float = 1e-4,
    initgrid_nx: int = 20,
    initgrid_nu: int = 20,
    audit_nx: int = 2500,
    audit_nu: int = 25,
    initgrid_x: pd.DataFrame | None = None,
    initgrid_u: Sequence[float] | None = None,
    audit_x: pd.DataFrame | None = None,
    audit_u: Sequence[float] | None = None,
    audit_add: int = 100,
    audit_max: int = 25,
    audit_tol: float = 1e-6,
    point: bool | None = None,
    point_eyeweight: bool = False,
    seed: int | np.random.Generator | None = None,
    noisy: bool = False,
) -> IVMTEResult:
    """Estimate or bound a treatment parameter with the MTE framework.

    Parameters
    ----------
    data : pandas.DataFrame
        Estimation sample. Rows with missing values in any variable used are
        dropped with a warning.
    m0, m1 : str
        One-sided formulas for the MTR functions of the untreated and treated
        arms, in the unobservable ``uname`` and covariates; see
        :class:`ivmte.MTRSpec`.
    target : {"ate", "att", "atu", "late", "avglate", "genlate"}, optional
        Target parameter. Omit when defining a custom target through
        ``target_weight0``/``target_weight1``.
    late_from, late_to : dict, optional
        Instrument values defining the LATE, e.g. ``{"z": 1}``, ``{"z": 3}``.
    late_x : dict, optional
        Covariate values to condition the LATE or generalised LATE on.
    genlate_lb, genlate_ub : float, optional
        Limits in ``u`` of the generalised LATE.
    target_weight0, target_weight1 : sequence, optional
        Custom weights (numbers or callables of covariate columns) on the
        pieces of ``[0, 1]`` defined by ``target_knots0``/``target_knots1``.
    target_knots0, target_knots1 : sequence, optional
        Knots in ``u`` (numbers or callables of covariate columns).
    uname : str, default "u"
        Name of the unobservable in ``m0`` and ``m1``.
    m0_lb, m0_ub, m1_lb, m1_ub : float, optional
        Bounds on the MTR functions; default to the range of the outcome.
    mte_lb, mte_ub : float, optional
        Bounds on ``m1 - m0``. No default.
    m0_inc, m0_dec, m1_inc, m1_dec, mte_inc, mte_dec : bool
        Monotonicity restrictions in ``u``.
    equal_coef : str, optional
        One-sided formula of terms whose coefficients are equal across arms.
    ivlike : str or sequence of str, optional
        IV-like regressions, ``"y ~ d + x"`` (OLS) or ``"y ~ d + x | z + x"``
        (TSLS). Either ``ivlike`` or ``outcome`` is required.
    components : sequence, optional
        One entry per formula: coefficients to use (``"intercept"`` for the
        constant), or ``None`` for all.
    subset : str or sequence, optional
        One :meth:`pandas.DataFrame.eval` expression per formula selecting
        the estimation sample of that regression, or ``None``.
    propensity : str
        Two-sided formula ``"d ~ z + x"`` fitted with ``link``, or the name of
        a column holding propensity scores (then ``treat`` is required).
    link : {"logit", "probit", "linear"}, default "logit"
        Propensity model.
    treat : str, optional
        Treatment variable; inferred from the propensity formula.
    outcome : str, optional
        Outcome variable for the regression approach, which fits the MTRs to
        the conditional means of the outcome.
    solver : str, optional
        ``"highs"``/``"clarabel"`` (defaults), ``"gurobi"`` or ``"mosek"``.
    solver_options : dict, optional
        Options passed to the solver.
    criterion_tol : float, default 1e-4
        Relative tolerance on the criterion when computing bounds.
    initgrid_nx, initgrid_nu, audit_nx, audit_nu : int
        Sizes of the initial constraint grid and the audit grid.
    initgrid_x, initgrid_u, audit_x, audit_u : optional
        Explicit grids overriding the sampled ones.
    audit_add : int, default 100
        Maximum number of violated grid points added per audit round.
    audit_max : int, default 25
        Maximum number of audit rounds.
    audit_tol : float, default 1e-6
        Tolerance when checking shape restrictions on the audit grid.
    point : bool, optional
        Force point identification (``True``) or bounds (``False``); detected
        automatically when omitted.
    point_eyeweight : bool, default False
        Use the identity weighting matrix in GMM.
    seed : int or numpy.random.Generator, optional
        Randomness for sampling the covariate grids.
    noisy : bool, default False
        Print progress messages.

    Returns
    -------
    IVMTEResult
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    log: list[str] = []
    options = {k: v for k, v in locals().items() if k not in ("data", "rng", "log")}

    # -- arguments -----------------------------------------------------------
    if (ivlike is None) == (outcome is None):
        raise ValueError(
            "Specify exactly one of 'ivlike' (moment approach) or 'outcome' (regression approach)"
        )
    ivlike_list = [ivlike] if isinstance(ivlike, str) else list(ivlike or [])
    if isinstance(subset, str):
        subset = [subset]
    custom = target_weight0 is not None or target_weight1 is not None
    if custom and target is not None:
        raise ValueError("Specify either 'target' or custom target weights, not both")
    if not custom and target is None:
        raise ValueError("A 'target' parameter is required")
    if custom and (target_weight0 is None or target_weight1 is None):
        raise ValueError("Custom targets need both 'target_weight0' and 'target_weight1'")
    if target is not None and target.lower() not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    restrictions = {
        "m0_lb": m0_lb, "m0_ub": m0_ub, "m1_lb": m1_lb, "m1_ub": m1_ub,
        "mte_lb": mte_lb, "mte_ub": mte_ub, "m0_inc": m0_inc, "m0_dec": m0_dec,
        "m1_inc": m1_inc, "m1_dec": m1_dec, "mte_inc": mte_inc, "mte_dec": mte_dec,
    }  # fmt: skip
    shape_given = any(v for v in restrictions.values())

    # -- data ----------------------------------------------------------------
    is_formula = "~" in propensity
    if not is_formula and treat is None:
        raise ValueError("'treat' is required when 'propensity' names a column of scores")
    formulas = [m0, m1, *ivlike_list] + ([equal_coef] if equal_coef else [])
    extra = [c for c in (outcome, treat) if c]
    if is_formula:
        formulas.append(propensity)
    else:
        extra.append(propensity)
    cols = _required_columns(data, formulas, extra)
    complete = data[cols].notna().all(axis=1)
    if not complete.all():
        warnings.warn(
            f"{int((~complete).sum())} rows with missing values were dropped", stacklevel=2
        )
        data = data.loc[complete]
    data = data.reset_index(drop=True)
    for c in cols:
        if data[c].dtype == bool:
            data[c] = data[c].astype(int)
    n = len(data)

    # -- propensity, MTRs, target -------------------------------------------
    if is_formula:
        prop = fit_propensity(data, propensity, link)
    else:
        assert treat is not None
        prop = propensity_from_column(data, propensity, treat)
    spec0 = MTRSpec.from_formula(m0, data, uname)
    spec1 = MTRSpec.from_formula(m1, data, uname)
    if custom:
        assert target_weight0 is not None and target_weight1 is not None
        tg: TargetGammas = custom_target_gammas(
            spec0, spec1, data,
            target_weight0=target_weight0, target_weight1=target_weight1,
            target_knots0=target_knots0, target_knots1=target_knots1,
        )  # fmt: skip
        target_name = "custom"
    else:
        assert target is not None
        target_name = target.lower()
        tg = target_gammas_from_weights(
            spec0, spec1, data,
            conventional_weights(
                target_name, data, prop, late_from=late_from, late_to=late_to,
                late_x=late_x, genlate_lb=genlate_lb, genlate_ub=genlate_ub,
            ),
        )  # fmt: skip
    gstar = np.concatenate([tg.gstar0, tg.gstar1])
    equal = None
    if equal_coef is not None:
        eq_spec = MTRSpec.from_formula(equal_coef, data, uname)
        equal = equal_coef_matrix(spec0, spec1, eq_spec.names)
    names = tuple(f"[m0]{v}" for v in spec0.names) + tuple(f"[m1]{v}" for v in spec1.names)
    gstar_series = pd.Series(gstar, index=list(names))
    outcome_var = outcome if outcome is not None else ivlike_list[0].split("~")[0].strip()
    y_all = np.asarray(data[outcome_var], dtype=float)
    for key, default in (
        ("m0_lb", y_all.min()),
        ("m1_lb", y_all.min()),
        ("m0_ub", y_all.max()),
        ("m1_ub", y_all.max()),
    ):
        if restrictions[key] is None:
            restrictions[key] = float(default)

    def result(**kw: Any) -> IVMTEResult:
        return IVMTEResult(
            target=target_name, gstar=gstar_series, propensity=prop, specs=(spec0, spec1),
            target_gammas=tg, messages=log, options=options, **kw,
        )  # fmt: skip

    # -- moment approach -------------------------------------------------------
    moments: MomentSet | None = None
    crit: Criterion
    if outcome is None:
        log.append("Generating IV-like moments...")
        moments = build_moments(
            data, ivlike_list, spec0, spec1, prop, components=components, subsets=subset
        )
        n_coef = spec0.n_coef + spec1.n_coef
        if point is None:
            point = moments.n_independent >= n_coef
            if point:
                msg = "MTR is point identified via GMM."
                if shape_given or equal is not None:
                    msg += " Shape constraints are ignored."
                warnings.warn(msg, stacklevel=2)
        elif point and shape_given:
            warnings.warn(
                "If 'point' is True, shape restrictions on m0 and m1 are ignored and the "
                "audit procedure is not implemented.",
                stacklevel=2,
            )
        if point:
            log.append("Point estimate via GMM")
            fit = gmm(moments, n, identity_weight=point_eyeweight)
            est = float(gstar @ fit.theta)
            log.append(f"Point estimate of the target parameter: {est:.7g}")
            _emit(log, noisy)
            j_test = (
                {"stat": fit.j_stat, "df": fit.j_df, "p_value": fit.j_pvalue}
                if fit.j_stat is not None
                else None
            )
            return result(
                bounds=None, point_estimate=est, mtr_coef=pd.Series(fit.theta, index=list(names)),
                gstar_coef=None, moments=moments.n_independent, ivlike=moments, audit=None,
                criterion=None, j_test=j_test, solver="none", method="gmm",
            )  # fmt: skip
        crit = L1Criterion(np.hstack([moments.gamma0, moments.gamma1]), moments.beta)
        method = "lp"
    # -- regression approach ---------------------------------------------------
    else:
        x_reg = _regression_design(data, spec0, spec1, prop)
        full_rank = np.linalg.matrix_rank(x_reg) == x_reg.shape[1]
        if point is None:
            point = full_rank
        if point and not full_rank:
            raise ValueError("The MTR coefficients are not point identified by the regression")
        if point:
            if shape_given:
                warnings.warn(
                    "MTR is point identified via linear regression. Shape constraints are ignored.",
                    stacklevel=2,
                )
            else:
                warnings.warn("MTR is point identified via linear regression.", stacklevel=2)
            theta = least_squares(x_reg, y_all, equal)
            est = float(gstar @ theta)
            log.append(f"Point estimate of the target parameter: {est:.7g}")
            _emit(log, noisy)
            return result(
                bounds=None, point_estimate=est, mtr_coef=pd.Series(theta, index=list(names)),
                gstar_coef=None, moments=None, ivlike=None, audit=None, criterion=None,
                j_test=None, solver="none", method="ols",
            )  # fmt: skip
        crit = LSCriterion.from_regression(x_reg, y_all)
        method = "qcqp"

    # -- partial identification ------------------------------------------------
    solver_name = solver or default_solver(qcqp=method == "qcqp")
    xvars = sorted(set(spec0.covariates) | set(spec1.covariates))
    log.append("Performing audit procedure...")
    log.append(f"    Solver: {solver_name}")
    audit = _audit_with_expansion(
        spec0, spec1, crit, gstar, data, xvars, restrictions, equal,
        initgrid_nx=initgrid_nx, initgrid_nu=initgrid_nu, audit_nx=audit_nx, audit_nu=audit_nu,
        initgrid_x=initgrid_x, initgrid_u=initgrid_u, audit_x=audit_x, audit_u=audit_u, rng=rng,
        criterion_tol=criterion_tol, audit_tol=audit_tol, audit_add=audit_add,
        audit_max=audit_max, solver=solver_name, solver_options=solver_options, log=log,
    )  # fmt: skip
    _emit(log, noisy)
    gstar_coef = pd.DataFrame(
        {"min": audit.theta_min, "max": audit.theta_max, "criterion": audit.theta_crit},
        index=list(names),
    )
    return result(
        bounds=(audit.lower, audit.upper), point_estimate=None, mtr_coef=None,
        gstar_coef=gstar_coef, moments=moments.n_independent if moments else None,
        ivlike=moments, audit=audit, criterion=audit.criterion, j_test=None,
        solver=solver_name, method=method,
    )  # fmt: skip


def _emit(log: list[str], noisy: bool) -> None:
    if noisy:
        print("\n".join(log))


def _audit_with_expansion(
    spec0: MTRSpec,
    spec1: MTRSpec,
    crit: Criterion,
    gstar: np.ndarray,
    data: pd.DataFrame,
    xvars: list[str],
    restrictions: dict[str, Any],
    equal: np.ndarray | None,
    *,
    initgrid_nx: int,
    initgrid_nu: int,
    audit_nx: int,
    audit_nu: int,
    initgrid_x: pd.DataFrame | None,
    initgrid_u: Sequence[float] | None,
    audit_x: pd.DataFrame | None,
    audit_u: Sequence[float] | None,
    rng: np.random.Generator,
    criterion_tol: float,
    audit_tol: float,
    audit_add: int,
    audit_max: int,
    solver: str,
    solver_options: dict[str, Any] | None,
    log: list[str],
) -> AuditResult:
    """Run the audit, enlarging the initial grid when a bound problem is unbounded.

    The R package retries up to three times with an initial grid 1.5 times
    larger when a bound problem reports an unbounded or suboptimal status.
    """
    grids: Grids | None = None
    for attempt in range(4):
        grids = build_grids(
            data, xvars,
            initgrid_nx=initgrid_nx, initgrid_nu=initgrid_nu, audit_nx=audit_nx, audit_nu=audit_nu,
            initgrid_x=initgrid_x, initgrid_u=initgrid_u, audit_x=audit_x, audit_u=audit_u, rng=rng,
        )  # fmt: skip
        try:
            return run_audit(
                spec0, spec1, crit, gstar, grids, restrictions, equal=equal,
                criterion_tol=criterion_tol, audit_tol=audit_tol, audit_add=audit_add,
                audit_max=audit_max, solver=solver, solver_options=solver_options, log=log,
            )  # fmt: skip
        except AuditError as err:
            if err.status not in (3, 4, 6) or attempt == 3:
                raise
            initgrid_nx = min(int(np.ceil(initgrid_nx * 1.5)), audit_nx)
            initgrid_nu = min(int(np.ceil(initgrid_nu * 1.5)), audit_nu)
            log.append("    Restarting audit with new settings:")
            log.append(f"    initgrid_nx = {initgrid_nx}")
            log.append(f"    initgrid_nu = {initgrid_nu}")
    raise AssertionError("unreachable")
