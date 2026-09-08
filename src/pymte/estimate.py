"""The :func:`pymte.ivmte` estimator."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from formulaic import Formula
from numpy.typing import NDArray

from pymte.audit import AuditError, AuditResult, audit
from pymte.bootstrap import (
    BootstrapRetry,
    Replicate,
    bound_ci,
    bound_pvalue,
    coef_ci,
    point_ci,
    point_pvalues,
    resample,
)
from pymte.ivlike import MomentSet, build_moments
from pymte.lp import (
    Criterion,
    L1Criterion,
    default_solver,
    lp_setup_criterion_boot,
    lp_setup_equal_coef,
    qp_setup,
    run_lp,
)
from pymte.monobound import Grids
from pymte.mtr import MTRSpec, gen_gamma, polyparse
from pymte.point import GMMResult, gmm, least_squares
from pymte.propensity import Propensity, fit_propensity, propensity_from_column
from pymte.results import IVMTEResult
from pymte.splines import USpline
from pymte.weights import (
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
) -> NDArray[np.float64]:
    """Design matrix of the regression approach (inverse propensity weighted integrals)."""
    d = np.asarray(data[prop.treat], dtype=float)
    p = prop.phat
    with np.errstate(divide="ignore", invalid="ignore"):
        w0 = np.where(d == 0, 1.0 / (1.0 - p), 0.0)
        w1 = np.where(d == 1, 1.0 / p, 0.0)
    g0 = gen_gamma(spec0, data, p, 1.0, w0, means=False)
    g1 = gen_gamma(spec1, data, 0.0, p, w1, means=False)
    return np.hstack([g0, g1])


@dataclass
class _Model:
    """Everything computed from one sample before choosing an estimator."""

    data: pd.DataFrame
    prop: Propensity
    spec0: MTRSpec
    spec1: MTRSpec
    target: TargetGammas
    gstar: NDArray[np.float64]
    names: tuple[str, ...]
    equal: NDArray[np.float64] | None
    y: NDArray[np.float64]
    moments: MomentSet | None = None
    x_reg: NDArray[np.float64] | None = None

    @property
    def n(self) -> int:
        return len(self.data)

    def restrictions(self, o: SimpleNamespace) -> dict[str, Any]:
        """Shape restrictions with the R defaults (MTR bounds from the outcome range)."""
        r = {k: getattr(o, k) for k in _SHAPE}
        for key in ("m0_lb", "m1_lb"):
            r[key] = float(np.min(self.y)) if r[key] is None else r[key]
        for key in ("m0_ub", "m1_ub"):
            r[key] = float(np.max(self.y)) if r[key] is None else r[key]
        return r


_SHAPE = (
    "m0_lb", "m0_ub", "m1_lb", "m1_ub", "mte_lb", "mte_ub",
    "m0_inc", "m0_dec", "m1_inc", "m1_dec", "mte_inc", "mte_dec",
)  # fmt: skip


def _spec(
    m: str | Sequence[tuple[int | USpline, str | None]], data: pd.DataFrame, uname: str
) -> MTRSpec:
    if isinstance(m, str):
        return polyparse(m, data, uname)
    return MTRSpec.from_columns(m, data, uname)


def _prepare(data: pd.DataFrame, o: SimpleNamespace) -> _Model:
    """Fit the propensity score, parse the MTRs and build the target and IV-like moments."""
    if "~" in o.propensity:
        prop = fit_propensity(data, o.propensity, o.link)
    else:
        prop = propensity_from_column(data, o.propensity, o.treat)
    spec0 = _spec(o.m0, data, o.uname)
    spec1 = _spec(o.m1, data, o.uname)
    if o.target is None:
        tg = custom_target_gammas(
            spec0, spec1, data,
            target_weight0=o.target_weight0, target_weight1=o.target_weight1,
            target_knots0=o.target_knots0, target_knots1=o.target_knots1,
        )  # fmt: skip
    else:
        tg = target_gammas_from_weights(
            spec0, spec1, data,
            conventional_weights(
                o.target, data, prop, late_from=o.late_from, late_to=o.late_to,
                late_x=o.late_x, genlate_lb=o.genlate_lb, genlate_ub=o.genlate_ub,
            ),
        )  # fmt: skip
    equal = None
    if o.equal_coef is not None:
        eq_spec = polyparse(o.equal_coef, data, o.uname)
        equal = lp_setup_equal_coef(spec0, spec1, eq_spec.names)
    names = tuple(f"[m0]{v}" for v in spec0.names) + tuple(f"[m1]{v}" for v in spec1.names)
    outcome = o.outcome if o.outcome is not None else o.ivlike[0].split("~")[0].strip()
    model = _Model(
        data=data, prop=prop, spec0=spec0, spec1=spec1, target=tg,
        gstar=np.concatenate([tg.gstar0, tg.gstar1]), names=names, equal=equal,
        y=np.asarray(data[outcome], dtype=float),
    )  # fmt: skip
    if o.outcome is None:
        model.moments = build_moments(
            data, o.ivlike, spec0, spec1, prop, components=o.components, subsets=o.subset
        )
    else:
        model.x_reg = _regression_design(data, spec0, spec1, prop)
    return model


def ivmte(
    data: pd.DataFrame,
    *,
    m0: str | Sequence[tuple[int | USpline, str | None]],
    m1: str | Sequence[tuple[int | USpline, str | None]],
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
    bootstraps: int = 0,
    bootstraps_m: int | None = None,
    bootstraps_replace: bool = True,
    levels: Sequence[float] = (0.99, 0.95, 0.90),
    ci_type: str = "backward",
    specification_test: bool = True,
    seed: int | np.random.Generator | None = None,
    noisy: bool = False,
) -> IVMTEResult:
    """Estimate or bound a treatment parameter with the MTE framework.

    Parameters
    ----------
    data : pandas.DataFrame
        Estimation sample. Rows with missing values in any variable used are
        dropped with a warning.
    m0, m1 : str or sequence of (u_part, column)
        MTR functions of the untreated and treated arms: one-sided formulas
        in the unobservable ``uname`` and covariates, or explicit term lists
        for :meth:`pymte.MTRSpec.from_columns`.
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
    bootstraps : int, default 0
        Number of bootstrap replicates for inference (0 for none).
    bootstraps_m : int, optional
        Observations per bootstrap draw; defaults to the sample size.
    bootstraps_replace : bool, default True
        Draw with replacement. ``False`` gives subsampling.
    levels : sequence of float
        Confidence levels.
    ci_type : {"backward", "forward"}
        Confidence region reported in the summary for bounds; both are
        computed.
    specification_test : bool, default True
        Run the bootstrap misspecification test in the partially identified
        moment approach (when the sample criterion is positive).
    seed : int or numpy.random.Generator, optional
        Randomness for sampling the covariate grids and the bootstrap.
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
    custom = target_weight0 is not None or target_weight1 is not None
    if custom and target is not None:
        raise ValueError("Specify either 'target' or custom target weights, not both")
    if not custom and target is None:
        raise ValueError("A 'target' parameter is required")
    if custom and (target_weight0 is None or target_weight1 is None):
        raise ValueError("Custom targets need both 'target_weight0' and 'target_weight1'")
    if target is not None and target.lower() not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if ci_type not in ("backward", "forward"):
        raise ValueError("ci_type must be 'backward' or 'forward'")
    if bootstraps == 1:
        raise ValueError("'bootstraps' must be 0 or at least 2")
    is_formula = "~" in propensity
    if not is_formula and treat is None:
        raise ValueError("'treat' is required when 'propensity' names a column of scores")
    o = SimpleNamespace(**options)
    o.ivlike = ivlike_list
    o.subset = [subset] if isinstance(subset, str) else subset
    o.target = target.lower() if target is not None else None
    shape_given = any(getattr(o, k) for k in _SHAPE)

    # -- data ----------------------------------------------------------------
    formulas = [f for f in (m0, m1) if isinstance(f, str)] + ivlike_list
    if equal_coef:
        formulas.append(equal_coef)
    extra_cols = [c for f in (m0, m1) if not isinstance(f, str) for _, c in f if c]
    extra = [c for c in (outcome, treat) if c] + extra_cols
    (formulas if is_formula else extra).append(propensity)
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

    model = _prepare(data, o)
    target_name = "custom" if custom else str(o.target)
    gstar_series = pd.Series(model.gstar, index=list(model.names))
    restrictions = model.restrictions(o)

    def result(**kw: Any) -> IVMTEResult:
        return IVMTEResult(
            target=target_name, gstar=gstar_series, propensity=model.prop,
            specs=(model.spec0, model.spec1), target_gammas=model.target, messages=log,
            options=options, levels=tuple(levels), ci_type=ci_type, **kw,
        )  # fmt: skip

    # -- point identification ---------------------------------------------------
    crit: Criterion
    fit: GMMResult | None = None
    if model.moments is not None:
        log.append("Generating IV-like moments...")
        n_coef = model.spec0.n_coef + model.spec1.n_coef
        if point is None:
            point = model.moments.n_independent >= n_coef
            if point:
                msg = "MTR is point identified via GMM."
                if shape_given or model.equal is not None:
                    msg += " Shape constraints are ignored."
                warnings.warn(msg, stacklevel=2)
        elif point and shape_given:
            warnings.warn(
                "If 'point' is True, shape restrictions on m0 and m1 are ignored and the "
                "audit procedure is not implemented.",
                stacklevel=2,
            )
        if point:
            fit = gmm(model.moments, model.n, identity_weight=point_eyeweight)
            theta, method = fit.theta, "gmm"
        else:
            crit = L1Criterion(
                np.hstack([model.moments.gamma0, model.moments.gamma1]), model.moments.beta
            )
            method = "lp"
    else:
        assert model.x_reg is not None
        full_rank = np.linalg.matrix_rank(model.x_reg) == model.x_reg.shape[1]
        if point is None:
            point = full_rank
        if point and not full_rank:
            raise ValueError("The MTR coefficients are not point identified by the regression")
        if point:
            msg = "MTR is point identified via linear regression."
            if shape_given:
                msg += " Shape constraints are ignored."
            warnings.warn(msg, stacklevel=2)
            theta, method = least_squares(model.x_reg, model.y, model.equal), "ols"
        else:
            crit = qp_setup(model.x_reg, model.y)
            method = "qcqp"

    if point:
        est = float(model.gstar @ theta)
        log.append(f"Point estimate of the target parameter: {est:.7g}")
        j_test = None
        if fit is not None and fit.j_stat is not None:
            j_test = {"stat": fit.j_stat, "df": fit.j_df, "p_value": fit.j_pvalue}
        res = result(
            bounds=None, point_estimate=est, mtr_coef=pd.Series(theta, index=list(model.names)),
            gstar_coef=None, moments=model.moments.n_independent if model.moments else None,
            ivlike=model.moments, audit=None, criterion=None, j_test=j_test, solver="none",
            method=method,
        )  # fmt: skip
        if bootstraps:
            _bootstrap_point(res, model, o, fit, rng)
        _emit(log, noisy)
        return res

    # -- partial identification ------------------------------------------------
    solver_name = solver or default_solver(qcqp=method == "qcqp")
    log.append("Performing audit procedure...")
    log.append(f"    Solver: {solver_name}")
    audit_res = _audit_with_expansion(model, crit, restrictions, o, rng, solver_name, log)
    gstar_coef = pd.DataFrame(
        {
            "min": audit_res.theta_min,
            "max": audit_res.theta_max,
            "criterion": audit_res.theta_crit,
        },
        index=list(model.names),
    )
    res = result(
        bounds=(audit_res.lower, audit_res.upper), point_estimate=None, mtr_coef=None,
        gstar_coef=gstar_coef, moments=model.moments.n_independent if model.moments else None,
        ivlike=model.moments, audit=audit_res, criterion=audit_res.criterion, j_test=None,
        solver=solver_name, method=method,
    )  # fmt: skip
    if bootstraps:
        _bootstrap_bounds(res, model, o, rng, solver_name)
    _emit(log, noisy)
    return res


def _emit(log: list[str], noisy: bool) -> None:
    if noisy:
        print("\n".join(log))


def _criterion(model: _Model) -> Criterion:
    if model.moments is not None:
        return L1Criterion(
            np.hstack([model.moments.gamma0, model.moments.gamma1]), model.moments.beta
        )
    assert model.x_reg is not None
    return qp_setup(model.x_reg, model.y)


def _audit_with_expansion(
    model: _Model,
    crit: Criterion,
    restrictions: dict[str, Any],
    o: SimpleNamespace,
    rng: np.random.Generator,
    solver: str,
    log: list[str],
) -> AuditResult:
    """Run the audit, enlarging the initial grid when a bound problem is unbounded.

    The R package retries up to three times with an initial grid 1.5 times
    larger when a bound problem reports an unbounded or suboptimal status.
    """
    nx, nu = o.initgrid_nx, o.initgrid_nu
    for attempt in range(4):
        try:
            return _run_audit(model, crit, None, restrictions, o, solver, log, nx, nu, rng)
        except AuditError as err:
            if err.status not in (3, 4, 6) or attempt == 3:
                raise
            nx = min(int(np.ceil(nx * 1.5)), o.audit_nx)
            nu = min(int(np.ceil(nu * 1.5)), o.audit_nu)
            log.append("    Restarting audit with new settings:")
            log.append(f"    initgrid_nx = {nx}")
            log.append(f"    initgrid_nu = {nu}")
    raise AssertionError("unreachable")


def _run_audit(
    model: _Model,
    crit: Criterion,
    grids: Grids | None,
    restrictions: dict[str, Any],
    o: SimpleNamespace,
    solver: str,
    log: list[str],
    nx: int | None = None,
    nu: int | None = None,
    rng: np.random.Generator | None = None,
) -> AuditResult:
    return audit(
        model.spec0, model.spec1, crit, model.gstar, model.data, restrictions, equal=model.equal,
        criterion_tol=o.criterion_tol, audit_tol=o.audit_tol, audit_add=o.audit_add,
        audit_max=o.audit_max, initgrid_nx=nx or o.initgrid_nx, initgrid_nu=nu or o.initgrid_nu,
        audit_nx=o.audit_nx, audit_nu=o.audit_nu, initgrid_x=o.initgrid_x,
        initgrid_u=o.initgrid_u, audit_x=o.audit_x, audit_u=o.audit_u, audit_grid=grids, rng=rng,
        solver=solver, solver_options_criterion=o.solver_options,
        solver_options_bounds=o.solver_options, log=log,
    )  # fmt: skip


def _bootstrap_bounds(
    res: IVMTEResult,
    model: _Model,
    o: SimpleNamespace,
    rng: np.random.Generator,
    solver: str,
) -> None:
    """Bootstrap the bounds with the audit grid held fixed, as the R package does."""
    assert res.bounds is not None and res.audit is not None
    grids = res.audit.grids
    n = model.n
    m = o.bootstraps_m or n
    spec_test = o.specification_test and model.moments is not None and res.audit.criterion > 0
    orig = _criterion(model)
    orig_crit = res.audit.criterion

    def replicate(idx: NDArray[np.intp]) -> Replicate:
        bm = _prepare(model.data.iloc[idx].reset_index(drop=True), o)
        if bm.names != model.names or (
            bm.prop.params is not None
            and model.prop.params is not None
            and len(bm.prop.params) != len(model.prop.params)
        ):
            raise BootstrapRetry("a factor level is missing from the resample")
        crit = _criterion(bm)
        rep = _run_audit(bm, crit, grids, bm.restrictions(o), o, solver, [])
        stat = None
        if spec_test:
            assert isinstance(crit, L1Criterion) and isinstance(orig, L1Criterion)
            c, boot_model = lp_setup_criterion_boot(
                crit, orig, orig_crit, o.criterion_tol, rep.constraints, bm.equal
            )
            sol = run_lp(c, boot_model, "min", solver, o.solver_options)
            if sol.obj is None:
                raise RuntimeError(
                    f"The specification test LP returned no solution ({sol.status_str})"
                )
            stat = float(sol.obj)
        return Replicate(bounds=(rep.lower, rep.upper), propensity=bm.prop.params, spec_stat=stat)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = resample(n, m, o.bootstraps_replace, replicate, o.bootstraps, rng)
    assert boot.bounds is not None
    levels = list(res.levels)
    res.bootstraps, res.bootstraps_failed = boot.n_draws, boot.n_failed
    res.bounds_bootstraps = boot.bounds
    res.bounds_se = np.std(boot.bounds, axis=0, ddof=1)
    res.bounds_ci = {
        k: bound_ci(res.bounds, boot.bounds, n, m, levels, k) for k in ("backward", "forward")
    }
    res.p_value = {
        k: bound_pvalue(res.bounds, boot.bounds, n, m, k) for k in ("backward", "forward")
    }
    if boot.spec_stats is not None:
        res.specification_p_value = float(np.mean(orig_crit <= boot.spec_stats))
    _propensity_inference(res, boot.propensity, levels)
    res.messages.append(f"Bootstraps: {boot.n_draws} ({boot.n_failed} failed draws)")


def _bootstrap_point(
    res: IVMTEResult,
    model: _Model,
    o: SimpleNamespace,
    fit: GMMResult | None,
    rng: np.random.Generator,
) -> None:
    """Bootstrap a point estimate; GMM replicates are recentred at the sample moments."""
    assert res.point_estimate is not None and res.mtr_coef is not None
    n = model.n
    m = o.bootstraps_m or n

    def replicate(idx: NDArray[np.intp]) -> Replicate:
        bm = _prepare(model.data.iloc[idx].reset_index(drop=True), o)
        if bm.names != model.names:
            raise BootstrapRetry("a factor level is missing from the resample")
        j_stat = None
        if fit is not None:
            assert bm.moments is not None
            if bm.moments.n_moments != len(fit.moments) + len(fit.redundant):
                raise BootstrapRetry("the number of moments changed in the resample")
            bfit = gmm(
                bm.moments,
                bm.n,
                identity_weight=o.point_eyeweight,
                center=fit.moments,
                redundant=fit.redundant,
            )
            theta, j_stat = bfit.theta, bfit.j_stat
        else:
            assert bm.x_reg is not None
            theta = least_squares(bm.x_reg, bm.y, bm.equal)
        return Replicate(
            point=float(bm.gstar @ theta), mtr=theta, propensity=bm.prop.params, j_stat=j_stat
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = resample(n, m, o.bootstraps_replace, replicate, o.bootstraps, rng)
    assert boot.points is not None and boot.mtr is not None
    levels = list(res.levels)
    res.bootstraps, res.bootstraps_failed = boot.n_draws, boot.n_failed
    res.point_estimate_bootstraps = boot.points
    res.point_estimate_se = float(np.std(boot.points, ddof=1))
    res.point_estimate_ci = point_ci(res.point_estimate, boot.points, levels)
    res.p_value = point_pvalues(res.point_estimate, boot.points)
    res.mtr_bootstraps = boot.mtr
    res.mtr_se = pd.Series(np.std(boot.mtr, axis=0, ddof=1), index=res.mtr_coef.index)
    res.mtr_ci = coef_ci(res.mtr_coef.to_numpy(), boot.mtr, list(res.mtr_coef.index), levels)
    if boot.j_stats is not None and res.j_test is not None:
        res.j_test_bootstraps = boot.j_stats
        res.j_test["bootstrap_p_value"] = float(np.mean(boot.j_stats >= res.j_test["stat"]))
    _propensity_inference(res, boot.propensity, levels)
    res.messages.append(f"Bootstraps: {boot.n_draws} ({boot.n_failed} failed draws)")


def _propensity_inference(
    res: IVMTEResult, draws: NDArray[np.float64] | None, levels: list[float]
) -> None:
    params = res.propensity.params
    if draws is None or params is None:
        return
    names = list(res.propensity.names)
    res.propensity_bootstraps = draws
    res.propensity_se = pd.Series(np.std(draws, axis=0, ddof=1), index=names)
    res.propensity_ci = coef_ci(params, draws, names, levels)
