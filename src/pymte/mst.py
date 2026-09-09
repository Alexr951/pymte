"""The :func:`ivmte` estimator, its single-sample routine and its result.

This module mirrors ``mst.R`` of the R package: :func:`ivmte` checks the
arguments, runs :func:`ivmte_estimate` on the sample and, when asked, on
bootstrap resamples, and assembles an :class:`IVMTEResult`.
:func:`ivmte_estimate` builds the target moments (:func:`gen_target`) and
the IV-like moments (:func:`gen_s_set`) and either estimates the MTR
coefficients by GMM (:func:`gmm_estimate`) or least squares when they are
point identified, or bounds the target with the audit procedure.
:func:`bound_ci` and :func:`bound_pvalue` form the bootstrap confidence
regions of partially identified targets.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2, norm

from pymte.audit import AuditError, AuditResult, audit, fmt_result
from pymte.callcheck import formula_vars, get_xz, required_columns
from pymte.ivlike import IVLikeFit, MomentSet, iv_estimate
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
from pymte.propensity import Propensity, propensity
from pymte.splines import USpline
from pymte.wweights import (
    TARGETS,
    Piece,
    TargetWeights,
    gen_weight,
    wate1,
    watt1,
    watu1,
    wgenlate1,
    wlate1,
)

# -- target and IV-like moments --------------------------------------------------


@dataclass(frozen=True)
class TargetGammas:
    """Integrated target weights: ``target = gstar0 @ theta0 + gstar1 @ theta1``.

    Attributes
    ----------
    gstar0, gstar1 : numpy.ndarray
        Integrals of the MTR bases against the target weights.
    weights : TargetWeights or None
        The interval weights of a conventional target (``None`` for custom
        weights).
    n : int
        Number of observations the target averages over.
    """

    gstar0: NDArray[np.float64]
    gstar1: NDArray[np.float64]
    weights: TargetWeights | None
    n: int


def gen_target(
    spec0: MTRSpec,
    spec1: MTRSpec,
    data: pd.DataFrame,
    prop: Propensity,
    target: str | None = None,
    *,
    late_from: dict[str, Any] | None = None,
    late_to: dict[str, Any] | None = None,
    late_x: dict[str, Any] | None = None,
    genlate_lb: float | None = None,
    genlate_ub: float | None = None,
    target_weight0: Sequence[Piece] | None = None,
    target_weight1: Sequence[Piece] | None = None,
    target_knots0: Sequence[Piece] = (),
    target_knots1: Sequence[Piece] = (),
) -> TargetGammas:
    """Generate the target MTR moments ``gstar0`` and ``gstar1``.

    Parameters
    ----------
    spec0, spec1 : MTRSpec
        MTR specifications of the two arms.
    data : pandas.DataFrame
        Estimation sample.
    prop : Propensity
        Propensity score for the sample.
    target : {"ate", "att", "atu", "late", "avglate", "genlate"}, optional
        Conventional target; omit for custom weights.
    late_from, late_to : dict, optional
        Instrument values defining the LATE, e.g. ``{"z": 1}`` and ``{"z": 3}``.
    late_x : dict, optional
        Covariate values to condition on, e.g. ``{"x": 2}``.
    genlate_lb, genlate_ub : float, optional
        Limits in ``u`` of the generalised LATE.
    target_weight0, target_weight1 : sequence, optional
        Custom weights on each piece of ``[0, 1]``, as numbers or callables
        of covariate columns (argument names must match column names). With
        ``K`` knots there must be ``K + 1`` weights. No sign convention is
        imposed: for a treatment effect the ``m0`` weights are typically
        negative.
    target_knots0, target_knots1 : sequence, optional
        Interior knots in ``u`` splitting [0, 1] into pieces, as numbers or
        callables of covariate columns.

    Returns
    -------
    TargetGammas
    """
    n = len(data)
    if target is None:
        if target_weight0 is None or target_weight1 is None:
            raise ValueError("Custom targets need both 'target_weight0' and 'target_weight1'")
        gammas = []
        for spec, weights, knots in (
            (spec0, target_weight0, target_knots0),
            (spec1, target_weight1, target_knots1),
        ):
            if len(weights) != len(knots) + 1:
                raise ValueError(
                    f"Expected {len(knots) + 1} weights for {len(knots)} knots, got {len(weights)}"
                )
            edges = [np.zeros(n)] + [gen_weight(data, k) for k in knots] + [np.ones(n)]
            total = np.zeros(spec.n_coef)
            for i, piece in enumerate(weights):
                total += gen_gamma(spec, data, edges[i], edges[i + 1], gen_weight(data, piece))
            gammas.append(total)
        return TargetGammas(gammas[0], gammas[1], None, n)
    target = target.lower()
    p = prop.phat
    d = np.asarray(data[prop.treat], dtype=float)
    if target == "ate":
        w = wate1(n)
    elif target == "att":
        w = watt1(p, d)
    elif target == "atu":
        w = watu1(p, d)
    elif target in ("late", "avglate"):
        if not late_from or not late_to:
            raise ValueError("Targets 'late' and 'avglate' need 'late_from' and 'late_to'")
        w = wlate1(prop, data, late_from, late_to, late_x, avglate=target == "avglate")
    elif target == "genlate":
        if genlate_lb is None or genlate_ub is None:
            raise ValueError("Target 'genlate' needs 'genlate_lb' and 'genlate_ub'")
        w = wgenlate1(data, genlate_lb, genlate_ub, late_x)
    else:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    g0 = -gen_gamma(spec0, data, w.lb, w.ub, w.mult, rows=w.rows)
    g1 = gen_gamma(spec1, data, w.lb, w.ub, w.mult, rows=w.rows)
    return TargetGammas(g0, g1, w, int(w.rows.sum()))


def gen_s_set(
    data: pd.DataFrame,
    formulas: Sequence[str],
    spec0: MTRSpec,
    spec1: MTRSpec,
    prop: Propensity,
    *,
    components: Sequence[Sequence[str] | None] | None = None,
    subsets: Sequence[str | None] | None = None,
) -> MomentSet:
    """Fit all IV-like regressions and integrate the MTR bases against their weights.

    Parameters
    ----------
    data : pandas.DataFrame
        Estimation sample.
    formulas : sequence of str
        IV-like regression formulas.
    spec0, spec1 : MTRSpec
        MTR specifications.
    prop : Propensity
        Propensity score for the sample.
    components : sequence, optional
        One entry per formula: the coefficients to use (``None`` for all).
    subsets : sequence, optional
        One entry per formula: a row-selection expression (``None`` for all).

    Returns
    -------
    MomentSet
    """
    k = len(formulas)
    comps = list(components) if components is not None else [None] * k
    subs = list(subsets) if subsets is not None else [None] * k
    if len(comps) != k or len(subs) != k:
        raise ValueError("'components' and 'subset' need one entry per IV-like formula")
    fits: list[IVLikeFit] = []
    names, betas, g0m, g1m, g0o, g1o, ys, rows = [], [], [], [], [], [], [], []
    p = prop.phat
    for i, (f, c, s) in enumerate(zip(formulas, comps, subs, strict=True)):
        fit = iv_estimate(f, data, prop.treat, components=c, subset=s)
        fits.append(fit)
        sub = data.loc[fit.rows]
        psub = p[fit.rows]
        for j, name in enumerate(fit.components):
            g0 = gen_gamma(spec0, sub, psub, 1.0, fit.s0[:, j], means=False)
            g1 = gen_gamma(spec1, sub, 0.0, psub, fit.s1[:, j], means=False)
            names.append(f"{i + 1}:{name}")
            betas.append(fit.beta[j])
            g0m.append(g0.mean(axis=0))
            g1m.append(g1.mean(axis=0))
            g0o.append(g0)
            g1o.append(g1)
            ys.append(fit.y * (fit.s1[:, j] * fit.d + fit.s0[:, j] * (1.0 - fit.d)))
            rows.append(fit.rows)
    n = len(data)
    stacked = np.zeros((2 * n, len(betas)))
    col = 0
    for fit in fits:
        for j in range(len(fit.components)):
            stacked[:n][fit.rows, col] = fit.s0[:, j]
            stacked[n:][fit.rows, col] = fit.s1[:, j]
            col += 1
    return MomentSet(
        fits=tuple(fits),
        names=tuple(names),
        beta=np.asarray(betas, dtype=float),
        gamma0=np.vstack(g0m),
        gamma1=np.vstack(g1m),
        gamma0_obs=tuple(g0o),
        gamma1_obs=tuple(g1o),
        ys=tuple(ys),
        rows=tuple(rows),
        n_independent=int(np.linalg.matrix_rank(stacked)),
    )


# -- point identification ---------------------------------------------------------


@dataclass(frozen=True)
class GMMResult:
    """Two-step GMM estimates of the MTR coefficients.

    Attributes
    ----------
    theta : numpy.ndarray
        Stacked coefficients ``(theta0, theta1)``.
    moments : numpy.ndarray
        Sample moment conditions ``beta - Gamma @ theta`` after dropping
        redundant moments.
    j_stat, j_df, j_pvalue : float or None
        Hansen J statistic, its degrees of freedom and asymptotic p-value
        (``None`` when the model is exactly identified).
    redundant : tuple of int
        Positions of moments dropped because of collinearity.
    """

    theta: NDArray[np.float64]
    moments: NDArray[np.float64]
    j_stat: float | None
    j_df: int | None
    j_pvalue: float | None
    redundant: tuple[int, ...]


def moment_matrix(mom: MomentSet, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-observation moments ``s_k Y`` and ``[g0, g1]`` zero-filled outside each subset.

    Following the R package, moments defined on a subset are averaged over
    the full sample size ``n``.
    """
    s = mom.n_moments
    j = mom.gamma0.shape[1] + mom.gamma1.shape[1]
    ys = np.zeros((n, s))
    gs = np.zeros((n, s, j))
    for k in range(s):
        rows = mom.rows[k]
        ys[rows, k] = mom.ys[k]
        gs[rows, k, :] = np.hstack([mom.gamma0_obs[k], mom.gamma1_obs[k]])
    return ys, gs


def gmm_estimate(
    mom: MomentSet,
    n: int,
    *,
    identity: bool = False,
    center: NDArray[np.float64] | None = None,
    redundant: tuple[int, ...] | None = None,
) -> GMMResult:
    """Estimate the MTR coefficients by two-step GMM.

    Parameters
    ----------
    mom : MomentSet
        IV-like moment conditions.
    n : int
        Sample size.
    identity : bool, default False
        Use the identity weighting matrix (one-step GMM). The default is the
        efficient two-step estimator.
    center : numpy.ndarray, optional
        Moment recentering used in the bootstrap (the original sample's
        moment residuals).
    redundant : tuple of int, optional
        Moments to drop; detected automatically when ``None``.

    Returns
    -------
    GMMResult
    """
    ys, gs = moment_matrix(mom, n)
    s = mom.n_moments
    xmat = gs.mean(axis=0)
    ymat = ys.mean(axis=0)
    j = xmat.shape[1]
    if s < j:
        raise ValueError(
            f"GMM system is underidentified: {j} MTR coefficients but only {s} moment conditions"
        )
    if np.linalg.matrix_rank(xmat) < j:
        raise ValueError(
            f"GMM system is underidentified: the {s} moment conditions identify fewer than "
            f"{j} MTR coefficients. Adjust the IV-like specifications or m0 and m1."
        )
    if redundant is None:
        # Drop moments whose residuals are collinear, as the R package does.
        rng = np.random.default_rng(0)
        resid = ys - gs @ rng.normal(size=j)
        omega = resid.T @ resid / n
        drop: list[int] = []
        keep = list(range(s))
        while True:
            w, v = np.linalg.eigh(omega[np.ix_(keep, keep)])
            small = np.flatnonzero(np.abs(w) < 1e-8)
            if not small.size:
                break
            loading = np.abs(v[:, small[0]])
            worst = max(k for k in range(len(keep)) if loading[k] > 1e-8)
            drop.append(keep.pop(worst))
        redundant = tuple(sorted(drop))
    keep_idx = [k for k in range(s) if k not in redundant]
    xk, yk = xmat[keep_idx], ymat[keep_idx]
    cen = np.zeros(len(keep_idx)) if center is None else center
    theta = np.linalg.lstsq(xk, yk - cen, rcond=None)[0]
    omega_inv = None
    if not identity:
        resid = ys[:, keep_idx] - gs[:, keep_idx, :] @ theta
        omega = resid.T @ resid / n
        omega_inv = np.linalg.pinv(omega)
        a = xk.T @ omega_inv @ xk
        theta = np.linalg.solve(a, xk.T @ omega_inv @ (yk - cen))
    moments = yk - xk @ theta
    j_stat = j_df = j_p = None
    if len(keep_idx) > j:
        m = moments - cen
        w_mat = omega_inv if omega_inv is not None else np.eye(len(keep_idx))
        j_stat = float(n * m @ w_mat @ m)
        j_df = len(keep_idx) - j
        j_p = float(chi2.sf(j_stat, j_df))
    return GMMResult(theta, moments, j_stat, j_df, j_p, tuple(redundant))


def _least_squares(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    equal: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Ordinary or equality-constrained (``equal @ theta == 0``) least squares."""
    xtx, xty = x.T @ x, x.T @ y
    if equal is None or len(equal) == 0:
        return np.asarray(np.linalg.solve(xtx, xty), dtype=float)
    # Solve the KKT system of the constrained problem.
    e = len(equal)
    kkt = np.block([[xtx, equal.T], [equal, np.zeros((e, e))]])
    rhs = np.concatenate([xty, np.zeros(e)])
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return np.asarray(sol[: x.shape[1]], dtype=float)


# -- bootstrap inference -----------------------------------------------------------


class BootstrapRetry(Exception):
    """A resample is unusable (for example a factor level disappeared); draw again."""


@dataclass
class Replicate:
    """What one bootstrap replicate produces."""

    bounds: tuple[float, float] | None = None
    point: float | None = None
    mtr: NDArray[np.float64] | None = None
    propensity: NDArray[np.float64] | None = None
    spec_stat: float | None = None
    j_stat: float | None = None


@dataclass
class BootstrapResult:
    """Collected replicates."""

    n_draws: int
    n_failed: int
    bounds: NDArray[np.float64] | None = None
    points: NDArray[np.float64] | None = None
    mtr: NDArray[np.float64] | None = None
    propensity: NDArray[np.float64] | None = None
    spec_stats: NDArray[np.float64] | None = None
    j_stats: NDArray[np.float64] | None = None


def _resample(
    n: int,
    m: int,
    replace: bool,
    replicate: Callable[[NDArray[np.intp]], Replicate],
    bootstraps: int,
    rng: np.random.Generator,
) -> BootstrapResult:
    """Draw ``bootstraps`` usable replicates, redrawing after failures."""
    if not replace and m > n:
        raise ValueError(
            "'bootstraps_m' cannot exceed the sample size when 'bootstraps_replace' is False"
        )
    max_failures = 10 * bootstraps
    reps: list[Replicate] = []
    failed = 0
    while len(reps) < bootstraps:
        idx = rng.choice(n, size=m, replace=replace)
        try:
            reps.append(replicate(np.asarray(idx, dtype=np.intp)))
        except Exception as err:  # noqa: BLE001 - any failure means redraw, as in R
            failed += 1
            if failed > max_failures:
                raise RuntimeError(
                    f"The bootstrap failed on {failed} draws; last error: {err}"
                ) from err
    if failed:
        warnings.warn(f"{failed} bootstrap draws failed and were redrawn", stacklevel=2)

    def stack(attr: str) -> NDArray[np.float64] | None:
        vals = [getattr(r, attr) for r in reps]
        if any(v is None for v in vals):
            return None
        return np.asarray(np.vstack([np.atleast_1d(v) for v in vals]), dtype=float)

    def column(attr: str) -> NDArray[np.float64] | None:
        v = stack(attr)
        return v[:, 0] if v is not None else None

    return BootstrapResult(
        n_draws=len(reps),
        n_failed=failed,
        bounds=stack("bounds"),
        points=column("point"),
        mtr=stack("mtr"),
        propensity=stack("propensity"),
        spec_stats=column("spec_stat"),
        j_stats=column("j_stat"),
    )


def _q(x: NDArray[np.float64], prob: float) -> float:
    """Type-1 (inverse ECDF) quantile, matching R's ``quantile(type = 1)``."""
    return float(np.quantile(x, prob, method="inverted_cdf"))


def bound_ci(
    bounds: tuple[float, float],
    resamples: NDArray[np.float64],
    n: int,
    m: int,
    levels: Sequence[float],
    kind: str,
) -> pd.DataFrame:
    """Backward or forward confidence region for partially identified bounds.

    Parameters
    ----------
    bounds : tuple of float
        Sample bounds ``(lb, ub)``.
    resamples : numpy.ndarray
        Bootstrap bounds, shape ``(B, 2)``.
    n, m : int
        Sample size and observations per bootstrap draw.
    levels : sequence of float
        Confidence levels.
    kind : {"backward", "forward"}
        Type of region.

    Returns
    -------
    pandas.DataFrame
        One row per level with columns ``lower`` and ``upper``.
    """
    lb, ub = bounds
    lo = np.sqrt(m) * (resamples[:, 0] - lb)
    hi = np.sqrt(m) * (resamples[:, 1] - ub)
    rows = []
    for a in levels:
        if kind == "backward":
            rows.append(
                (lb + _q(lo, 0.5 * (1 - a)) / np.sqrt(n), ub + _q(hi, 0.5 * (1 + a)) / np.sqrt(n))
            )
        elif kind == "forward":
            rows.append(
                (lb - _q(lo, 0.5 * (1 + a)) / np.sqrt(n), ub - _q(hi, 0.5 * (1 - a)) / np.sqrt(n))
            )
        else:
            raise ValueError("kind must be 'backward' or 'forward'")
    return pd.DataFrame(rows, index=list(levels), columns=["lower", "upper"])


def bound_pvalue(
    bounds: tuple[float, float], resamples: NDArray[np.float64], n: int, m: int, kind: str
) -> float:
    """p-value for the null that the target is zero, by inverting the confidence region."""
    b = len(resamples)
    levels = np.arange(1, b + 1) / b
    ci = bound_ci(bounds, resamples, n, m, levels.tolist(), kind)
    excludes = ~((ci["lower"] <= 0) & (ci["upper"] >= 0)).to_numpy()
    if excludes.all():
        return 0.0
    if not excludes.any():
        return 1.0
    return float(1 - levels[excludes].max())


def _point_ci(
    estimate: float, resamples: NDArray[np.float64], levels: Sequence[float]
) -> dict[str, pd.DataFrame]:
    """Percentile (``nonparametric``) and normal-approximation (``normal``) intervals."""
    sd = float(np.std(resamples, ddof=1))
    nonpar = [(_q(resamples, (1 - a) / 2), _q(resamples, 1 - (1 - a) / 2)) for a in levels]
    normal = [
        (estimate + norm.ppf((1 - a) / 2) * sd, estimate + norm.ppf(1 - (1 - a) / 2) * sd)
        for a in levels
    ]
    cols = ["lower", "upper"]
    return {
        "nonparametric": pd.DataFrame(nonpar, index=list(levels), columns=cols),
        "normal": pd.DataFrame(normal, index=list(levels), columns=cols),
    }


def _point_pvalues(estimate: float, resamples: NDArray[np.float64]) -> dict[str, float]:
    """Two-sided p-values for a zero target from the bootstrap distribution."""
    dev = resamples - estimate
    nonpar = float(((dev >= abs(estimate)).sum() + (dev <= -abs(estimate)).sum()) / len(resamples))
    sd = float(np.std(resamples, ddof=1))
    parametric = float(2 * norm.cdf(-abs(estimate - resamples.mean()) / sd)) if sd > 0 else 0.0
    return {"nonparametric": nonpar, "parametric": parametric}


def _coef_ci(
    estimates: NDArray[np.float64],
    resamples: NDArray[np.float64],
    names: Sequence[str],
    levels: Sequence[float],
) -> dict[str, pd.DataFrame]:
    """Per-coefficient percentile and normal intervals, one column pair per level."""
    sd = np.std(resamples, axis=0, ddof=1)
    out = {}
    for kind in ("nonparametric", "normal"):
        cols = {}
        for a in levels:
            if kind == "nonparametric":
                cols[f"{a} lower"] = np.quantile(
                    resamples, (1 - a) / 2, axis=0, method="inverted_cdf"
                )
                cols[f"{a} upper"] = np.quantile(
                    resamples, 1 - (1 - a) / 2, axis=0, method="inverted_cdf"
                )
            else:
                cols[f"{a} lower"] = estimates + norm.ppf((1 - a) / 2) * sd
                cols[f"{a} upper"] = estimates + norm.ppf(1 - (1 - a) / 2) * sd
        out[kind] = pd.DataFrame(cols, index=list(names))
    return out


# -- the result -----------------------------------------------------------------------


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame | pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _ci_lines(ci: pd.DataFrame) -> list[str]:
    return [
        f"    {float(level):.0%}: [{fmt_result(row['lower'])}, {fmt_result(row['upper'])}]"
        for level, row in zip(ci.index.to_numpy(dtype=float), ci.to_dict("records"), strict=True)
    ]


@dataclass
class IVMTEResult:
    """Estimates, bounds and diagnostics from :func:`pymte.ivmte`.

    Exactly one of ``bounds`` and ``point_estimate`` is set. Attribute names
    follow the R package with dots replaced by underscores. The inference
    attributes are filled when ``bootstraps > 0``.

    Attributes
    ----------
    target : str
        Name of the target parameter (``"custom"`` for user weights).
    bounds : tuple of float or None
        ``(lower, upper)`` in the partially identified case.
    point_estimate : float or None
        Estimate in the point identified case.
    mtr_coef : pandas.Series or None
        MTR coefficients (point identified case), indexed by ``[m0]``/``[m1]``
        prefixed names.
    gstar_coef : pandas.DataFrame or None
        MTR coefficients at the lower bound, upper bound and criterion
        minimum (columns ``min``, ``max``, ``criterion``).
    gstar : pandas.Series
        Integrated target weights; ``gstar @ theta`` is the target.
    moments : int or None
        Number of linearly independent IV-like moments.
    ivlike : MomentSet or None
        The IV-like moments (moment approach only).
    propensity : Propensity
        Fitted propensity score.
    audit : AuditResult or None
        Audit diagnostics (partially identified case), including the grids.
    criterion : float or None
        Minimum criterion (partially identified case).
    j_test : dict or None
        Hansen J statistic, degrees of freedom, asymptotic p-value and, after
        a bootstrap, ``bootstrap_p_value`` (GMM case).
    specs : tuple of MTRSpec
        The parsed ``m0`` and ``m1`` specifications.
    target_gammas : TargetGammas
        The target weights and their integrals.
    solver : str
        Solver used.
    method : str
        ``"lp"``, ``"qcqp"``, ``"gmm"`` or ``"ols"``.
    messages : list of str
        Progress log.
    options : dict
        The call's options.
    bootstraps, bootstraps_failed : int
        Number of bootstrap replicates used and of draws that failed.
    bounds_bootstraps : numpy.ndarray or None
        Bootstrap bounds, shape ``(B, 2)``.
    bounds_se : numpy.ndarray or None
        Standard errors of the two bounds.
    bounds_ci : dict or None
        ``"backward"`` and ``"forward"`` confidence regions, one row per
        level.
    p_value : dict or None
        p-values for a zero target: ``backward``/``forward`` (bounds) or
        ``nonparametric``/``parametric`` (point estimate).
    specification_p_value : float or None
        Bootstrap p-value of the misspecification test (partially identified
        moment approach with a positive criterion).
    point_estimate_bootstraps, point_estimate_se, point_estimate_ci
        Bootstrap draws, standard error and ``nonparametric``/``normal``
        intervals of the point estimate.
    mtr_bootstraps, mtr_se, mtr_ci
        The same for the MTR coefficients.
    propensity_bootstraps, propensity_se, propensity_ci
        The same for the propensity score coefficients.
    j_test_bootstraps : numpy.ndarray or None
        Bootstrap J statistics.
    levels : tuple of float
        Confidence levels.
    ci_type : str
        Region reported by :meth:`summary` for bounds.
    """

    target: str
    bounds: tuple[float, float] | None
    point_estimate: float | None
    mtr_coef: pd.Series | None
    gstar_coef: pd.DataFrame | None
    gstar: pd.Series
    moments: int | None
    ivlike: MomentSet | None
    propensity: Propensity
    audit: AuditResult | None
    criterion: float | None
    j_test: dict[str, Any] | None
    specs: tuple[MTRSpec, MTRSpec]
    target_gammas: TargetGammas
    solver: str
    method: str
    messages: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    bootstraps: int = 0
    bootstraps_failed: int = 0
    bounds_bootstraps: np.ndarray | None = None
    bounds_se: np.ndarray | None = None
    bounds_ci: dict[str, pd.DataFrame] | None = None
    p_value: dict[str, float] | None = None
    specification_p_value: float | None = None
    point_estimate_bootstraps: np.ndarray | None = None
    point_estimate_se: float | None = None
    point_estimate_ci: dict[str, pd.DataFrame] | None = None
    mtr_bootstraps: np.ndarray | None = None
    mtr_se: pd.Series | None = None
    mtr_ci: dict[str, pd.DataFrame] | None = None
    propensity_bootstraps: np.ndarray | None = None
    propensity_se: pd.Series | None = None
    propensity_ci: dict[str, pd.DataFrame] | None = None
    j_test_bootstraps: np.ndarray | None = None
    levels: tuple[float, ...] = (0.99, 0.95, 0.90)
    ci_type: str = "backward"

    def summary(self) -> str:
        """Return a text summary in the style of R's ``summary.ivmte``."""
        s0, s1 = self.specs
        lines = []
        if self.bounds is not None:
            lines.append(
                "Bounds on the target parameter: "
                f"[{fmt_result(self.bounds[0])}, {fmt_result(self.bounds[1])}]"
            )
            assert self.audit is not None
            if len(self.audit.violations):
                lines.append(f"Audit reached audit_max ({self.audit.audit_count})")
            else:
                lines.append(
                    f"Audit terminated successfully after {self.audit.audit_count} round(s)"
                )
        else:
            assert self.point_estimate is not None
            lines.append(
                f"Point estimate of the target parameter: {fmt_result(self.point_estimate)}"
            )
        lines.append(f"MTR coefficients: {s0.n_coef + s1.n_coef}")
        if self.ivlike is not None:
            lines.append(f"Independent/total moments: {self.moments}/{self.ivlike.n_moments}")
        if self.criterion is not None:
            lines.append(f"Minimum criterion: {fmt_result(self.criterion)}")
        lines.append(f"Solver: {self.solver}")
        if self.bootstraps:
            if self.bounds_ci is not None:
                lines.append(f"\nBootstrapped confidence intervals ({self.ci_type}):")
                lines += _ci_lines(self.bounds_ci[self.ci_type])
                if self.p_value is not None:
                    lines.append(f"p-value: {fmt_result(self.p_value[self.ci_type])}")
                if self.specification_p_value is not None:
                    lines.append(
                        "Bootstrapped specification test p-value: "
                        f"{fmt_result(self.specification_p_value)}"
                    )
            elif self.point_estimate_ci is not None:
                lines.append("\nBootstrapped confidence intervals (nonparametric):")
                lines += _ci_lines(self.point_estimate_ci["nonparametric"])
                if self.p_value is not None:
                    lines.append(f"p-value: {fmt_result(self.p_value['nonparametric'])}")
                if self.j_test is not None and "bootstrap_p_value" in self.j_test:
                    lines.append(
                        f"Bootstrapped J-test p-value: {fmt_result(self.j_test['bootstrap_p_value'])}"
                    )
            lines.append(f"Number of bootstraps: {self.bootstraps}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        """Return the numerical results as plain Python objects."""
        return {
            "target": self.target,
            "method": self.method,
            "solver": self.solver,
            "bounds": self.bounds,
            "point_estimate": self.point_estimate,
            "criterion": self.criterion,
            "moments": self.moments,
            "gstar": _to_plain(self.gstar),
            "mtr_coef": _to_plain(self.mtr_coef),
            "gstar_coef": _to_plain(self.gstar_coef),
            "j_test": self.j_test,
            "propensity_coef": (
                dict(zip(self.propensity.names, self.propensity.params.tolist(), strict=True))
                if self.propensity.params is not None
                else None
            ),
            "audit_count": self.audit.audit_count if self.audit else None,
            "bootstraps": self.bootstraps,
            "bounds_ci": _to_plain(self.bounds_ci),
            "point_estimate_ci": _to_plain(self.point_estimate_ci),
            "p_value": self.p_value,
            "specification_p_value": self.specification_p_value,
            "options": _to_plain(self.options),
        }


# -- one estimation pass ---------------------------------------------------------------

_SHAPE = (
    "m0_lb", "m0_ub", "m1_lb", "m1_ub", "mte_lb", "mte_ub",
    "m0_inc", "m0_dec", "m1_inc", "m1_dec", "mte_inc", "mte_dec",
)  # fmt: skip


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

    def criterion(self) -> Criterion:
        if self.moments is not None:
            return L1Criterion(
                np.hstack([self.moments.gamma0, self.moments.gamma1]), self.moments.beta
            )
        assert self.x_reg is not None
        return qp_setup(self.x_reg, self.y)


@dataclass
class _Estimate:
    """Output of one pass of :func:`ivmte_estimate` (bounds or a point estimate)."""

    model: _Model
    point: bool
    method: str
    solver: str
    theta: NDArray[np.float64] | None = None
    estimate: float | None = None
    fit: GMMResult | None = None
    audit: AuditResult | None = None
    spec_stat: float | None = None


def _spec(
    m: str | Sequence[tuple[int | USpline, str | None]], data: pd.DataFrame, uname: str
) -> MTRSpec:
    if isinstance(m, str):
        return polyparse(m, data, uname)
    return MTRSpec.from_columns(m, data, uname)


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


def _prepare(data: pd.DataFrame, o: SimpleNamespace) -> _Model:
    """Fit the propensity score, parse the MTRs and build the target and IV-like moments."""
    prop = propensity(o.propensity, data, o.link, o.treat)
    spec0 = _spec(o.m0, data, o.uname)
    spec1 = _spec(o.m1, data, o.uname)
    tg = gen_target(
        spec0, spec1, data, prop, o.target, late_from=o.late_from, late_to=o.late_to,
        late_x=o.late_x, genlate_lb=o.genlate_lb, genlate_ub=o.genlate_ub,
        target_weight0=o.target_weight0, target_weight1=o.target_weight1,
        target_knots0=o.target_knots0, target_knots1=o.target_knots1,
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
        model.moments = gen_s_set(
            data, o.ivlike, spec0, spec1, prop, components=o.components, subsets=o.subset
        )
    else:
        model.x_reg = _regression_design(data, spec0, spec1, prop)
    return model


def ivmte_estimate(
    data: pd.DataFrame,
    o: SimpleNamespace,
    rng: np.random.Generator,
    log: list[str],
    *,
    orig: _Estimate | None = None,
) -> _Estimate:
    """Run the estimation procedure once on a sample.

    This is the single iteration of the R package's ``ivmteEstimate``:
    propensity score, target and IV-like moments, then either the point
    estimate or the audit procedure. With ``orig`` (a bootstrap replicate)
    the choices of the original sample are reused: point identification
    status, the audit grid, the GMM moment centring and the dropped
    moments; the misspecification test statistic is computed when the
    original criterion is positive.
    """
    model = _prepare(data, o)
    boot = orig is not None
    if boot:
        assert orig is not None
        if model.names != orig.model.names or (
            model.prop.params is not None
            and orig.model.prop.params is not None
            and len(model.prop.params) != len(orig.model.prop.params)
        ):
            raise BootstrapRetry("a factor level is missing from the resample")
    shape_given = any(getattr(o, k) for k in _SHAPE)
    point = orig.point if boot and orig is not None else o.point
    fit: GMMResult | None = None
    if model.moments is not None:
        method = "lp"
        if not boot:
            log.append("Generating IV-like moments...")
            lacking = [
                str(i + 1)
                for i, f in enumerate(o.ivlike)
                if o.treat not in formula_vars(get_xz(f)[1])
            ]
            if lacking and model.moments.n_independent < model.moments.n_moments:
                warnings.warn(
                    "The following IV-like specifications do not include the treatment "
                    f"variable: {', '.join(lacking)}. This may result in fewer independent "
                    "moment conditions than expected.",
                    stacklevel=3,
                )
            n_coef = model.spec0.n_coef + model.spec1.n_coef
            if point is None:
                point = model.moments.n_independent >= n_coef
                if point:
                    msg = "MTR is point identified via GMM."
                    if shape_given or model.equal is not None:
                        msg += " Shape constraints are ignored."
                    warnings.warn(msg, stacklevel=3)
            elif point and shape_given:
                warnings.warn(
                    "If 'point' is True, shape restrictions on m0 and m1 are ignored and the "
                    "audit procedure is not implemented.",
                    stacklevel=3,
                )
        if point:
            method = "gmm"
            center = redundant = None
            if boot and orig is not None and orig.fit is not None:
                if model.moments.n_moments != len(orig.fit.moments) + len(orig.fit.redundant):
                    raise BootstrapRetry("the number of moments changed in the resample")
                center, redundant = orig.fit.moments, orig.fit.redundant
            fit = gmm_estimate(
                model.moments, model.n, identity=o.point_eyeweight, center=center,
                redundant=redundant,
            )  # fmt: skip
            theta = fit.theta
    else:
        assert model.x_reg is not None
        method = "qcqp"
        if not boot:
            full_rank = np.linalg.matrix_rank(model.x_reg) == model.x_reg.shape[1]
            # As in R, a collinear design falls through to the bounds even
            # when point identification was requested.
            point = full_rank if point is None else point and full_rank
            if point:
                msg = "MTR is point identified via linear regression."
                if shape_given:
                    msg += " Shape constraints are ignored."
                warnings.warn(msg, stacklevel=3)
        if point:
            method = "ols"
            theta = _least_squares(model.x_reg, model.y, model.equal)
    if point:
        est = _Estimate(model, True, method, "none", theta=theta, fit=fit)
        est.estimate = float(model.gstar @ theta)
        if not boot:
            log.append(f"Point estimate of the target parameter: {est.estimate:.7g}")
        return est

    # -- partial identification -----------------------------------------------------
    solver = o.solver or default_solver(qcqp=method == "qcqp")
    crit = model.criterion()
    restrictions = model.restrictions(o)
    kw: dict[str, Any] = {
        "equal": model.equal, "criterion_tol": o.criterion_tol, "audit_tol": o.audit_tol,
        "audit_add": o.audit_add, "audit_max": o.audit_max, "audit_nx": o.audit_nx, "audit_nu": o.audit_nu,
        "initgrid_x": o.initgrid_x, "initgrid_u": o.initgrid_u, "audit_x": o.audit_x, "audit_u": o.audit_u,
        "solver": solver, "solver_options_criterion": o.solver_options_criterion,
        "solver_options_bounds": o.solver_options_bounds,
    }  # fmt: skip
    if boot and orig is not None:
        assert orig.audit is not None
        grids: Grids = orig.audit.grids
        res = audit(
            model.spec0, model.spec1, crit, model.gstar, data, restrictions, audit_grid=grids,
            log=[], **kw,
        )  # fmt: skip
        est = _Estimate(model, False, method, solver, audit=res)
        orig_crit = orig.audit.criterion
        if o.specification_test and isinstance(crit, L1Criterion) and orig_crit > 0:
            orig_l1 = orig.model.criterion()
            assert isinstance(orig_l1, L1Criterion)
            c, boot_model = lp_setup_criterion_boot(
                crit, orig_l1, orig_crit, o.criterion_tol, res.constraints, model.equal
            )
            sol = run_lp(c, boot_model, "min", solver, o.solver_options_criterion)
            if sol.obj is None:
                raise RuntimeError(
                    f"The specification test LP returned no solution ({sol.status_str})"
                )
            est.spec_stat = float(sol.obj)
        return est
    log.append("Performing audit procedure...")
    log.append(f"    Solver: {solver}")
    # The R package retries up to three times with an initial grid 1.5 times
    # larger when a bound problem reports an unbounded or suboptimal status.
    nx, nu = o.initgrid_nx, o.initgrid_nu
    for attempt in range(4):
        try:
            res = audit(
                model.spec0, model.spec1, crit, model.gstar, data, restrictions,
                initgrid_nx=nx, initgrid_nu=nu, rng=rng, log=log, **kw,
            )  # fmt: skip
            return _Estimate(model, False, method, solver, audit=res)
        except AuditError as err:
            if err.status not in (3, 4, 6) or attempt == 3:
                raise
            nx = min(int(np.ceil(nx * 1.5)), o.audit_nx)
            nu = min(int(np.ceil(nu * 1.5)), o.audit_nu)
            log.append("    Restarting audit with new settings:")
            log.append(f"    initgrid_nx = {nx}")
            log.append(f"    initgrid_nu = {nu}")
    raise AssertionError("unreachable")


# -- the estimator ------------------------------------------------------------------------


def _solver_options(
    solver: str | None,
    options: dict[str, Any] | None,
    criterion: dict[str, Any] | None,
    bounds: dict[str, Any] | None,
    presolve: bool | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve the option dictionaries as the R package does.

    ``solver_options`` applies to both problems and overrides the specific
    lists; otherwise each problem uses its own list. ``solver_presolve`` is
    written into the options of HiGHS (``presolve``) and Gurobi
    (``Presolve``).
    """
    crit_opts = dict(options) if options is not None else dict(criterion or {})
    bound_opts = dict(options) if options is not None else dict(bounds or {})
    if presolve is not None:
        name = solver or default_solver()
        key = {"highs": "presolve", "gurobi": "Presolve"}.get(name.lower())
        if key is not None:
            crit_opts[key] = bool(presolve) if key == "presolve" else int(presolve)
            bound_opts[key] = bool(presolve) if key == "presolve" else int(presolve)
    return crit_opts or None, bound_opts or None


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
    target_weight0: Sequence[Piece] | None = None,
    target_weight1: Sequence[Piece] | None = None,
    target_knots0: Sequence[Piece] = (),
    target_knots1: Sequence[Piece] = (),
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
    solver_presolve: bool | None = None,
    solver_options_criterion: dict[str, Any] | None = None,
    solver_options_bounds: dict[str, Any] | None = None,
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
        Options passed to the solver for every problem; overrides
        ``solver_options_criterion`` and ``solver_options_bounds``.
    solver_presolve : bool, optional
        Turn the solver's presolve on or off (HiGHS and Gurobi).
    solver_options_criterion, solver_options_bounds : dict, optional
        Options specific to the criterion problem and to the bound problems.
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
    if is_formula:
        ptreat = propensity.partition("~")[0].strip()
        if treat is not None and treat != ptreat:
            raise ValueError(
                f"'treat' ({treat!r}) differs from the dependent variable of the propensity "
                f"score formula ({ptreat!r})"
            )
        treat = ptreat
        if target is not None and target.lower() in ("late", "avglate"):
            late_vars = set(late_from or {}) | set(late_to or {})
            if not late_vars <= formula_vars(propensity):
                raise ValueError(
                    "All variables in 'late_to' and 'late_from' must be included in the "
                    "propensity score model"
                )
    if len({get_xz(f)[0] for f in ivlike_list}) > 1:
        raise ValueError("Multiple response variables specified in the IV-like specifications")
    mtr_vars: set[str] = set()
    for m in (m0, m1):
        mtr_vars |= formula_vars(m) if isinstance(m, str) else {c for _, c in m if c}
    if treat in mtr_vars:
        raise ValueError("Treatment variable cannot be included in the MTRs")
    o = SimpleNamespace(**options)
    o.treat = treat
    o.ivlike = ivlike_list
    o.subset = [subset] if isinstance(subset, str) else subset
    o.target = target.lower() if target is not None else None
    o.solver_options_criterion, o.solver_options_bounds = _solver_options(
        solver, solver_options, solver_options_criterion, solver_options_bounds, solver_presolve
    )

    # -- data ----------------------------------------------------------------
    formulas = [f for f in (m0, m1) if isinstance(f, str)] + ivlike_list
    if equal_coef:
        formulas.append(equal_coef)
    extra_cols = [c for f in (m0, m1) if not isinstance(f, str) for _, c in f if c]
    extra = [c for c in (outcome, treat) if c] + extra_cols
    (formulas if is_formula else extra).append(propensity)
    cols = required_columns(data, formulas, extra)
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

    # -- estimation ------------------------------------------------------------
    est = ivmte_estimate(data, o, rng, log)
    model = est.model
    target_name = "custom" if custom else str(o.target)
    n_moments = model.moments.n_independent if model.moments else None
    common: dict[str, Any] = {
        "target": target_name, "gstar": pd.Series(model.gstar, index=list(model.names)),
        "propensity": model.prop, "specs": (model.spec0, model.spec1), "target_gammas": model.target,
        "messages": log, "options": options, "levels": tuple(levels), "ci_type": ci_type,
        "moments": n_moments, "ivlike": model.moments, "solver": est.solver, "method": est.method,
    }  # fmt: skip
    if est.point:
        assert est.theta is not None
        j_test = None
        if est.fit is not None and est.fit.j_stat is not None:
            j_test = {"stat": est.fit.j_stat, "df": est.fit.j_df, "p_value": est.fit.j_pvalue}
        res = IVMTEResult(
            bounds=None, point_estimate=est.estimate,
            mtr_coef=pd.Series(est.theta, index=list(model.names)), gstar_coef=None, audit=None,
            criterion=None, j_test=j_test, **common,
        )  # fmt: skip
    else:
        assert est.audit is not None
        gstar_coef = pd.DataFrame(
            {
                "min": est.audit.theta_min,
                "max": est.audit.theta_max,
                "criterion": est.audit.theta_crit,
            },
            index=list(model.names),
        )
        res = IVMTEResult(
            bounds=(est.audit.lower, est.audit.upper), point_estimate=None, mtr_coef=None,
            gstar_coef=gstar_coef, audit=est.audit, criterion=est.audit.criterion, j_test=None,
            **common,
        )  # fmt: skip
    if bootstraps:
        _bootstrap(res, est, o, rng)
    if noisy:
        print("\n".join(log))
    return res


def _bootstrap(
    res: IVMTEResult, est: _Estimate, o: SimpleNamespace, rng: np.random.Generator
) -> None:
    """Bootstrap the estimate or the bounds and fill the inference attributes of ``res``.

    Bound replicates reuse the audit grid of the sample, as in the R
    package; GMM replicates are recentred at the sample moments.
    """
    model = est.model
    n = model.n
    m = o.bootstraps_m or n
    levels = list(res.levels)

    def replicate(idx: NDArray[np.intp]) -> Replicate:
        rep = ivmte_estimate(model.data.iloc[idx].reset_index(drop=True), o, rng, [], orig=est)
        if rep.point:
            return Replicate(
                point=rep.estimate, mtr=rep.theta, propensity=rep.model.prop.params,
                j_stat=rep.fit.j_stat if rep.fit is not None else None,
            )  # fmt: skip
        assert rep.audit is not None
        return Replicate(
            bounds=(rep.audit.lower, rep.audit.upper), propensity=rep.model.prop.params,
            spec_stat=rep.spec_stat,
        )  # fmt: skip

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = _resample(n, m, o.bootstraps_replace, replicate, o.bootstraps, rng)
    res.bootstraps, res.bootstraps_failed = boot.n_draws, boot.n_failed
    if res.bounds is not None:
        assert boot.bounds is not None and res.audit is not None
        res.bounds_bootstraps = boot.bounds
        res.bounds_se = np.std(boot.bounds, axis=0, ddof=1)
        res.bounds_ci = {
            k: bound_ci(res.bounds, boot.bounds, n, m, levels, k) for k in ("backward", "forward")
        }
        res.p_value = {
            k: bound_pvalue(res.bounds, boot.bounds, n, m, k) for k in ("backward", "forward")
        }
        if boot.spec_stats is not None:
            res.specification_p_value = float(np.mean(res.audit.criterion <= boot.spec_stats))
    else:
        assert res.point_estimate is not None and res.mtr_coef is not None
        assert boot.points is not None and boot.mtr is not None
        res.point_estimate_bootstraps = boot.points
        res.point_estimate_se = float(np.std(boot.points, ddof=1))
        res.point_estimate_ci = _point_ci(res.point_estimate, boot.points, levels)
        res.p_value = _point_pvalues(res.point_estimate, boot.points)
        res.mtr_bootstraps = boot.mtr
        res.mtr_se = pd.Series(np.std(boot.mtr, axis=0, ddof=1), index=res.mtr_coef.index)
        res.mtr_ci = _coef_ci(res.mtr_coef.to_numpy(), boot.mtr, list(res.mtr_coef.index), levels)
        if boot.j_stats is not None and res.j_test is not None:
            res.j_test_bootstraps = boot.j_stats
            res.j_test["bootstrap_p_value"] = float(np.mean(boot.j_stats >= res.j_test["stat"]))
    params = res.propensity.params
    if boot.propensity is not None and params is not None:
        names = list(res.propensity.names)
        res.propensity_bootstraps = boot.propensity
        res.propensity_se = pd.Series(np.std(boot.propensity, axis=0, ddof=1), index=names)
        res.propensity_ci = _coef_ci(params, boot.propensity, names, levels)
    res.messages.append(f"Bootstraps: {boot.n_draws} ({boot.n_failed} failed draws)")
