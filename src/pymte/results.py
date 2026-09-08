"""Result container returned by :func:`pymte.ivmte`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pymte.audit import AuditResult, fmt_result
from pymte.ivlike import MomentSet
from pymte.mtr import MTRSpec
from pymte.propensity import Propensity
from pymte.weights import TargetGammas


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
        Audit diagnostics (partially identified case).
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
    j_test: dict[str, float] | None
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
                f"Bounds on the target parameter: [{fmt_result(self.bounds[0])}, {fmt_result(self.bounds[1])}]"
            )
            assert self.audit is not None
            lines.append(f"Audit terminated successfully after {self.audit.audit_count} round(s)")
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
