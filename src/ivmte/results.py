"""Result container returned by :func:`ivmte.ivmte`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ivmte.audit import AuditResult, _fmt
from ivmte.ivlike import MomentSet
from ivmte.mtr import MTRSpec
from ivmte.propensity import Propensity
from ivmte.weights import TargetGammas


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


@dataclass
class IVMTEResult:
    """Estimates, bounds and diagnostics from :func:`ivmte.ivmte`.

    Exactly one of ``bounds`` and ``point_estimate`` is set. Attribute names
    follow the R package with dots replaced by underscores.

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
        Hansen J statistic, degrees of freedom and p-value (GMM case).
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

    def summary(self) -> str:
        """Return a text summary in the style of R's ``summary.ivmte``."""
        s0, s1 = self.specs
        lines = []
        if self.bounds is not None:
            lines.append(
                f"Bounds on the target parameter: [{_fmt(self.bounds[0])}, {_fmt(self.bounds[1])}]"
            )
            assert self.audit is not None
            lines.append(f"Audit terminated successfully after {self.audit.audit_count} round(s)")
        else:
            assert self.point_estimate is not None
            lines.append(f"Point estimate of the target parameter: {_fmt(self.point_estimate)}")
        lines.append(f"MTR coefficients: {s0.n_coef + s1.n_coef}")
        if self.ivlike is not None:
            lines.append(f"Independent/total moments: {self.moments}/{self.ivlike.n_moments}")
        if self.criterion is not None:
            lines.append(f"Minimum criterion: {_fmt(self.criterion)}")
        lines.append(f"Solver: {self.solver}")
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
            "options": _to_plain(self.options),
        }
