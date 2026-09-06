"""Bootstrap inference: confidence regions for bounds, confidence intervals for estimates.

The estimator is re-run on resamples of the data (with replacement, or
without for subsampling; ``bootstraps_m`` observations per draw). For
bounds we form the backward and forward confidence regions of the R
package, for point estimates percentile and normal intervals. p-values
come from inverting the confidence regions (bounds) or from the bootstrap
distribution (points). The specification tests compare the sample
criterion, or the J statistic, with their bootstrap distributions.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm


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
    """Collected replicates and derived inference."""

    n_draws: int
    n_failed: int
    bounds: NDArray[np.float64] | None = None
    points: NDArray[np.float64] | None = None
    mtr: NDArray[np.float64] | None = None
    propensity: NDArray[np.float64] | None = None
    spec_stats: NDArray[np.float64] | None = None
    j_stats: NDArray[np.float64] | None = None


def resample(
    n: int, m: int, replace: bool, replicate: Callable[[NDArray[np.intp]], Replicate],
    bootstraps: int, rng: np.random.Generator,
) -> BootstrapResult:  # fmt: skip
    """Draw ``bootstraps`` usable replicates, redrawing after failures.

    Parameters
    ----------
    n : int
        Sample size.
    m : int
        Observations per draw.
    replace : bool
        Draw with replacement (bootstrap) or without (subsampling).
    replicate : callable
        Maps a row index array to a :class:`Replicate`; may raise
        :class:`BootstrapRetry` or any exception to reject the draw.
    bootstraps : int
        Number of usable replicates wanted.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    BootstrapResult
    """
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


def point_ci(
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


def point_pvalues(estimate: float, resamples: NDArray[np.float64]) -> dict[str, float]:
    """Two-sided p-values for a zero target from the bootstrap distribution."""
    dev = resamples - estimate
    nonpar = float(((dev >= abs(estimate)).sum() + (dev <= -abs(estimate)).sum()) / len(resamples))
    sd = float(np.std(resamples, ddof=1))
    parametric = float(2 * norm.cdf(-abs(estimate - resamples.mean()) / sd)) if sd > 0 else 0.0
    return {"nonparametric": nonpar, "parametric": parametric}


def coef_ci(
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
