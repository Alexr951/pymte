"""Propensity score models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from formulaic import ModelSpec
from numpy.typing import NDArray
from scipy.stats import norm

LINKS = ("logit", "probit", "linear")


@dataclass(frozen=True)
class Propensity:
    """A fitted propensity score model or a user-supplied score.

    Attributes
    ----------
    treat : str
        Name of the treatment variable.
    phat : numpy.ndarray
        Fitted probabilities for the estimation sample, clipped to [0, 1].
    link : str or None
        ``"logit"``, ``"probit"`` or ``"linear"``; ``None`` when the score
        was read from a data column.
    params : numpy.ndarray or None
        Estimated coefficients (``None`` for a supplied score).
    names : tuple of str
        Coefficient names.
    """

    treat: str
    phat: NDArray[np.float64]
    link: str | None
    params: NDArray[np.float64] | None
    names: tuple[str, ...]
    _spec: ModelSpec | None
    formula: str | None = None
    variable: str | None = None

    @property
    def fitted(self) -> bool:
        """Whether a model was fitted (as opposed to a supplied column)."""
        return self._spec is not None

    def predict(self, data: pd.DataFrame) -> NDArray[np.float64]:
        """Predict propensity scores for new data.

        Parameters
        ----------
        data : pandas.DataFrame
            Covariates and instruments used in the propensity formula.

        Returns
        -------
        numpy.ndarray
            Predicted probabilities, clipped to [0, 1].
        """
        if self._spec is None or self.params is None:
            raise ValueError("Cannot predict from a propensity score supplied as a variable")
        x = np.asarray(self._spec.get_model_matrix(data), dtype=float)
        return _inverse_link(x @ self.params, self.link)


def _inverse_link(index: NDArray[np.float64], link: str | None) -> NDArray[np.float64]:
    if link == "logit":
        out = 1.0 / (1.0 + np.exp(-index))
    elif link == "probit":
        out = norm.cdf(index)
    else:
        out = index
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=float)


def propensity(
    formula: str, data: pd.DataFrame, link: str = "logit", treat: str | None = None
) -> Propensity:
    """Estimate propensity scores, or read them from a column.

    Parameters
    ----------
    formula : str
        Two-sided formula, e.g. ``"d ~ z + x"``, or the name of a column
        holding propensity scores in [0, 1] (then ``treat`` is required).
    data : pandas.DataFrame
        Estimation sample.
    link : {"logit", "probit", "linear"}, default "logit"
        Logistic regression, probit, or a linear probability model. Linear
        fitted values are truncated to [0, 1].
    treat : str, optional
        Name of the treatment variable when ``formula`` names a column.

    Returns
    -------
    Propensity
    """
    if "~" not in formula:
        if treat is None:
            raise ValueError("'treat' is required when 'propensity' names a column of scores")
        phat = np.asarray(data[formula], dtype=float)
        if phat.min() < 0 or phat.max() > 1:
            raise ValueError("Provided propensity scores are not between 0 and 1")
        return Propensity(
            treat=treat, phat=phat, link=None, params=None, names=(), _spec=None, variable=formula
        )
    link = link.lower()
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    lhs, _, rhs = formula.partition("~")
    treat = lhs.strip()
    if not treat:
        raise ValueError(f"The propensity formula needs a left-hand side: {formula!r}")
    matrix = ModelSpec.from_spec(rhs).get_model_matrix(data)
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(data[treat], dtype=float)
    if link == "logit":
        params = sm.Logit(y, x).fit(disp=0, maxiter=200, tol=1e-10).params
    elif link == "probit":
        params = sm.Probit(y, x).fit(disp=0, maxiter=200, tol=1e-10).params
    else:
        params = sm.OLS(y, x).fit().params
    params = np.asarray(params, dtype=float)
    return Propensity(
        treat=treat,
        phat=_inverse_link(x @ params, link),
        link=link,
        params=params,
        names=tuple(matrix.columns),
        _spec=cast(ModelSpec, matrix.model_spec),
        formula=formula,
    )
