"""Design matrices of IV-like specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from formulaic import ModelSpec
from numpy.typing import NDArray

from pymte.callcheck import get_xz


@dataclass(frozen=True)
class Design:
    """Matrices of one IV-like regression ``y ~ x | z`` on its estimation subset.

    Attributes
    ----------
    y : numpy.ndarray
        Outcome.
    x : numpy.ndarray
        Second-stage regressors.
    z : numpy.ndarray or None
        Instruments (``None`` for OLS).
    x_names : tuple of str
        Column names of ``x``.
    x0, x1 : numpy.ndarray
        ``x`` with the treatment variable fixed to 0 and to 1.
    z0, z1 : numpy.ndarray or None
        The same for ``z``.
    rows : numpy.ndarray of bool
        Rows of the data used (the ``subset``).
    """

    y: NDArray[np.float64]
    x: NDArray[np.float64]
    z: NDArray[np.float64] | None
    x_names: tuple[str, ...]
    x0: NDArray[np.float64]
    x1: NDArray[np.float64]
    z0: NDArray[np.float64] | None
    z1: NDArray[np.float64] | None
    rows: NDArray[np.bool_]


def _matrix(spec_str: str, data: pd.DataFrame) -> tuple[NDArray[np.float64], ModelSpec]:
    mm = ModelSpec.from_spec(spec_str).get_model_matrix(data)
    return np.asarray(mm, dtype=float), cast(ModelSpec, mm.model_spec)


def design(
    formula: str, data: pd.DataFrame, subset: str | None = None, treat: str | None = None
) -> Design:
    """Generate the design matrices of an IV-like specification.

    Parameters
    ----------
    formula : str
        ``"y ~ d + x"`` for OLS or ``"y ~ d + x | z + x"`` for TSLS, where the
        part after ``|`` lists the instruments (exogenous covariates must be
        repeated there, as in R).
    data : pandas.DataFrame
        Estimation sample.
    subset : str, optional
        A :meth:`pandas.DataFrame.eval` expression selecting the rows used.
    treat : str, optional
        Treatment variable; when given, the matrices with the treatment fixed
        to 0 and 1 are also returned (needed for the weights).

    Returns
    -------
    Design
    """
    lhs, x_part, z_part = get_xz(formula)
    rows = np.ones(len(data), dtype=bool)
    if subset:
        rows = np.asarray(data.eval(subset), dtype=bool)
        if not rows.any():
            raise ValueError(f"The subset {subset!r} selects no observations")
    sub = data.loc[rows]
    y = np.asarray(sub[lhs], dtype=float)
    x, xspec = _matrix(x_part, sub)
    z = zspec = None
    if z_part is not None:
        z, zspec = _matrix(z_part, sub)
    if treat is None:
        x0 = x1 = x
        z0 = z1 = z
    else:
        sub0, sub1 = sub.assign(**{treat: 0}), sub.assign(**{treat: 1})
        x0 = np.asarray(xspec.get_model_matrix(sub0), dtype=float)
        x1 = np.asarray(xspec.get_model_matrix(sub1), dtype=float)
        z0 = z1 = None
        if zspec is not None:
            z0 = np.asarray(zspec.get_model_matrix(sub0), dtype=float)
            z1 = np.asarray(zspec.get_model_matrix(sub1), dtype=float)
    return Design(y, x, z, tuple(xspec.column_names), x0, x1, z0, z1, rows)
