"""Helpers for checking and dissecting the arguments of :func:`pymte.ivmte`."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from formulaic import Formula


def get_xz(formula: str) -> tuple[str, str, str | None]:
    """Split an IV-like formula ``y ~ x | z`` into its outcome, regressors and instruments.

    Parameters
    ----------
    formula : str
        Two-sided formula; the part after ``|`` (optional) lists the
        instruments.

    Returns
    -------
    tuple
        Outcome name, regressor specification and instrument specification
        (``None`` for OLS).
    """
    lhs, sep, rhs = formula.partition("~")
    if not sep or not lhs.strip():
        raise ValueError(f"IV-like formulas need an outcome on the left-hand side: {formula!r}")
    x_part, bar, z_part = rhs.partition("|")
    return lhs.strip(), x_part.strip(), (z_part.strip() if bar else None)


def formula_vars(formula: str) -> set[str]:
    """Variables referenced by a formula (both sides, instruments included)."""
    vars_: set[str] = set()
    for part in formula.replace("|", "+").split("~"):
        part = part.strip()
        if part:
            vars_ |= set(Formula(part).required_variables)
    return vars_


def required_columns(
    data: pd.DataFrame, formulas: Sequence[str], extra: Sequence[str]
) -> list[str]:
    """List the data columns referenced by the formulas; unknown names such as ``u`` are ignored."""
    names: set[str] = set(extra)
    for f in formulas:
        names |= formula_vars(f)
    return [c for c in data.columns if c in names]
