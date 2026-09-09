"""Marginal treatment response (MTR) specifications.

An MTR function ``m_d(u, x)`` is specified by a one-sided formula in the
unobservable ``u`` and covariates, for example ``"u + I(u**2) + yob + u:yob"``
or ``"uSplines(degree=2, knots=[.1, .3]) * yob"``. Every term is the product
of a covariate part (evaluated by :mod:`formulaic`) and a *u-part*, which is
either a monomial ``u**k`` or one B-spline basis function in ``u``. Because
the u-part is known analytically, integrals of the MTR against weights in
``u`` are computed exactly.

:func:`polyparse` parses a formula into an :class:`MTRSpec`, :func:`gen_gamma`
integrates the terms against interval weights (with :func:`gen_gamma_splines`
for the spline blocks), following the functions of the same names in the R
package.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from formulaic import Formula, ModelSpec
from numpy.typing import ArrayLike, NDArray

from pymte.splines import USpline

_BAD_U_MESSAGE = (
    "The unobservable variable '{u}' must enter as a monomial ('{u}', 'I({u}**2)', ...) "
    "or through 'uSplines(...)', optionally interacted with covariates, e.g. 'x:{u}' or "
    "'x:I({u}**3)'. Terms like 'exp({u})' or 'I((x*{u})**2)' are not allowed: {term}"
)


def _one(*args: Any, **kwargs: Any) -> float:
    return 1.0


def _c(*args: float) -> list[float]:
    return list(args)


def _u_patterns(uname: str) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    u = re.escape(uname)
    mono = re.compile(rf"^(?:{u}|I\(\s*{u}\s*(?:\*\*|\^)\s*(\d+)\s*\))$")
    spline = re.compile(r"^uSplines?\((.*)\)$", re.DOTALL)
    mentions = re.compile(rf"(?<![\w.]){u}(?![\w.])")
    return mono, spline, mentions


def _rewrite_caret(formula: str, uname: str) -> str:
    """Rewrite the R spelling ``I(u^k)`` as ``I(u**k)``."""
    u = re.escape(uname)
    return re.sub(rf"I\(\s*{u}\s*\^\s*(\d+)\s*\)", rf"I({uname}**\1)", formula)


def _strip_tilde(formula: str) -> str:
    formula = formula.strip()
    if formula.startswith("~"):
        formula = formula[1:]
    if "~" in formula:
        raise ValueError(f"MTR formulas must be one-sided: {formula!r}")
    return formula


@dataclass(frozen=True)
class SplineBlock:
    """One B-spline basis and the covariate columns it is interacted with."""

    spline: USpline
    columns: tuple[int, ...]
    names: tuple[str, ...]


@dataclass(frozen=True)
class MTRSpec:
    """A parsed MTR specification for one treatment arm.

    Instances are created with :func:`polyparse` or :meth:`from_columns`. The
    materialised design is evaluated with the unobservable set to one, so
    every column of :meth:`covariate_matrix` is the covariate part of a
    term; the u-part is recorded separately as an exponent (polynomial
    terms) or a spline basis.

    Attributes
    ----------
    formula : str
        The formula as given by the user.
    uname : str
        Name of the unobservable.
    names : tuple of str
        Coefficient names in the order used everywhere in the package:
        polynomial terms first (in formula order), then spline blocks
        ``uS{j}.{b}:{interaction}`` with ``j`` the spline index, ``b`` the
        basis index (1-based) and ``1`` for no interaction.
    """

    formula: str
    uname: str
    _spec: ModelSpec | None
    poly_columns: tuple[int, ...]
    exponents: tuple[int, ...]
    poly_names: tuple[str, ...]
    splines: tuple[SplineBlock, ...]
    _columns: tuple[str | None, ...] = ()

    @classmethod
    def from_columns(
        cls,
        terms: Sequence[tuple[int | USpline, str | None]],
        data: pd.DataFrame,
        uname: str = "u",
    ) -> MTRSpec:
        """Build a specification from an explicit list of terms.

        Parameters
        ----------
        terms : sequence of (u_part, column)
            Each term is the product of a u-part and a data column. The
            u-part is an integer exponent (``0`` for no dependence on ``u``)
            or a :class:`~pymte.USpline`; the column is a name in ``data`` or
            ``None`` for a constant.
        data : pandas.DataFrame
            Data the columns refer to.
        uname : str, default "u"
            Name used for the unobservable in coefficient names.

        Returns
        -------
        MTRSpec

        Examples
        --------
        ``[(0, None), (1, None), (0, "x"), (1, "x")]`` is ``"~ u + x + u:x"``.
        """
        columns: list[str | None] = []
        poly_columns: list[int] = []
        exponents: list[int] = []
        poly_names: list[str] = []
        spline_specs: list[USpline] = []
        spline_inter: list[list[tuple[int, str]]] = []
        for u_part, col in terms:
            if col is not None and col not in data.columns:
                raise ValueError(f"Column {col!r} not found in the data")
            columns.append(col)
            idx = len(columns) - 1
            if isinstance(u_part, USpline):
                inter = col or "1"
                if u_part in spline_specs:
                    spline_inter[spline_specs.index(u_part)].append((idx, inter))
                else:
                    spline_specs.append(u_part)
                    spline_inter.append([(idx, inter)])
                continue
            e = int(u_part)
            upart = "" if e == 0 else (uname if e == 1 else f"I({uname} ** {e})")
            name = ":".join(part for part in (upart, col) if part) or "Intercept"
            poly_columns.append(idx)
            exponents.append(e)
            poly_names.append(name)
        blocks = tuple(
            SplineBlock(sp, tuple(c for c, _ in inter), tuple(n for _, n in inter))
            for sp, inter in zip(spline_specs, spline_inter, strict=True)
        )
        return cls(
            formula="",
            uname=uname,
            _spec=None,
            poly_columns=tuple(poly_columns),
            exponents=tuple(exponents),
            poly_names=tuple(poly_names),
            splines=blocks,
            _columns=tuple(columns),
        )

    # -- structure ---------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Coefficient names, polynomial terms then spline blocks."""
        out = list(self.poly_names)
        for j, block in enumerate(self.splines, start=1):
            for inter in block.names:
                out += [f"uS{j}.{b}:{inter}" for b in range(1, block.spline.n_basis + 1)]
        return tuple(out)

    @property
    def n_coef(self) -> int:
        """Number of MTR coefficients."""
        return len(self.names)

    @property
    def covariates(self) -> tuple[str, ...]:
        """Names of the data columns the specification depends on."""
        if self._spec is None:
            return tuple(sorted({c for c in self._columns if c is not None}))
        return tuple(v for v in sorted(self._spec.required_variables) if v != self.uname)

    def covariate_matrix(self, data: pd.DataFrame) -> NDArray[np.float64]:
        """Evaluate every term's covariate part on ``data`` (u set to one)."""
        if self._spec is None:
            cols = [
                np.ones(len(data)) if c is None else np.asarray(data[c], dtype=float)
                for c in self._columns
            ]
            return np.column_stack(cols) if cols else np.empty((len(data), 0))
        frame = data.assign(**{self.uname: 1.0})
        context = {"uSplines": _one, "uSpline": _one, "c": _c}
        return np.asarray(self._spec.get_model_matrix(frame, context=context), dtype=float)

    def design(self, data: pd.DataFrame, u: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the MTR basis at ``(u_i, x_i)`` for each row.

        Parameters
        ----------
        data : pandas.DataFrame
            Covariates, one row per evaluation point.
        u : array_like
            Value of the unobservable for each row (or a scalar).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, n_coef)``; ``design @ theta`` is ``m_d``.
        """
        n = len(data)
        uu = np.broadcast_to(np.asarray(u, dtype=float), n)
        x = self.covariate_matrix(data)
        parts = []
        if self.poly_columns:
            e = np.asarray(self.exponents, dtype=float)
            parts.append(x[:, list(self.poly_columns)] * uu[:, None] ** e)
        for block in self.splines:
            basis = block.spline.basis(uu)
            for col in block.columns:
                parts.append(x[:, [col]] * basis)
        out = np.hstack(parts) if parts else np.empty((n, 0))
        return np.asarray(out, dtype=float)


def polyparse(formula: str, data: pd.DataFrame, uname: str = "u") -> MTRSpec:
    """Parse an MTR formula into polynomial and spline terms in the unobservable.

    Parameters
    ----------
    formula : str
        One-sided formula, e.g. ``"u + I(u**2) + x"``. The R spelling
        ``I(u^2)`` is accepted. ``uSplines(degree, knots=[...],
        intercept=False)`` (or ``uSpline``, as in R) adds a B-spline basis in ``u``.
    data : pandas.DataFrame
        Data used to encode covariates (factor levels, interactions).
    uname : str, default "u"
        Name of the unobservable variable in the formula.

    Returns
    -------
    MTRSpec

    Raises
    ------
    ValueError
        When ``uname`` enters other than as a monomial or through
        ``uSplines``, or the formula is two-sided.
    """
    rhs = _rewrite_caret(_strip_tilde(formula), uname)
    mono_re, spline_re, mentions_re = _u_patterns(uname)
    # Validate the u-terms before formulaic evaluates anything.
    for term in cast(Iterable[Any], Formula(rhs)):
        exprs = [f.expr for f in term.factors if mentions_re.search(f.expr)]
        if len(exprs) > 1 or any(not mono_re.match(e) for e in exprs):
            raise ValueError(_BAD_U_MESSAGE.format(u=uname, term=str(term)))
    context = {"uSplines": _one, "uSpline": _one, "c": _c}
    frame = data.assign(**{uname: 1.0})
    spec = ModelSpec.from_spec(rhs, context=context)
    matrix = spec.get_model_matrix(frame, context=context)
    spec = cast(ModelSpec, matrix.model_spec)

    poly_columns: list[int] = []
    exponents: list[int] = []
    poly_names: list[str] = []
    spline_specs: list[USpline] = []
    spline_inter: list[list[tuple[int, str]]] = []
    colnames = list(matrix.columns)
    for term, indices in spec.term_indices.items():
        u_factors = [
            f.expr for f in term.factors if mentions_re.search(f.expr) or spline_re.match(f.expr)
        ]
        if len(u_factors) > 1:
            raise ValueError(_BAD_U_MESSAGE.format(u=uname, term=str(term)))
        sp: USpline | None = None
        exponent: int | None = 0
        if u_factors:
            expr = u_factors[0]
            m = mono_re.match(expr)
            if m:
                exponent = int(m.group(1)) if m.group(1) is not None else 1
            else:
                exponent = None
                sp = eval(  # noqa: S307 - the expression comes from the formula
                    "USpline" + expr[expr.index("(") :], {"USpline": USpline, "c": _c}
                )
        for col in indices:
            name = colnames[col]
            if exponent is not None:
                poly_columns.append(col)
                exponents.append(exponent)
                poly_names.append(name)
                continue
            inter = ":".join(p for p in name.split(":") if not spline_re.match(p)) or "1"
            assert sp is not None
            if sp in spline_specs:
                spline_inter[spline_specs.index(sp)].append((col, inter))
            else:
                spline_specs.append(sp)
                spline_inter.append([(col, inter)])
    blocks = tuple(
        SplineBlock(sp, tuple(c for c, _ in inter), tuple(n for _, n in inter))
        for sp, inter in zip(spline_specs, spline_inter, strict=True)
    )
    return MTRSpec(
        formula=formula,
        uname=uname,
        _spec=spec,
        poly_columns=tuple(poly_columns),
        exponents=tuple(exponents),
        poly_names=tuple(poly_names),
        splines=blocks,
    )


def gen_gamma_splines(
    block: SplineBlock, x: NDArray[np.float64], lb: NDArray[np.float64], ub: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Integrals of one spline block: basis integrals times each interacted covariate column."""
    integ = block.spline.integral(lb, ub)
    return np.hstack([x[:, [col]] * integ for col in block.columns])


def gen_gamma(
    spec: MTRSpec,
    data: pd.DataFrame,
    lb: ArrayLike,
    ub: ArrayLike,
    multiplier: ArrayLike = 1.0,
    rows: ArrayLike | None = None,
    means: bool = True,
) -> NDArray[np.float64]:
    """Integrals ``multiplier_i * int_{lb_i}^{ub_i} b_j(u) x_ij du`` of the MTR terms.

    Parameters
    ----------
    spec : MTRSpec
        The MTR specification.
    data : pandas.DataFrame
        Covariates, one row per observation.
    lb, ub : array_like
        Integration limits, scalars or one value per row.
    multiplier : array_like, default 1.0
        Weight multiplying the integral, scalar or one value per row.
    rows : array_like of bool, optional
        Restrict the computation to these rows.
    means : bool, default True
        Average over rows (the ``gamma`` vector of a moment condition) or
        return the per-observation integrals, shape ``(n, n_coef)``.

    Returns
    -------
    numpy.ndarray
    """
    n = len(data)
    lo = np.broadcast_to(np.asarray(lb, dtype=float), n)
    hi = np.broadcast_to(np.asarray(ub, dtype=float), n)
    w = np.broadcast_to(np.asarray(multiplier, dtype=float), n)
    if rows is not None:
        mask = np.asarray(rows, dtype=bool)
        data, lo, hi, w = data.loc[mask], lo[mask], hi[mask], w[mask]
    x = spec.covariate_matrix(data)
    parts = []
    if spec.poly_columns:
        e = np.asarray(spec.exponents, dtype=float)
        mono = (hi[:, None] ** (e + 1) - lo[:, None] ** (e + 1)) / (e + 1)
        parts.append(x[:, list(spec.poly_columns)] * mono)
    for block in spec.splines:
        parts.append(gen_gamma_splines(block, x, lo, hi))
    out = np.hstack(parts) if parts else np.empty((len(x), 0))
    out = np.asarray(out * w[:, None], dtype=float)
    return out.mean(axis=0) if means else out
