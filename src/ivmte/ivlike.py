"""IV-like estimands and their representation as moments of the MTR coefficients.

An IV-like estimand is a coefficient ``beta_k`` from a linear regression
(OLS or two-stage least squares) of the outcome on treatment, covariates and
instruments. By Mogstad, Santos and Torgovitsky (2018) every such
coefficient equals ``E[s_k(D, Z) Y]`` for a known weight ``s_k``, and hence
is linear in the MTR coefficients:

``beta_k = Gamma_0[k] @ theta0 + Gamma_1[k] @ theta1``

with ``Gamma_d[k, j] = E[ s_k(d, Z) * int b_j(u, X) 1{u in I_d(p)} du ]``,
``I_0(p) = [p, 1]`` and ``I_1(p) = [0, p]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from formulaic import ModelSpec
from numpy.typing import NDArray

from ivmte.mtr import MTRSpec
from ivmte.propensity import Propensity


@dataclass(frozen=True)
class IVLikeFit:
    """A fitted IV-like regression and the weights of its selected coefficients.

    Attributes
    ----------
    formula : str
        The regression formula.
    components : tuple of str
        Names of the selected coefficients.
    beta : numpy.ndarray
        The selected coefficient estimates.
    s0, s1 : numpy.ndarray
        Weights ``s_k(0, Z_i)`` and ``s_k(1, Z_i)``, shape ``(n_subset, K)``.
    rows : numpy.ndarray of bool
        Observations used (the ``subset``).
    y : numpy.ndarray
        Outcome on the subset.
    d : numpy.ndarray
        Treatment on the subset.
    """

    formula: str
    components: tuple[str, ...]
    beta: NDArray[np.float64]
    s0: NDArray[np.float64]
    s1: NDArray[np.float64]
    rows: NDArray[np.bool_]
    y: NDArray[np.float64]
    d: NDArray[np.float64]
    instrumented: bool


@dataclass(frozen=True)
class MomentSet:
    """The IV-like moment conditions ``beta = Gamma0 @ theta0 + Gamma1 @ theta1``.

    Attributes
    ----------
    fits : tuple of IVLikeFit
        The underlying regressions.
    names : tuple of str
        One name per moment, ``"{spec}:{coefficient}"``.
    beta : numpy.ndarray
        Estimated IV-like coefficients, shape ``(S,)``.
    gamma0, gamma1 : numpy.ndarray
        Averaged integrals, shapes ``(S, J0)`` and ``(S, J1)``.
    gamma0_obs, gamma1_obs : tuple of numpy.ndarray
        Per-observation integrals for each moment, on its subset.
    ys : tuple of numpy.ndarray
        Per-observation ``s_k(D_i, Z_i) Y_i`` for each moment, on its subset.
    rows : tuple of numpy.ndarray
        Subset mask of each moment.
    n_independent : int
        Rank of the stacked weight matrix, the number of linearly independent
        moment conditions.
    """

    fits: tuple[IVLikeFit, ...]
    names: tuple[str, ...]
    beta: NDArray[np.float64]
    gamma0: NDArray[np.float64]
    gamma1: NDArray[np.float64]
    gamma0_obs: tuple[NDArray[np.float64], ...]
    gamma1_obs: tuple[NDArray[np.float64], ...]
    ys: tuple[NDArray[np.float64], ...]
    rows: tuple[NDArray[np.bool_], ...]
    n_independent: int

    @property
    def n_moments(self) -> int:
        """Number of moment conditions."""
        return len(self.beta)


def _split_formula(formula: str) -> tuple[str, str, str | None]:
    lhs, sep, rhs = formula.partition("~")
    if not sep or not lhs.strip():
        raise ValueError(f"IV-like formulas need an outcome on the left-hand side: {formula!r}")
    x_part, bar, z_part = rhs.partition("|")
    return lhs.strip(), x_part.strip(), (z_part.strip() if bar else None)


def _design(spec_str: str, data: pd.DataFrame) -> tuple[NDArray[np.float64], ModelSpec]:
    mm = ModelSpec.from_spec(spec_str).get_model_matrix(data)
    return np.asarray(mm, dtype=float), cast(ModelSpec, mm.model_spec)


def _check_rank(x: NDArray[np.float64], what: str) -> None:
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError(f"The {what} of an IV-like specification are collinear")


def _resolve_components(
    components: Sequence[str] | None, names: list[str], formula: str
) -> list[int]:
    if not components:
        return list(range(len(names)))
    lookup = {n: i for i, n in enumerate(names)}
    lookup["intercept"] = lookup.get("Intercept", -1)
    out = []
    for c in components:
        if c not in lookup or lookup[c] < 0:
            raise ValueError(
                f"Component {c!r} is not a coefficient of {formula!r}; available: {names}"
            )
        out.append(lookup[c])
    return out


def fit_ivlike(
    data: pd.DataFrame,
    formula: str,
    treat: str,
    components: Sequence[str] | None = None,
    subset: str | None = None,
) -> IVLikeFit:
    """Fit one IV-like regression and compute the weights of its coefficients.

    Parameters
    ----------
    data : pandas.DataFrame
        Estimation sample.
    formula : str
        ``"y ~ d + x"`` for OLS or ``"y ~ d + x | z + x"`` for TSLS, where the
        part after ``|`` lists the instruments (exogenous covariates must be
        repeated there, as in R).
    treat : str
        Name of the treatment variable.
    components : sequence of str, optional
        Coefficients to use as estimands; ``"intercept"`` names the constant.
        Default: all coefficients.
    subset : str, optional
        A :meth:`pandas.DataFrame.eval` expression selecting the rows used.

    Returns
    -------
    IVLikeFit
    """
    lhs, x_part, z_part = _split_formula(formula)
    rows = np.ones(len(data), dtype=bool)
    if subset:
        rows = np.asarray(data.eval(subset), dtype=bool)
        if not rows.any():
            raise ValueError(f"The subset {subset!r} selects no observations")
    sub = data.loc[rows]
    y = np.asarray(sub[lhs], dtype=float)
    d = np.asarray(sub[treat], dtype=float)
    n = len(sub)
    x, xspec = _design(x_part, sub)
    _check_rank(x, "regressors")
    xnames = list(xspec.column_names)
    cpos = _resolve_components(components, xnames, formula)
    x0 = np.asarray(xspec.get_model_matrix(sub.assign(**{treat: 0})), dtype=float)
    x1 = np.asarray(xspec.get_model_matrix(sub.assign(**{treat: 1})), dtype=float)
    if z_part is None:
        exx_inv = np.linalg.inv(x.T @ x / n)
        beta = exx_inv @ (x.T @ y / n)
        s0, s1 = x0 @ exx_inv, x1 @ exx_inv
    else:
        z, zspec = _design(z_part, sub)
        _check_rank(z, "instruments")
        if z.shape[1] < x.shape[1]:
            raise ValueError("TSLS needs at least as many instruments as regressors")
        z0 = np.asarray(zspec.get_model_matrix(sub.assign(**{treat: 0})), dtype=float)
        z1 = np.asarray(zspec.get_model_matrix(sub.assign(**{treat: 1})), dtype=float)
        exz = x.T @ z / n
        pi = exz @ np.linalg.inv(z.T @ z / n)
        w = np.linalg.solve(pi @ exz.T, pi)  # K x L
        beta = w @ (z.T @ y / n)
        s0, s1 = z0 @ w.T, z1 @ w.T
    return IVLikeFit(
        formula=formula,
        components=tuple(xnames[i] for i in cpos),
        beta=beta[cpos],
        s0=np.asarray(s0[:, cpos], dtype=float),
        s1=np.asarray(s1[:, cpos], dtype=float),
        rows=rows,
        y=y,
        d=d,
        instrumented=z_part is not None,
    )


def build_moments(
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
    fits, names, betas, g0m, g1m, g0o, g1o, ys, rows = [], [], [], [], [], [], [], [], []
    p = prop.phat
    for i, (f, c, s) in enumerate(zip(formulas, comps, subs, strict=True)):
        fit = fit_ivlike(data, f, prop.treat, components=c, subset=s)
        fits.append(fit)
        sub = data.loc[fit.rows]
        psub = p[fit.rows]
        for j, name in enumerate(fit.components):
            g0 = spec0.gamma(sub, psub, 1.0, fit.s0[:, j])
            g1 = spec1.gamma(sub, 0.0, psub, fit.s1[:, j])
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
