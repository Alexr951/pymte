"""Synthetic populations used by the test suite.

These reproduce the data-generating functions of the R package
(``testdata.R``). Each returns a population in which the unobservable has
already been integrated out, so that the IV-like estimands, the target
moments and the bounds can be computed by hand and compared with the
estimator. The full data sets are the populations expanded into
observations, with treatment assigned deterministically within each
covariate cell so that the empirical propensity score equals the
population one.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pymte.mtr import MTRSpec, gen_gamma
from pymte.splines import USpline


def _conditional_means(
    data: pd.DataFrame,
    terms0: Sequence[tuple[int, str | None]],
    coef0: Sequence[float],
    terms1: Sequence[tuple[int, str | None]],
    coef1: Sequence[float],
    p: str = "p",
) -> tuple[np.ndarray, np.ndarray]:
    """``E[m0(u, x) | u > p]`` and ``E[m1(u, x) | u <= p]`` for polynomial MTRs."""
    pz = np.asarray(data[p], dtype=float)
    spec0 = MTRSpec.from_columns(terms0, data)
    spec1 = MTRSpec.from_columns(terms1, data)
    g0 = gen_gamma(spec0, data, pz, 1.0, 1.0 / (1.0 - pz), means=False)
    g1 = gen_gamma(spec1, data, 0.0, pz, 1.0 / pz, means=False)
    return g0 @ np.asarray(coef0), g1 @ np.asarray(coef1)


def _assign_treatment(full: pd.DataFrame, cells: list[str], p: str = "p") -> pd.DataFrame:
    """Within each covariate cell, treat the first ``round(p * N)`` observations."""
    full = full.copy()
    full["d"] = 0
    full["i"] = full.groupby(cells, sort=False).cumcount() + 1
    n_cell = full.groupby(cells, sort=False)["i"].transform("size")
    full["dcut"] = np.round(full[p] * n_cell).astype(int)
    full.loc[full["i"] <= full["dcut"], "d"] = 1
    return full


def gendist_mosquito() -> pd.DataFrame:
    """Generate the mosquito-net population of Mogstad, Santos and Torgovitsky (2018).

    Four instrument values with propensity scores 0.12, 0.29, 0.48 and 0.78,
    100 observations each, and quadratic MTRs ``m0 = 0.9 - 1.1u + 0.3u^2``,
    ``m1 = 0.35 - 0.3u - 0.05u^2``.

    Returns
    -------
    pandas.DataFrame
        Columns ``i``, ``z``, ``pz``, ``d``, ``ey0``, ``ey1``, ``ey``.
    """
    sub_n = 100
    pz = [0.12, 0.29, 0.48, 0.78]
    dt = pd.DataFrame(
        {
            "i": np.tile(np.arange(1, sub_n + 1), 4),
            "z": np.repeat([1, 2, 3, 4], sub_n),
            "pz": np.repeat(pz, sub_n),
        }
    )
    dt["d"] = 0
    for z, cut in zip([1, 2, 3, 4], [12, 29, 48, 78], strict=True):
        dt.loc[(dt["z"] == z) & (dt["i"] <= cut), "d"] = 1
    terms = [(0, None), (1, None), (2, None)]
    dt["ey0"], dt["ey1"] = _conditional_means(
        dt, terms, [0.9, -1.1, 0.3], terms, [0.35, -0.3, -0.05], p="pz"
    )
    dt["ey"] = dt["d"] * dt["ey1"] + (1 - dt["d"]) * dt["ey0"]
    return dt


def gendist_basic() -> dict[str, pd.DataFrame]:
    """Generate the basic population with one covariate and one instrument.

    ``m0 = 2 + x + 2u`` and ``m1 = 6 + 5u^2``; the propensity score is
    ``0.2 - 0.1x + 0.2z`` on the support ``x in {1, 2, 3}``,
    ``z in {1, 2, 3, 4}``.

    Returns
    -------
    dict
        ``"data_full"`` (1,000 observations) and ``"data_dist"`` (the 12
        cells with their probabilities ``f``).
    """
    dtb = pd.DataFrame([(x, z) for z in (1, 2, 3, 4) for x in (1, 2, 3)], columns=["x", "z"])
    dtb["p"] = 0.2 - 0.10 * dtb["x"] + 0.2 * dtb["z"]
    dtb["ey0"], dtb["ey1"] = _conditional_means(
        dtb, [(0, None), (0, "x"), (1, None)], [2, 1, 2], [(0, None), (2, None)], [6, 5]
    )
    dtb["f"] = [0.02, 0.07, 0.02, 0.19, 0.09, 0.10, 0.17, 0.15, 0.01, 0.01, 0.16, 0.01]
    dtb["multiplier"] = dtb["f"] * 1000
    full = dtb.loc[dtb.index.repeat(dtb["multiplier"].round().astype(int))].reset_index(drop=True)
    full = _assign_treatment(full, ["x", "z"]).drop(columns="dcut")
    full["ey"] = full["d"] * full["ey1"] + (1 - full["d"]) * full["ey0"]
    return {"data_full": full, "data_dist": dtb}


def gendist_covariates() -> dict[str, pd.DataFrame]:
    """Generate the population with two covariates, two instruments and polynomial MTRs.

    ``m0 = 0.3 + 0.4x1 - 0.1x2 u - 0.2x2 u^2`` and
    ``m1 = 0.5 + 0.2x1 - 0.1x1 x2 - 0.02u + 0.3x1 u - 0.05x2 u^2``, with a
    logit propensity score rounded to two decimals.

    Returns
    -------
    dict
        ``"data_full"`` (10,000 observations) and ``"data_dist"`` (the 36
        cells with their probabilities ``f``).
    """
    supp = itertools.product((1, 2, 3), (0, 1), (1, 2, 3), (0, 1))
    # expand.grid order: the first variable varies fastest.
    dtc = pd.DataFrame(
        [(x1, x2, z1, z2) for z2, z1, x2, x1 in supp], columns=["x1", "x2", "z1", "z2"]
    )
    dtc["latent"] = -0.5 - 0.4 * dtc["x1"] + 0.2 * dtc["x2"] - 1.0 * dtc["z1"] + 0.3 * dtc["z2"]
    dtc["p"] = np.round(1.0 / (1.0 + np.exp(-dtc["latent"])), 2)
    x1, x2, z1, z2 = (dtc[c] for c in ("x1", "x2", "z1", "z2"))
    ey0, ey1 = _conditional_means(
        dtc.assign(x1x2=x1 * x2),
        [(0, None), (0, "x1"), (1, "x2"), (2, "x2")],
        [0.3, 0.4, -0.1, -0.2],
        [(0, None), (0, "x1"), (0, "x1x2"), (1, None), (1, "x1"), (2, "x2")],
        [0.5, 0.2, -0.1, -0.02, 0.3, -0.05],
    )
    dtc["ey0"], dtc["ey1"] = ey0, ey1
    # Build the distribution marginally, allowing for correlations, then
    # normalise; the rounding adjustments follow the R code.
    f = np.zeros(len(dtc))
    f += np.where(x1 == 0, 0.1, 0.13)
    f += np.select([x2 == 1, x2 == 2, x2 == 3], [0.05, 0.1, 0.01])
    f -= 0.03 * ((x1 == 0) & (z1 == 1))
    f -= 0.01 * ((x1 == 0) & (z2 == 2))
    f -= 0.02 * ((x1 == 0) & (z2 == 3))
    f += 0.02 * ((z1 == 1) & (z2 == 2))
    f += 0.01 * ((z1 == 1) & (z2 == 3))
    f += 0.05 * (z2 == 2) + 0.01 * (z2 == 3)
    f += 0.01 * ((x2 == 2) & (z2 == 2)) + 0.01 * ((x2 == 2) & (z2 == 3))
    f += 0.02 * ((x2 == 3) & (z2 == 2)) + 0.04 * ((x2 == 3) & (z2 == 3))
    f = np.round(f / f.sum(), 2)
    f[((x1 == 1) & (x2 == 1) & (z1 == 0) & (z2 == 1)).to_numpy()] -= 0.01
    f[((x1 == 1) & (x2 == 3) & (z1 == 1) & (z2 == 3)).to_numpy()] -= 0.01
    dtc["f"] = np.round(f, 2)
    dtc["multiplier"] = 100 * 100 * dtc["f"]
    full = dtc.loc[dtc.index.repeat(dtc["multiplier"].round().astype(int))].reset_index(drop=True)
    full = _assign_treatment(full, ["x1", "x2", "z1", "z2"])
    full["ey"] = full["d"] * full["ey1"] + (1 - full["d"]) * full["ey0"]
    return {"data_full": full, "data_dist": dtc}


def gendist_splines() -> dict[str, pd.DataFrame]:
    """Generate the population with spline MTRs.

    ``m1 = 30 + 35x + uSplines(degree=2, knots=(0.3, 0.6))`` and
    ``m0 = x * uSplines(degree=0, knots=(0.2, 0.5, 0.8), intercept=True) +
    uSplines(degree=1, knots=(0.4,), intercept=True) + 20u^2``; the
    propensity score is ``0.5 - 0.1x + 0.2z`` on ``x in {-1, 0, 1}``,
    ``z in {0, 1}``. The outcome columns are the MTRs integrated over
    ``[0, p]`` (treated) and ``[p, 1]`` (untreated).

    Returns
    -------
    dict
        ``"data_full"`` (4,200 observations) and ``"data_dist"`` (the 6
        cells with their probabilities ``f``).
    """
    distr = pd.DataFrame(
        {
            "group": np.arange(1, 7),
            "x": np.tile([-1, 0, 1], 2),
            "z": np.repeat([0, 1], 3),
            "f": [0.1, 0.2, 0.1, 0.1, 0.3, 0.2],
        }
    )
    p = 0.5 - 0.1 * distr["x"] + 0.2 * distr["z"]
    distr["p"] = p
    pz = p.to_numpy()
    u1s1 = USpline(2, (0.3, 0.6), intercept=False).integral(0.0, pz)
    u0s1 = USpline(0, (0.2, 0.5, 0.8), intercept=True).integral(pz, 1.0)
    u0s2 = USpline(1, (0.4,), intercept=True).integral(pz, 1.0)
    distr["ey1"] = 30 * pz + 35 * distr["x"] * pz + u1s1 @ np.array([-75, 45, 75, 55])
    distr["ey0"] = (
        distr["x"] * (u0s1 @ np.array([25, 60, 45, 30]))
        + u0s2 @ np.array([65, 70, 40])
        + 20 / 3 * (1 - pz**3)
    )
    n = 4200
    distr["multiplier"] = n * distr["f"]
    distr["controls"] = distr["multiplier"] * (1 - distr["p"])
    full = distr.loc[distr.index.repeat(distr["multiplier"].round().astype(int))].reset_index(
        drop=True
    )
    count = full.groupby("group", sort=False).cumcount() + 1
    full["d"] = np.where(count.round() <= full["controls"].round(), 0, 1)
    full["ey"] = (1 - full["d"]) * full["ey0"] + full["d"] * full["ey1"]
    full = full[["x", "z", "f", "p", "ey1", "ey0", "d", "ey"]]
    return {"data_full": full, "data_dist": distr}
