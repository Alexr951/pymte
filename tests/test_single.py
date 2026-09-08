"""Estimation with a single IV-like specification, checked by hand.

This mirrors ``test_single.R`` of the R package: the OLS coefficients, the
Gamma moments and the bounds are rebuilt from the population distribution
of :func:`pymte.testdata.gendist_covariates` and compared with the estimator.
"""

import numpy as np
import pytest
from conftest import grid_designs

import pymte
from pymte.testdata import gendist_covariates
from pymte.testfunctions_covariates import gen_gamma_tt, popmean, s_ols3, symat

M0 = "~ x1 + x2:u + x2:I(u^2)"
M1 = "~ x1 + x1:x2 + u + x1:u + x2:I(u^2)"
PROPENSITY = "d ~ x1 + x2 + z1 + z2"
MEANS = [
    "ey", "eyd", "p", "x1", "x2", "p * x1", "p * x2", "x1 * x1", "x1 * x2", "x2 * x2",
    "ey * x1", "ey * x2",
]  # fmt: skip


def design0(g):
    return np.column_stack([np.ones(len(g)), g.x1, g.x2 * g.u, g.x2 * g.u**2])


def design1(g):
    return np.column_stack([np.ones(len(g)), g.x1, g.u, g.x1 * g.x2, g.x1 * g.u, g.x2 * g.u**2])


@pytest.fixture(scope="module")
def pop():
    return gendist_covariates()


@pytest.fixture(scope="module")
def result(pop):
    dtcf = pop["data_full"]
    support = dtcf[["x1", "x2"]].drop_duplicates().reset_index(drop=True)
    return pymte.ivmte(
        data=dtcf,
        ivlike="ey ~ 1 + d + x1 + x2",
        components=[["d", "x1"]],
        subset=["z2 in [2, 3]"],
        propensity=PROPENSITY,
        link="logit",
        m0=M0,
        m1=M1,
        target="late",
        late_from={"z1": 1, "z2": 2},
        late_to={"z1": 0, "z2": 3},
        late_x={"x1": 0, "x2": 1},
        criterion_tol=0.01,
        initgrid_nu=4,
        audit_nu=5,
        initgrid_x=support.iloc[:2],
        audit_x=support.iloc[:5],
        solver="highs",
    )


@pytest.fixture(scope="module")
def hand(pop):
    """Population OLS coefficients, weights and Gamma moments."""
    dtc = pop["data_dist"].copy()
    dtc["ey"] = dtc["ey1"] * dtc["p"] + dtc["ey0"] * (1 - dtc["p"])
    dtc["eyd"] = dtc["ey1"] * dtc["p"]
    sub = dtc[dtc["z2"].isin([2, 3])]
    # OLS with controls on the subset: E[XX'] for X = (1, d, x1, x2).
    m = popmean(MEANS, sub)
    exx = symat(
        [1, m["p"], m["x1"], m["x2"], m["p"], m["p * x1"], m["p * x2"], m["x1 * x1"],
         m["x1 * x2"], m["x2 * x2"]]
    )  # fmt: skip
    ols = np.linalg.solve(exx, [m["ey"], m["eyd"], m["ey * x1"], m["ey * x2"]])
    # Gamma terms use the fitted propensity score, as in the R test.
    prop = pymte.propensity(PROPENSITY, pop["data_full"])
    dtc["p"] = prop.predict(dtc)
    x = dtc[["x1", "x2"]].to_numpy()
    for j, name in ((1, "d"), (2, "x1")):
        for d in (0, 1):
            dtc[f"s.ols.{d}.{name}"] = [s_ols3(r, d, j, exx) for r in x]
    sub = dtc[dtc["z2"].isin([2, 3])]
    g_ols = [gen_gamma_tt(sub, f"s.ols.0.{n}", f"s.ols.1.{n}") for n in ("d", "x1")]
    # LATE for fixed covariates.
    cell = (dtc["x1"] == 0) & (dtc["x2"] == 1)
    late_ub = float(dtc.loc[cell & (dtc["z1"] == 0) & (dtc["z2"] == 3), "p"].iloc[0])
    late_lb = float(dtc.loc[cell & (dtc["z1"] == 1) & (dtc["z2"] == 2), "p"].iloc[0])
    dtc["w.late.1"] = 1 / (late_ub - late_lb)
    dtc["w.late.0"] = -dtc["w.late.1"]
    gstar = gen_gamma_tt(dtc[cell], "w.late.0", "w.late.1", lb=late_lb, ub=late_ub)
    return {"ols": ols, "g_ols": g_ols, "gstar": gstar}


def test_iv_like_estimates(result, hand):
    np.testing.assert_allclose(result.ivlike.beta, hand["ols"][1:3], atol=1e-8)


def test_gamma_moments(result, hand):
    g0, g1 = hand["gstar"]
    np.testing.assert_allclose(result.target_gammas.gstar0, g0, atol=1e-8)
    np.testing.assert_allclose(result.target_gammas.gstar1, g1, atol=1e-8)
    for k, (g0, g1) in enumerate(hand["g_ols"]):
        np.testing.assert_allclose(result.ivlike.gamma0[k], g0, atol=1e-8)
        np.testing.assert_allclose(result.ivlike.gamma1[k], g1, atol=1e-8)


def test_lp_problem(result, hand, hand_lp, pop, oracle):
    gamma = np.array([np.concatenate(g) for g in hand["g_ols"]])
    beta = hand["ols"][1:3]
    mono0, mono1, _, _ = grid_designs(oracle("tt_single")["audit_grid"], design0, design1)
    y = pop["data_full"]["ey"]
    miny, maxy = float(y.min()), float(y.max())
    z0, z1 = np.zeros_like(mono0), np.zeros_like(mono1)
    m0b = np.hstack([mono0, z1])
    m1b = np.hstack([z0, mono1])
    mteb = np.hstack([-mono0, mono1])
    rows = [
        (m0b, ">=", miny), (m1b, ">=", miny), (mteb, ">=", miny - maxy),
        (m0b, "<=", maxy), (m1b, "<=", maxy), (mteb, "<=", maxy - miny),
    ]  # fmt: skip
    gstar = np.concatenate(hand["gstar"])
    crit, lower, upper = hand_lp(gamma, beta, rows, gstar, 0.01)
    assert result.audit.criterion == pytest.approx(crit, abs=1e-8)
    assert result.bounds == pytest.approx((lower, upper), abs=1e-6)
    assert result.bounds == pytest.approx(tuple(oracle("tt_single")["bounds"]), abs=1e-6)
