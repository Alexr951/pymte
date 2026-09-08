"""Estimation with spline MTRs, checked by hand.

This mirrors ``test_splines.R`` of the R package: the IV-like estimands,
the Gamma moments of a specification with three spline bases, the ATT
bounds and a custom target defined by weight and knot functions are
rebuilt from the population distribution of
:func:`pymte.testdata.gendist_splines` and compared with the estimator.
"""

import numpy as np
import pytest

import pymte
from pymte.testdata import gendist_splines
from pymte.testfunctions_covariates import popmean, symat
from pymte.testfunctions_splines import (
    gen_gamma_splines_tt,
    s_ols_splines,
    s_tsls_splines,
    spline_int,
    w_att_splines,
)

M1 = "~ x + uSplines(degree=2, knots=c(0.3, 0.6), intercept=False)"
M0 = (
    "~ 0 + x:uSplines(degree=0, knots=c(0.2, 0.5, 0.8), intercept=True)"
    " + uSplines(degree=1, knots=c(0.4), intercept=True) + I(u^2)"
)
IVLIKE = ["ey ~ d", "ey ~ d + x", "ey ~ d + x | z + x"]
COMPONENTS = [["intercept", "d"], ["d"], ["d", "x"]]
VARS = ["ey", "eyd", "p", "x", "z"]
MEANS = VARS + [f"{a} * {b}" for a in VARS for b in VARS[2:]]
SHAPE = {"m1_ub": 55, "m0_lb": 0, "mte_inc": True}


@pytest.fixture(scope="module")
def pop():
    return gendist_splines()


@pytest.fixture(scope="module")
def support(pop):
    return pop["data_full"][["x"]].drop_duplicates().reset_index(drop=True)


@pytest.fixture(scope="module")
def result(pop, support):
    return pymte.ivmte(
        data=pop["data_full"],
        ivlike=IVLIKE,
        components=COMPONENTS,
        propensity="p",
        treat="d",
        m1=M1,
        m0=M0,
        target="att",
        criterion_tol=0.01,
        initgrid_nu=25,
        audit_nu=25,
        initgrid_x=support,
        audit_x=support,
        solver="highs",
        **SHAPE,
    )


@pytest.fixture(scope="module")
def hand(pop):
    """Population IV-like estimands and Gamma moments."""
    dts = pop["data_dist"].copy()
    dts["ey"] = dts["ey1"] * dts["p"] + dts["ey0"] * (1 - dts["p"])
    dts["eyd"] = dts["ey1"] * dts["p"]
    m = popmean(MEANS, dts)
    exx = symat([1, m["p"], m["x"], m["p"], m["p * x"], m["x * x"]])
    exy = np.array([m["ey"], m["eyd"], m["ey * x"]])
    ols1 = np.linalg.solve(exx[:2, :2], exy[:2])
    ols2 = np.linalg.solve(exx, exy)
    exz = np.column_stack(
        [[1, m["p"], m["x"]], [m["z"], m["z * p"], m["z * x"]], [m["x"], m["x * p"], m["x * x"]]]
    )
    ezz = symat([1, m["z"], m["x"], m["z * z"], m["z * x"], m["x * x"]])
    pi = exz @ np.linalg.inv(ezz)
    ezy = np.array([m["ey"], m["ey * z"], m["ey * x"]])
    tsls = np.linalg.solve(pi @ exz.T, pi @ ezy)
    beta = np.array([ols1[0], ols1[1], ols2[1], tsls[1], tsls[2]])

    p = dts["p"].to_numpy()
    t1s1 = spline_int(p, 0.0, (0.3, 0.6), 2, False)
    t0s1 = spline_int(p, 0.0, (0.2, 0.5, 0.8), 0, True)
    t0s2 = spline_int(p, 0.0, (0.4,), 1, True)
    u1s1 = t1s1
    u0s1 = spline_int(1.0, p, (0.2, 0.5, 0.8), 0, True)
    u0s2 = spline_int(1.0, p, (0.4,), 1, True)
    g0, g1 = gen_gamma_splines_tt(dts, w_att_splines, t1s1, t0s1, t0s2, target=True, ed=m["p"])
    gstar = (-g0, g1)
    common = dict(u1s1=u1s1, u0s1=u0s1, u0s2=u0s2)
    gammas = [
        gen_gamma_splines_tt(dts, s_ols_splines, j=0, exx=exx[:2, :2], **common),
        gen_gamma_splines_tt(dts, s_ols_splines, j=1, exx=exx[:2, :2], **common),
        gen_gamma_splines_tt(dts, s_ols_splines, zvars=["x"], j=1, exx=exx, **common),
        gen_gamma_splines_tt(dts, s_tsls_splines, zvars=["z", "x"], j=1, exz=exz, pi=pi, **common),
        gen_gamma_splines_tt(dts, s_tsls_splines, zvars=["z", "x"], j=2, exz=exz, pi=pi, **common),
    ]
    return {"beta": beta, "gstar": gstar, "gammas": gammas}


def test_iv_like_estimates(result, hand):
    np.testing.assert_allclose(result.ivlike.beta, hand["beta"], atol=1e-8)


def test_gamma_moments(result, hand):
    spec0, spec1 = result.specs
    g0, g1 = hand["gstar"]
    assert list(g0.index) == [n.replace("uS", "u0S").replace(" ** ", "^") for n in spec0.names]
    assert list(g1.index) == [
        n.replace("uS", "u1S").replace("Intercept", "(Intercept)") for n in spec1.names
    ]
    np.testing.assert_allclose(result.target_gammas.gstar0, g0, atol=1e-8)
    np.testing.assert_allclose(result.target_gammas.gstar1, g1, atol=1e-8)
    for k, (g0, g1) in enumerate(hand["gammas"]):
        np.testing.assert_allclose(result.ivlike.gamma0[k], g0, atol=1e-8)
        np.testing.assert_allclose(result.ivlike.gamma1[k], g1, atol=1e-8)


def test_lp_problem(result, hand, hand_lp, pop, oracle):
    gamma = np.array([np.concatenate(g) for g in hand["gammas"]])
    grids = result.audit.grids
    spec0, spec1 = result.specs
    nx, nu = len(grids.support), len(grids.audit_u)
    grid_x = grids.support.iloc[np.repeat(np.arange(nx), nu)].reset_index(drop=True)
    grid_u = np.tile(grids.audit_u, nx)
    amono0 = spec0.design(grid_x, grid_u)
    amono1 = spec1.design(grid_x, grid_u)
    hi = np.array([c * nu + k for c in range(nx) for k in range(1, nu)])
    d0, d1 = amono0[hi] - amono0[hi - 1], amono1[hi] - amono1[hi - 1]
    dts = pop["data_dist"]
    maxy = float(max(dts["ey0"].max(), dts["ey1"].max()))
    miny = float(min(dts["ey0"].min(), dts["ey1"].min()))
    z0, z1 = np.zeros_like(amono0), np.zeros_like(amono1)
    m0b = np.hstack([amono0, z1])
    m1b = np.hstack([z0, amono1])
    mteb = np.hstack([-amono0, amono1])
    rows = [
        (m0b, ">=", 0.0), (m1b, ">=", miny), (mteb, ">=", miny - maxy),
        (m0b, "<=", maxy), (m1b, "<=", 55.0), (mteb, "<=", 55.0 - 0.0),
        (np.hstack([-d0, d1]), ">=", 0.0),
    ]  # fmt: skip
    gstar = np.concatenate(hand["gstar"])
    crit, lower, upper = hand_lp(gamma, hand["beta"], rows, gstar, 0.01)
    assert result.audit.criterion == pytest.approx(crit, abs=1e-8)
    assert result.bounds == pytest.approx((lower, upper), abs=1e-6)
    assert result.bounds == pytest.approx(tuple(oracle("tt_splines")["bounds"]), abs=1e-6)


def weight11(z):
    return 1 / (1 + z)


def weight01(z, x):
    return -1 / (1 + z + abs(x))


def weight02(z):
    return -(1 / (1 + z)) * 1.33


def knots01(z, x):
    return 0.3 + 0.3 * z + 0.1 * x


def test_custom_weights(pop, support, oracle):
    result = pymte.ivmte(
        data=pop["data_full"],
        ivlike=IVLIKE,
        components=COMPONENTS,
        propensity="p",
        treat="d",
        m1=M1,
        m0=M0,
        target_weight0=[weight01, weight02],
        target_weight1=[weight11],
        target_knots0=[knots01],
        criterion_tol=0.01,
        initgrid_x=support,
        audit_x=support,
        mte_ub=10,
        solver="highs",
        **SHAPE,
    )
    dts = pop["data_dist"].copy()
    f = dts["f"].to_numpy()
    w11 = weight11(dts["z"]).to_numpy()
    w01 = weight01(dts["z"], dts["x"]).to_numpy()
    w02 = weight02(dts["z"]).to_numpy()
    knot = knots01(dts["z"], dts["x"]).to_numpy()
    x = dts["x"].to_numpy()
    # Control arm: the u^2 term, then the two splines, each integrated over
    # [0, knot] with weight01 and over [knot, 1] with weight02.
    u2a, u2b = knot**3 / 3, 1 / 3 - knot**3 / 3
    ns0 = np.sum(u2a * w01 * f + u2b * w02 * f)
    s0a = spline_int(knot, 0.0, (0.2, 0.5, 0.8), 0, True)
    s0b = spline_int(1.0, knot, (0.2, 0.5, 0.8), 0, True)
    s01 = (s0a * (x * w01 * f)[:, None] + s0b * (x * w02 * f)[:, None]).sum(axis=0)
    s0c = spline_int(knot, 0.0, (0.4,), 1, True)
    s0d = spline_int(1.0, knot, (0.4,), 1, True)
    s02 = (s0c * (w01 * f)[:, None] + s0d * (w02 * f)[:, None]).sum(axis=0)
    gstar0 = np.concatenate([[ns0], s02, s01])
    # Treated arm: intercept, x and the spline over [0, 1] with weight11.
    ns1 = [np.sum(w11 * f), np.sum(x * w11 * f)]
    s11 = (spline_int(1.0, 0.0, (0.3, 0.6), 2, False) * (w11 * f)[:, None]).sum(axis=0)
    gstar1 = np.concatenate([ns1, s11])
    np.testing.assert_allclose(result.target_gammas.gstar0, gstar0, atol=1e-8)
    np.testing.assert_allclose(result.target_gammas.gstar1, gstar1, atol=1e-8)
    ref = oracle("tt_splines_custom_weights")
    assert result.bounds == pytest.approx(tuple(ref["bounds"]), abs=1e-6)
