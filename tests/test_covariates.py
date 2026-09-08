"""Estimation with covariates and several IV-like specifications, checked by hand.

This mirrors ``test_covariates.R`` of the R package: OLS, TSLS and Wald
estimands, their Gamma moments, the generalised LATE and the bounds under
monotonicity restrictions are rebuilt from the population distribution of
:func:`pymte.testdata.gendist_covariates` and compared with the estimator.
"""

import numpy as np
import pytest
from conftest import grid_designs

import pymte
from pymte.testdata import gendist_covariates
from pymte.testfunctions_covariates import (
    gen_gamma_tt,
    popmean,
    s_ols1d,
    s_ols2d,
    s_ols3,
    s_tsls,
    s_wald,
    symat,
)

M0 = "~ x1 + x2:u + x2:I(u^2)"
M1 = "~ x1 + x1:x2 + u + x1:u + x2:I(u^2)"
IVLIKE = [
    "ey ~ d",
    "ey ~ d + x1",
    "ey ~ d + x1 + x2",
    "ey ~ d + x1 + x2 | x1 + x2 + z1 + z2",
    "ey ~ d | C(z2)",
]
COMPONENTS = [["d"], ["d"], ["d", "x1", "x2"], ["d"], ["d"]]
SUBSET = [None, None, "z2 in [2, 3]", None, "z2 in [2, 3]"]
VARS = ["ey", "eyd", "p", "x1", "x2", "z1", "z2"]
MEANS = VARS + [f"{a} * {b}" for a in VARS for b in VARS[2:]]


@pytest.fixture(scope="module")
def pop():
    return gendist_covariates()


@pytest.fixture(scope="module")
def result(pop):
    dtcf = pop["data_full"]
    support = dtcf[["x1", "x2"]].drop_duplicates().reset_index(drop=True)
    return pymte.ivmte(
        data=dtcf,
        ivlike=IVLIKE,
        components=COMPONENTS,
        subset=SUBSET,
        propensity="p",
        treat="d",
        m0=M0,
        m1=M1,
        target="genlate",
        genlate_lb=0.2,
        genlate_ub=0.7,
        criterion_tol=0.01,
        initgrid_nu=1,
        audit_nu=5,
        initgrid_x=support.iloc[:2],
        audit_x=support.iloc[:3],
        m0_inc=True,
        m1_inc=True,
        mte_dec=True,
        solver="highs",
    )


@pytest.fixture(scope="module")
def hand(pop):
    """Population IV-like estimands, weights and Gamma moments."""
    dtc = pop["data_dist"].copy()
    dtc["ey"] = dtc["ey1"] * dtc["p"] + dtc["ey0"] * (1 - dtc["p"])
    dtc["eyd"] = dtc["ey1"] * dtc["p"]
    m1 = popmean(MEANS, dtc)
    sub = dtc[dtc["z2"].isin([2, 3])]
    m2 = popmean(MEANS, sub)

    def exx_of(m):
        return symat(
            [1, m["p"], m["x1"], m["x2"], m["p"], m["p * x1"], m["p * x2"], m["x1 * x1"],
             m["x1 * x2"], m["x2 * x2"]]
        )  # fmt: skip

    def exy_of(m):
        return np.array([m["ey"], m["eyd"], m["ey * x1"], m["ey * x2"]])

    ols1_exx, ols1_exy = exx_of(m1), exy_of(m1)
    ols1 = np.linalg.solve(ols1_exx[:2, :2], ols1_exy[:2])[1]
    ols2 = np.linalg.solve(ols1_exx[:3, :3], ols1_exy[:3])[1]
    ols2_exx, ols2_exy = exx_of(m2), exy_of(m2)
    ols3 = np.linalg.solve(ols2_exx, ols2_exy)[1:4]
    # TSLS with Z = (1, z1, z2, x1, x2) and X = (1, d, x1, x2).
    zv = ["z1", "z2", "x1", "x2"]
    exz = np.column_stack(
        [[1, m1["p"], m1["x1"], m1["x2"]]]
        + [[m1[z], m1[f"p * {z}"], m1[f"x1 * {z}"], m1[f"x2 * {z}"]] for z in zv]
    )
    ezz = symat(
        [1, m1["z1"], m1["z2"], m1["x1"], m1["x2"], m1["z1 * z1"], m1["z1 * z2"],
         m1["z1 * x1"], m1["z1 * x2"], m1["z2 * z2"], m1["z2 * x1"], m1["z2 * x2"],
         m1["x1 * x1"], m1["x1 * x2"], m1["x2 * x2"]]
    )  # fmt: skip
    pi = exz @ np.linalg.inv(ezz)
    ezy = np.array([m1["ey"], m1["ey * z1"], m1["ey * z2"], m1["ey * x1"], m1["ey * x2"]])
    tsls = np.linalg.solve(pi @ exz.T, pi @ ezy)[1]
    # Wald estimand for z2 moving from 2 to 3.
    ey_z2 = {z: popmean(["ey", "p"], dtc[dtc["z2"] == z]) for z in (2, 3)}
    wald = (ey_z2[3]["ey"] - ey_z2[2]["ey"]) / (ey_z2[3]["p"] - ey_z2[2]["p"])
    betas = np.array([ols1, ols2, *ols3, tsls, wald])

    # Weights and Gamma moments.
    x = dtc[["x1", "x2"]].to_numpy()
    z = dtc[zv].to_numpy()
    for d in (0, 1):
        dtc[f"s.ols1.{d}"] = s_ols1d(d, ols1_exx)
        dtc[f"s.ols2.{d}"] = [s_ols2d(v, d, ols1_exx) for v in dtc["x1"]]
        for j, name in ((1, "d"), (2, "x1"), (3, "x2")):
            dtc[f"s.ols3.{d}.{name}"] = [s_ols3(r, d, j, ols2_exx) for r in x]
        dtc[f"s.tsls.{d}"] = [s_tsls(r, 1, exz, pi) for r in z]
        p_to, p_from = (dtc.loc[dtc["z2"] == k, "f"].sum() for k in (3, 2))
        dtc[f"s.wald.{d}"] = [
            s_wald(v, p_to, p_from, ey_z2[3]["p"], ey_z2[2]["p"]) for v in dtc["z2"]
        ]
    gammas = [
        gen_gamma_tt(dtc, "s.ols1.0", "s.ols1.1"),
        gen_gamma_tt(dtc, "s.ols2.0", "s.ols2.1"),
        gen_gamma_tt(dtc[dtc["z2"].isin([2, 3])], "s.ols3.0.d", "s.ols3.1.d"),
        gen_gamma_tt(dtc[dtc["z2"].isin([2, 3])], "s.ols3.0.x1", "s.ols3.1.x1"),
        gen_gamma_tt(dtc[dtc["z2"].isin([2, 3])], "s.ols3.0.x2", "s.ols3.1.x2"),
        gen_gamma_tt(dtc, "s.tsls.0", "s.tsls.1"),
        gen_gamma_tt(dtc, "s.wald.0", "s.wald.1"),
    ]
    dtc["w.genlate.1"] = 1 / (0.7 - 0.2)
    dtc["w.genlate.0"] = -dtc["w.genlate.1"]
    gstar = gen_gamma_tt(dtc, "w.genlate.0", "w.genlate.1", lb=0.2, ub=0.7)
    return {"beta": betas, "gammas": gammas, "gstar": gstar}


def test_iv_like_estimates(result, hand):
    assert result.ivlike.n_moments == 7
    np.testing.assert_allclose(result.ivlike.beta, hand["beta"], atol=1e-8)


def test_gamma_moments(result, hand):
    g0, g1 = hand["gstar"]
    np.testing.assert_allclose(result.target_gammas.gstar0, g0, atol=1e-8)
    np.testing.assert_allclose(result.target_gammas.gstar1, g1, atol=1e-8)
    for k, (g0, g1) in enumerate(hand["gammas"]):
        np.testing.assert_allclose(result.ivlike.gamma0[k], g0, atol=1e-8)
        np.testing.assert_allclose(result.ivlike.gamma1[k], g1, atol=1e-8)


def test_lp_problem(result, hand, hand_lp, pop, oracle):
    gamma = np.array([np.concatenate(g) for g in hand["gammas"]])
    mono0, mono1, d0, d1 = grid_designs(result)
    y = pop["data_full"]["ey"]
    miny, maxy = float(y.min()), float(y.max())
    z0, z1 = np.zeros_like(mono0), np.zeros_like(mono1)
    m0b, m1b, mteb = np.hstack([mono0, z1]), np.hstack([z0, mono1]), np.hstack([-mono0, mono1])
    rows = [
        (m0b, ">=", miny), (m1b, ">=", miny), (mteb, ">=", miny - maxy),
        (m0b, "<=", maxy), (m1b, "<=", maxy), (mteb, "<=", maxy - miny),
        (np.hstack([d0, np.zeros_like(d1)]), ">=", 0.0),
        (np.hstack([np.zeros_like(d0), d1]), ">=", 0.0),
        (np.hstack([-d0, d1]), "<=", 0.0),
    ]  # fmt: skip
    gstar = np.concatenate(hand["gstar"])
    crit, lower, upper = hand_lp(gamma, hand["beta"], rows, gstar, 0.01)
    assert result.audit.criterion == pytest.approx(crit, abs=1e-8)
    assert result.bounds == pytest.approx((lower, upper), abs=1e-6)
    assert result.bounds == pytest.approx(tuple(oracle("tt_covariates")["bounds"]), abs=1e-6)
