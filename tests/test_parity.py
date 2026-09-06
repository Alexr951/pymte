"""Reproduction of the results published in the R package vignette.

The reference values come from the vignette of R ivmte (solved with Gurobi)
and from the R oracle run with lpSolveAPI shipped in ``tests/fixtures``.
Bounds and point estimates must agree to 1e-6 (LP, GMM, OLS) and 1e-5
(QCQP). Audit round counts are compared with the lpSolveAPI oracle; they
depend on which optimal vertex a solver returns and are documented to differ
from Gurobi in some cases.
"""

import numpy as np
import pytest

import ivmte

AE_IVLIKE = "worked ~ morekids + samesex + morekids*samesex"
AE_PROP = "morekids ~ samesex + yob"
POLY0 = "~ u + I(u^2) + yob + u*yob"
POLY1 = "~ u + I(u^2) + I(u^3) + yob + u*yob"
SPL0 = "~ u + uSplines(degree = 1, knots = c(.2, .4, .6, .8)) + yob"
SPL1 = "~ uSplines(degree = 2, knots = c(.1, .3, .5, .7))*yob"
CUBIC = "~ u + I(u^2) + I(u^3) + x"
SIM_SPL1 = "~ uSplines(degree = 1, knots = c(.25, .5, .75)) + x"
SIM_SPL3 = "~ uSplines(degree = 3, knots = c(.25, .5, .75)) + x"
MULTI = ["y ~ I(z == 1) + I(z == 2) + I(z == 3) + x", "y ~ d + x", "y ~ d | z"]


@pytest.fixture(scope="module")
def ae():
    return ivmte.load_ae()


@pytest.fixture(scope="module")
def sim():
    return ivmte.load_sim_data()


# (oracle case, published bounds, audit rounds in the vignette, kwargs)
BOUNDS = {
    "ae_att_linear_u": (
        (-0.1028836, -0.07818869), 1,
        dict(target="att", m0="~ u + yob", m1="~ u + yob", ivlike=AE_IVLIKE, propensity=AE_PROP),
    ),
    "ae_att_linear_u_probit": (
        (-0.100781, -0.0825274), 1,
        dict(target="att", m0="~ u + yob", m1="~ u + yob", ivlike=AE_IVLIKE, propensity=AE_PROP, link="probit"),
    ),
    "ae_att_poly": (
        (-0.2950822, 0.1254494), 3,
        dict(target="att", m0=POLY0, m1=POLY1, ivlike=AE_IVLIKE, propensity=AE_PROP),
    ),
    "ae_ate_poly": (
        (-0.375778, 0.1957841), 1,
        dict(target="ate", m0=POLY0, m1=POLY1, ivlike=AE_IVLIKE, propensity=AE_PROP),
    ),
    "ae_att_poly_uname_v": (
        (-0.2950822, 0.1254494), 3,
        dict(target="att", m0=POLY0.replace("u", "v"), m1=POLY1.replace("u", "v"), uname="v",
             ivlike=AE_IVLIKE, propensity=AE_PROP),
    ),
    "ae_att_boolean_terms": (
        (-0.1028836, -0.07818869), 1,
        dict(target="att", m0="~ u + yob", m1="~ u + I(yob == 55) + I(yob == 60)",
             ivlike=AE_IVLIKE, propensity=AE_PROP),
    ),
    "ae_att_splines": (
        (-0.4545814, 0.3117817), 2,
        dict(target="att", m0=SPL0, m1=SPL1, ivlike=AE_IVLIKE, propensity=AE_PROP),
    ),
    "ae_att_splines_shape": (
        (-0.09769381, 0.09247149), 1,
        dict(target="att", m0=SPL0, m1=SPL1, ivlike=AE_IVLIKE, propensity=AE_PROP,
             m1_inc=True, m0_inc=True, mte_dec=True),
    ),
    "sim_late_vignette": (
        # The vignette predates the redefinition of "late" in R (2022-07); its
        # value is what R and this package now call "avglate".
        (-0.6931532, -0.4397993), 2,
        dict(target="avglate", late_from={"z": 1}, late_to={"z": 3}, m0=CUBIC, m1=CUBIC,
             ivlike="y ~ d + z + d*z", propensity="d ~ z + x"),
    ),
    "sim_late": (
        # Current R definition of the LATE (lpSolveAPI oracle value).
        (-0.6931332, -0.4398240), 2,
        dict(target="late", late_from={"z": 1}, late_to={"z": 3}, m0=CUBIC, m1=CUBIC,
             ivlike="y ~ d + z + d*z", propensity="d ~ z + x"),
    ),
    "sim_late_x2": (
        (-0.8419396, -0.2913049), 2,
        dict(target="late", late_from={"z": 1}, late_to={"z": 3}, late_x={"x": 2}, m0=CUBIC,
             m1=CUBIC, ivlike="y ~ d + z + d*z", propensity="d ~ z + x"),
    ),
    "sim_late_x8": (
        (-0.7721625, -0.3209851), 2,
        dict(target="late", late_from={"z": 1}, late_to={"z": 3}, late_x={"x": 8}, m0=CUBIC,
             m1=CUBIC, ivlike="y ~ d + z + d*z", propensity="d ~ z + x"),
    ),
    "sim_genlate": (
        (-0.7504255, -0.3182317), 2,
        dict(target="genlate", genlate_lb=0.2, genlate_ub=0.42, m0=CUBIC, m1=CUBIC,
             ivlike="y ~ d + z + d*z", propensity="d ~ z + x"),
    ),
    "sim_genlate_x2": (
        (-0.867551, -0.2750135), 2,
        dict(target="genlate", genlate_lb=0.2, genlate_ub=0.42, late_x={"x": 2}, m0=CUBIC,
             m1=CUBIC, ivlike="y ~ d + z + d*z", propensity="d ~ z + x"),
    ),
    "sim_multi_ivlike": (
        (-0.6427017, -0.3727193), 1,
        dict(target="ate", m0=SIM_SPL1, m1=SIM_SPL1, ivlike=MULTI, propensity="d ~ z + x"),
    ),
    "sim_multi_ivlike_components": (
        (-0.6865291, -0.2573525), 1,
        dict(target="ate", m0=SIM_SPL1, m1=SIM_SPL1, ivlike=MULTI, propensity="d ~ z + x",
             components=[["intercept", "x"], ["d"], None]),
    ),
    "sim_multi_ivlike_subset": (
        (-0.6697228, -0.3331582), 2,
        dict(target="ate", m0=SIM_SPL3, m1=SIM_SPL3, ivlike=["y ~ z + x", "y ~ d + x", "y ~ d | z"],
             subset=["x <= 9", None, "z in [1, 3]"], propensity="d ~ z + x"),
    ),
    "sim_point_lp_tol0": (
        (-0.5349027, -0.5349027), 1,
        dict(target="ate", m0="~ u", m1="~ u", ivlike="y ~ d + C(z)", propensity="d ~ C(z)",
             point=False, criterion_tol=0),
    ),
    "ae_att_splines_noint_shape": (
        (-1.240035e-05, -2.231762e-17), 1,
        dict(target="att", m0="~ 0 + uSplines(degree = 2, knots = c(1/3, 2/3))",
             m1="~ 0 + uSplines(degree = 2, knots = c(1/3, 2/3))", m1_inc=True, m0_inc=True,
             mte_dec=True, ivlike=AE_IVLIKE, propensity="morekids ~ samesex"),
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(BOUNDS))
def test_published_bounds(oracle, ae, sim, case):
    published, _rounds, kwargs = BOUNDS[case]
    data = ae if case.startswith("ae") else sim
    r = ivmte.ivmte(data, seed=0, **kwargs)
    assert r.bounds is not None
    np.testing.assert_allclose(r.bounds, published, atol=1e-6)
    assert r.audit is not None
    try:
        ref = oracle(case)
    except FileNotFoundError:
        return
    np.testing.assert_allclose(r.bounds, ref["bounds"], atol=1e-6)
    assert r.audit.audit_count == ref["audit_count"]
    assert r.moments == ref["moments"]


def test_custom_weights_replicate_conditional_late(sim):
    prop = ivmte.fit_propensity(sim, "d ~ z + x")
    px = (sim["x"] == 2).mean()

    def p_at(x, z):
        import pandas as pd

        return float(prop.predict(pd.DataFrame({"x": [x], "z": [z]}))[0])

    def weight1(x):
        return 0.0 if x != 2 else 1.0 / ((p_at(2, 3) - p_at(2, 1)) * px)

    def weight0(x):
        return -weight1(x)

    def knot1(x):
        return p_at(x, 1)

    def knot2(x):
        return p_at(x, 3)

    r = ivmte.ivmte(
        sim,
        ivlike="y ~ d + z + d*z",
        target_knots0=[knot1, knot2],
        target_knots1=[knot1, knot2],
        target_weight0=[0, weight0, 0],
        target_weight1=[0, weight1, 0],
        m0=CUBIC,
        m1=CUBIC,
        propensity="d ~ z + x",
        seed=0,
    )
    np.testing.assert_allclose(r.bounds, (-0.8419396, -0.2913049), atol=1e-6)
    assert r.target == "custom"


# (oracle case, published estimate, kwargs)
POINTS = {
    "sim_point_gmm": (-0.5389508, dict(ivlike="y ~ d + C(z)", target="ate", m0="~ u", m1="~ u",
                                      propensity="d ~ C(z)", point=True)),
    "sim_point_gmm_eyeweight": (-0.5325135, dict(ivlike="y ~ d + C(z)", target="ate", m0="~ u",
                                                m1="~ u", propensity="d ~ C(z)", point=True,
                                                point_eyeweight=True)),
    "sim_point_gmm_many_moments": (-0.5559325, dict(ivlike="y ~ d + C(z) + d:C(z)", target="ate",
                                                   m0="~ u", m1="~ u", propensity="d ~ C(z)",
                                                   point=True)),
    "sim_shape_ignored_point": (-0.5389508, dict(ivlike="y ~ d + C(z)", target="ate", m0="~ u",
                                                m1="~ u", m0_dec=True, m1_dec=True,
                                                propensity="d ~ C(z)")),
    "sim_point_regression_ols": (-0.5530917, dict(outcome="y", target="ate", m0="~ u", m1="~ u",
                                                 propensity="d ~ C(z)", point=True)),
    "sim_point_regression_equal_coef": (-0.5547433, dict(outcome="y", target="ate", m0="~ u",
                                                        m1="~ u", propensity="d ~ C(z)",
                                                        equal_coef="~ 0 + u", point=True)),
    "ae_att_regression_point": (-0.08616707, dict(target="att", m0="~ u + yob", m1="~ u + yob",
                                                 outcome="worked", propensity=AE_PROP)),
    "ae_att_point_gmm": (-0.09160436, dict(target="att", m0="~ u", m1="~ u", ivlike=AE_IVLIKE,
                                          propensity="morekids ~ samesex", point=True)),
}  # fmt: skip


@pytest.mark.parametrize("case", list(POINTS))
def test_published_point_estimates(oracle, ae, sim, case):
    published, kwargs = POINTS[case]
    data = ae if case.startswith("ae") else sim
    with pytest.warns(UserWarning, match="point identified") if "point" not in kwargs else _noop():
        r = ivmte.ivmte(data, **kwargs)
    assert r.bounds is None
    assert r.point_estimate == pytest.approx(published, abs=1e-6)
    try:
        ref = oracle(case)
    except FileNotFoundError:
        return
    assert r.point_estimate == pytest.approx(ref["point_estimate"], abs=1e-8)
    if ref.get("mtr_coef"):
        assert r.mtr_coef is not None
        np.testing.assert_allclose(
            r.mtr_coef.to_numpy(), list(ref["mtr_coef"].values()), atol=1e-7
        )
    if ref.get("j_test"):
        assert r.j_test is not None
        j = list(ref["j_test"].values())
        assert r.j_test["stat"] == pytest.approx(j[0], rel=1e-6)
        assert r.j_test["p_value"] == pytest.approx(j[1], abs=1e-8)
        assert r.j_test["df"] == j[2]


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_equal_coef_regression_mtr_coefficients(sim):
    with pytest.warns(UserWarning, match="point identified"):
        r = ivmte.ivmte(
            sim, outcome="y", target="ate", m0="~ x + u", m1="~ x + u", equal_coef="~ 0 + x",
            propensity="d ~ x + C(z)",
        )  # fmt: skip
    assert r.mtr_coef is not None
    expected = {
        "[m0]Intercept": 0.675148700, "[m0]x": -0.003755079, "[m0]u": 0.137248852,
        "[m1]Intercept": 0.132391436, "[m1]x": -0.003755079, "[m1]u": 0.106476957,
    }  # fmt: skip
    for name, value in expected.items():
        assert r.mtr_coef[name] == pytest.approx(value, abs=1e-8)


def test_regression_qcqp_bounds_contain_least_squares_target(sim):
    with pytest.warns(UserWarning, match="point identified"):
        ols = ivmte.ivmte(
            sim, outcome="y", target="ate", m0="~ u", m1="~ u", propensity="d ~ C(z)"
        )
    r = ivmte.ivmte(
        sim, outcome="y", target="ate", m0="~ u", m1="~ u", propensity="d ~ C(z)", point=False,
        criterion_tol=1e-3,
    )  # fmt: skip
    assert r.method == "qcqp" and r.bounds is not None
    assert r.bounds[0] - 1e-6 <= ols.point_estimate <= r.bounds[1] + 1e-6
    tight = ivmte.ivmte(
        sim, outcome="y", target="ate", m0="~ u", m1="~ u", propensity="d ~ C(z)", point=False,
        criterion_tol=0,
    )  # fmt: skip
    # Gurobi reports [-0.553351, -0.5528199] here, a width driven by its
    # feasibility tolerance; the least-squares set itself is a point.
    np.testing.assert_allclose(tight.bounds, (-0.5530917, -0.5530917), atol=1e-5)
    assert tight.bounds[1] - tight.bounds[0] < r.bounds[1] - r.bounds[0]


def test_result_summary_and_dict(ae):
    r = ivmte.ivmte(
        ae, target="att", m0="~ u + yob", m1="~ u + yob", ivlike=AE_IVLIKE, propensity=AE_PROP
    )
    text = r.summary()
    assert "Bounds on the target parameter: [-0.1028836, -0.07818869]" in text
    assert "Audit terminated successfully after 1 round(s)" in text
    assert "Independent/total moments: 4/4" in text
    d = r.to_dict()
    assert d["audit_count"] == 1 and d["moments"] == 4
    assert any("Audit count: 1" in m for m in r.messages)
