"""Parity of target and IV-like moments with the R package."""

import re

import numpy as np
import pandas as pd
import pytest

from pymte import load_ae, load_sim_data
from pymte.ivlike import iv_estimate
from pymte.mst import gen_s_set, gen_target
from pymte.mtr import polyparse
from pymte.propensity import propensity


def r_name(name: str) -> str:
    """Map an R coefficient name to the Python naming."""
    name = re.sub(r"^\[m[01]\]", "", name)
    name = name.replace("(Intercept)", "Intercept")
    name = re.sub(r"^u[01]S", "uS", name)
    name = re.sub(r"I\((\w+)\^(\d+)\)", r"I(\1 ** \2)", name)
    return name


def assert_named_close(py_names, py_values, r_dict, atol):
    r_map = {r_name(k): v for k, v in r_dict.items()}
    assert set(py_names) == set(r_map), (sorted(py_names), sorted(r_map))
    np.testing.assert_allclose(py_values, [r_map[n] for n in py_names], atol=atol)


AE_IVLIKE = ["worked ~ morekids + samesex + morekids*samesex"]
AE_PROP = "morekids ~ samesex + yob"
POLY0 = "~ u + I(u^2) + yob + u*yob"
POLY1 = "~ u + I(u^2) + I(u^3) + yob + u*yob"
CUBIC = "~ u + I(u^2) + I(u^3) + x"
SPL1 = "~ uSplines(degree = 1, knots = c(.25, .5, .75)) + x"
SPL3 = "~ uSplines(degree = 3, knots = c(.25, .5, .75)) + x"
MULTI = ["y ~ I(z == 1) + I(z == 2) + I(z == 3) + x", "y ~ d + x", "y ~ d | z"]

# name: (data, m0, m1, ivlike, propensity, target kwargs, components, subsets)
CASES = {
    "ae_att_linear_u": (
        "ae",
        "~ u + yob",
        "~ u + yob",
        AE_IVLIKE,
        AE_PROP,
        {"target": "att"},
        None,
        None,
    ),
    "ae_att_poly": ("ae", POLY0, POLY1, AE_IVLIKE, AE_PROP, {"target": "att"}, None, None),
    "ae_ate_poly": ("ae", POLY0, POLY1, AE_IVLIKE, AE_PROP, {"target": "ate"}, None, None),
    "ae_atu_poly": ("ae", POLY0, POLY1, AE_IVLIKE, AE_PROP, {"target": "atu"}, None, None),
    "ae_att_splines": (
        "ae",
        "~ u + uSplines(degree = 1, knots = c(.2, .4, .6, .8)) + yob",
        "~ uSplines(degree = 2, knots = c(.1, .3, .5, .7))*yob",
        AE_IVLIKE,
        AE_PROP,
        {"target": "att"},
        None,
        None,
    ),
    "sim_late": (
        "sim",
        CUBIC,
        CUBIC,
        ["y ~ d + z + d*z"],
        "d ~ z + x",
        {"target": "late", "late_from": {"z": 1}, "late_to": {"z": 3}},
        None,
        None,
    ),
    "sim_late_x2": (
        "sim",
        CUBIC,
        CUBIC,
        ["y ~ d + z + d*z"],
        "d ~ z + x",
        {"target": "late", "late_from": {"z": 1}, "late_to": {"z": 3}, "late_x": {"x": 2}},
        None,
        None,
    ),
    "sim_genlate_x2": (
        "sim",
        CUBIC,
        CUBIC,
        ["y ~ d + z + d*z"],
        "d ~ z + x",
        {"target": "genlate", "genlate_lb": 0.2, "genlate_ub": 0.42, "late_x": {"x": 2}},
        None,
        None,
    ),
    "sim_multi_ivlike": ("sim", SPL1, SPL1, MULTI, "d ~ z + x", {"target": "ate"}, None, None),
    "sim_multi_ivlike_components": (
        "sim",
        SPL1,
        SPL1,
        MULTI,
        "d ~ z + x",
        {"target": "ate"},
        [["intercept", "x"], ["d"], None],
        None,
    ),
    "sim_multi_ivlike_subset": (
        "sim",
        SPL3,
        SPL3,
        ["y ~ z + x", "y ~ d + x", "y ~ d | z"],
        "d ~ z + x",
        {"target": "ate"},
        None,
        ["x <= 9", None, "z in [1, 3]"],
    ),
    "sim_point_gmm": (
        "sim",
        "~ u",
        "~ u",
        ["y ~ d + C(z)"],
        "d ~ C(z)",
        {"target": "ate"},
        None,
        None,
    ),
}


@pytest.fixture(scope="module")
def datasets():
    return {"ae": load_ae(), "sim": load_sim_data()}


@pytest.mark.parametrize("case", list(CASES))
def test_target_and_ivlike_moments_match_r(oracle, datasets, case):
    data_key, m0, m1, ivlike, pform, tkw, comps, subs = CASES[case]
    data = datasets[data_key]
    prop = propensity(pform, data)
    spec0 = polyparse(m0, data)
    spec1 = polyparse(m1, data)
    target = gen_target(spec0, spec1, data, prop, **tkw)
    moments = gen_s_set(data, ivlike, spec0, spec1, prop, components=comps, subsets=subs)

    ref = oracle(case)
    assert_named_close(spec0.names, target.gstar0, ref["gstar"]["g0"], atol=1e-7)
    assert_named_close(spec1.names, target.gstar1, ref["gstar"]["g1"], atol=1e-7)
    r_sset = list(ref["s_set"].values())
    assert moments.n_moments == len(r_sset)
    for k, s in enumerate(r_sset):
        assert moments.beta[k] == pytest.approx(s["beta"], abs=1e-8)
        assert_named_close(spec0.names, moments.gamma0[k], s["g0"], atol=1e-7)
        assert_named_close(spec1.names, moments.gamma1[k], s["g1"], atol=1e-7)
    if ref.get("moments") is not None:
        assert moments.n_independent == ref["moments"]


def test_custom_weights_replicate_conditional_late(oracle, datasets):
    sim = datasets["sim"]
    prop = propensity("d ~ z + x", sim)
    spec = polyparse(CUBIC, sim)
    px = (sim["x"] == 2).mean()

    def p_at(x, z):
        return float(prop.predict(pd.DataFrame({"x": [x], "z": [z]}))[0])

    def weight1(x):
        return 0.0 if x != 2 else 1.0 / ((p_at(2, 3) - p_at(2, 1)) * px)

    def weight0(x):
        return -weight1(x)

    def knot1(x):
        return p_at(x, 1)

    def knot2(x):
        return p_at(x, 3)

    custom = gen_target(
        spec,
        spec,
        sim,
        prop,
        target_weight0=[0, weight0, 0],
        target_weight1=[0, weight1, 0],
        target_knots0=[knot1, knot2],
        target_knots1=[knot1, knot2],
    )
    ref = oracle("sim_custom_weights_late_x2")
    assert_named_close(spec.names, custom.gstar0, ref["gstar"]["g0"], atol=1e-7)
    assert_named_close(spec.names, custom.gstar1, ref["gstar"]["g1"], atol=1e-7)
    late = gen_target(
        spec, spec, sim, prop, "late", late_from={"z": 1}, late_to={"z": 3}, late_x={"x": 2}
    )
    np.testing.assert_allclose(custom.gstar0, late.gstar0, atol=1e-10)
    np.testing.assert_allclose(custom.gstar1, late.gstar1, atol=1e-10)


def test_ols_coefficients_equal_weighted_outcome_means(datasets):
    sim = datasets["sim"]
    fit = iv_estimate("y ~ d + x", sim, treat="d")
    s_actual = np.where(fit.d[:, None] == 1, fit.s1, fit.s0)
    np.testing.assert_allclose((s_actual * fit.y[:, None]).mean(axis=0), fit.beta, atol=1e-12)


def test_components_and_errors(datasets):
    sim = datasets["sim"]
    fit = iv_estimate("y ~ d + x", sim, treat="d", components=["intercept", "d"])
    assert fit.components == ("Intercept", "d")
    with pytest.raises(ValueError, match="not a coefficient"):
        iv_estimate("y ~ d + x", sim, treat="d", components=["w"])
    # A collinear regressor is dropped like lm.fit does, not rejected.
    fit = iv_estimate("y ~ d + x + x2", sim.assign(x2=sim["x"]), treat="d")
    assert fit.components == ("Intercept", "d", "x") and fit.s0.shape[1] == 3
    np.testing.assert_allclose(fit.beta, iv_estimate("y ~ d + x", sim, treat="d").beta)
    fit = iv_estimate("y ~ d + C(z) + d:C(z)", sim, treat="d", components=["C(z):d"])
    assert fit.components == ("d:C(z)[T.1]", "d:C(z)[T.2]", "d:C(z)[T.3]")
    with pytest.raises(ValueError, match="selects no observations"):
        iv_estimate("y ~ d", sim, treat="d", subset="x > 100")
