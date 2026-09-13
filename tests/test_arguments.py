"""Argument checks of :func:`pymte.ivmte` that mirror the R package's ``callcheck``."""

import pytest

import pymte

SIM = dict(target="ate", m0="~ u", m1="~ u", ivlike="y ~ d + z", propensity="d ~ z")


@pytest.fixture(scope="module")
def sim():
    return pymte.load_sim_data()


def test_ivlike_formulas_must_share_the_outcome(sim):
    with pytest.raises(ValueError, match="Multiple response variables"):
        pymte.ivmte(sim, **{**SIM, "ivlike": ["y ~ d + z", "d ~ z"]})


def test_no_variable_may_be_called_intercept(sim):
    with pytest.raises(ValueError, match="named 'intercept'"):
        pymte.ivmte(sim.assign(intercept=1), **{**SIM, "ivlike": "y ~ d + z + intercept"})


def test_treatment_cannot_enter_the_mtrs(sim):
    with pytest.raises(ValueError, match="Treatment variable cannot be included in the MTRs"):
        pymte.ivmte(sim, **{**SIM, "m0": "~ u + d"})


def test_treat_must_match_the_propensity_formula(sim):
    with pytest.raises(ValueError, match="'treat' .* differs"):
        pymte.ivmte(sim, treat="z", **SIM)
    r = pymte.ivmte(sim, treat="d", **SIM)
    assert r.propensity.treat == "d"


def test_late_variables_must_be_in_the_propensity_model(sim):
    with pytest.raises(ValueError, match="must be included in the propensity score model"):
        pymte.ivmte(
            sim, target="late", late_from={"x": 1}, late_to={"x": 3}, m0="~ u", m1="~ u",
            ivlike="y ~ d + z", propensity="d ~ z",
        )  # fmt: skip


def test_warns_when_a_specification_without_treatment_loses_moments(sim):
    with pytest.warns(UserWarning, match="do not include the treatment variable: 1"):
        r = pymte.ivmte(
            sim, target="ate", m0="~ u + x", m1="~ u + x", ivlike=["y ~ z + x", "y ~ d | z"],
            propensity="d ~ z + x", seed=0,
        )  # fmt: skip
    assert r.moments < r.ivlike.n_moments


def test_summary_reports_an_unfinished_audit():
    ae = pymte.load_ae()
    with pytest.warns(UserWarning, match="audit_max"):
        r = pymte.ivmte(
            ae, target="att", ivlike="worked ~ morekids + samesex + morekids*samesex",
            propensity="morekids ~ samesex + yob",
            m0="~ u + uSplines(degree = 1, knots = c(.2, .4, .6, .8)) + yob",
            m1="~ uSplines(degree = 2, knots = c(.1, .3, .5, .7))*yob",
            m1_inc=True, m0_inc=True, mte_dec=True, initgrid_nu=2, initgrid_nx=2, audit_max=1,
            seed=0,
        )  # fmt: skip
    assert "Audit reached audit_max (1)" in r.summary()
    assert "successfully" not in r.summary()


def test_propensity_column_as_one_sided_formula(sim):
    p = pymte.propensity("d ~ z", sim)
    df = sim.assign(p=p.phat)
    a = pymte.ivmte(df, **{**SIM, "propensity": "p"}, treat="d", seed=0)
    b = pymte.ivmte(df, **{**SIM, "propensity": "~ p"}, treat="d", seed=0)
    assert a.bounds == b.bounds


def test_usplines_singular_alias(sim):
    a = pymte.polyparse("~ uSplines(degree=1, knots=[.5]) + x", sim)
    b = pymte.polyparse("~ uSpline(degree = 1, knots = c(.5)) + x", sim)
    assert a.names == b.names and a.splines[0].spline == b.splines[0].spline
