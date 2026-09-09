"""Argument checks of :func:`pymte.ivmte` that mirror the R package's ``callcheck``."""

import pytest

import pymte

SIM = dict(target="ate", m0="~ u", m1="~ u", ivlike="y ~ d + z", propensity="d ~ z")


@pytest.fixture(scope="module")
def sim():
    return pymte.load_sim_data()


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
