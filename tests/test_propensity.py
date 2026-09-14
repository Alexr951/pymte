import numpy as np
import pytest

from pymte import load_ae, load_sim_data
from pymte.propensity import propensity

CASES = [
    ("ae_att_linear_u", "ae", "morekids ~ samesex + yob", "logit"),
    ("ae_att_linear_u_probit", "ae", "morekids ~ samesex + yob", "probit"),
    ("ae_att_linear_u_linear_link", "ae", "morekids ~ samesex + yob", "linear"),
    ("sim_point_gmm", "sim", "d ~ C(z)", "logit"),
    ("sim_late", "sim", "d ~ z + x", "logit"),
]


@pytest.fixture(scope="module")
def datasets():
    return {"ae": load_ae(), "sim": load_sim_data()}


@pytest.mark.parametrize(("case", "data", "formula", "link"), CASES)
def test_matches_r_glm(oracle, datasets, case, data, formula, link):
    ref = oracle(case)["propensity"]
    fit = propensity(formula, datasets[data], link)
    assert fit.treat == formula.split("~")[0].strip()
    np.testing.assert_allclose(fit.params, list(ref["coef"].values()), atol=1e-8)
    np.testing.assert_allclose(fit.phat[:50], ref["phat_head"], atol=1e-8)
    assert fit.phat.mean() == pytest.approx(ref["phat_mean"], abs=1e-9)


def test_predict_on_new_data(datasets):
    sim = datasets["sim"]
    fit = propensity("d ~ z + x", sim)
    new = sim.head(10)
    np.testing.assert_allclose(fit.predict(new), fit.phat[:10])


def test_late_weights_warn_when_a_linear_score_is_clipped(datasets):
    from pymte.wweights import wlate1

    sim = datasets["sim"]
    fit = propensity("d ~ z + x", sim, link="linear")
    with pytest.warns(UserWarning, match="greater than 1 set to 1"):
        w = wlate1(fit, sim, {"z": 3}, {"z": 5})
    assert w.ub.max() == 1.0


def test_late_weights_raise_when_clipping_empties_the_interval(datasets):
    from pymte.wweights import wlate1

    sim = datasets["sim"]
    fit = propensity("d ~ z + x", sim, link="linear")
    with (
        pytest.warns(UserWarning, match="greater than 1 set to 1"),
        pytest.raises(ValueError, match="LATE interval is empty"),
    ):
        wlate1(fit, sim, {"z": 5}, {"z": 10})


def test_supplied_column(datasets):
    sim = datasets["sim"].assign(p=0.5)
    prop = propensity("p", sim, treat="d")
    assert not prop.fitted
    np.testing.assert_array_equal(prop.phat, 0.5)
    with pytest.raises(ValueError):
        propensity("p", sim.assign(p=1.5), treat="d")


def test_bad_link_and_formula(datasets):
    with pytest.raises(ValueError):
        propensity("d ~ z", datasets["sim"], link="cloglog")
    with pytest.raises(ValueError):
        propensity("~ z", datasets["sim"])
