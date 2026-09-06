import numpy as np
import pytest

from ivmte import load_ae, load_sim_data
from ivmte.propensity import fit_propensity, propensity_from_column

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
    fit = fit_propensity(datasets[data], formula, link)
    assert fit.treat == formula.split("~")[0].strip()
    np.testing.assert_allclose(fit.params, list(ref["coef"].values()), atol=1e-8)
    np.testing.assert_allclose(fit.phat[:50], ref["phat_head"], atol=1e-8)
    assert fit.phat.mean() == pytest.approx(ref["phat_mean"], abs=1e-9)


def test_predict_on_new_data(datasets):
    sim = datasets["sim"]
    fit = fit_propensity(sim, "d ~ z + x")
    new = sim.head(10)
    np.testing.assert_allclose(fit.predict(new), fit.phat[:10])


def test_supplied_column(datasets):
    sim = datasets["sim"].assign(p=0.5)
    prop = propensity_from_column(sim, "p", treat="d")
    assert not prop.fitted
    np.testing.assert_array_equal(prop.phat, 0.5)
    with pytest.raises(ValueError):
        propensity_from_column(sim.assign(p=1.5), "p", treat="d")


def test_bad_link_and_formula(datasets):
    with pytest.raises(ValueError):
        fit_propensity(datasets["sim"], "d ~ z", link="cloglog")
    with pytest.raises(ValueError):
        fit_propensity(datasets["sim"], "~ z")
