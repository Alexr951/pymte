"""Bootstrap inference: formulas checked on R's draws, distributions checked on our own."""

import numpy as np
import pytest

import pymte
from pymte.mst import _coef_ci, _point_ci, _point_pvalues, bound_ci, bound_pvalue

LEVELS = [0.9, 0.95, 0.99]

AE_ARGS = dict(
    target="att",
    m0="~ u + yob",
    m1="~ u + yob",
    ivlike="worked ~ morekids + samesex + morekids*samesex",
    propensity="morekids ~ samesex + yob",
)


def _matrix(entry):
    return np.asarray(entry["values"], dtype=float)


# -- formulas applied to R's own bootstrap draws must reproduce R's intervals --


@pytest.mark.parametrize(
    "case", ["ae_att_linear_u_boot50", "sim_late_boot50", "sim_late_boot50_m2000_subsample"]
)
def test_bound_confidence_regions_match_r_formulas(oracle, case):
    ref = oracle(case)
    draws = _matrix(ref["bounds_bootstraps"])
    n = {"ae": 209_133, "sim": 5_000}[case[:3] if case[:3] == "sim" else "ae"]
    m = 2000 if "m2000" in case else n
    bounds = tuple(ref["bounds"])
    for kind in ("backward", "forward"):
        ci = bound_ci(bounds, draws, n, m, LEVELS, kind)
        np.testing.assert_allclose(ci.to_numpy(), _matrix(ref["bounds_ci"][kind]), atol=1e-10)
        assert bound_pvalue(bounds, draws, n, m, kind) == pytest.approx(ref["p_value"][kind])
    np.testing.assert_allclose(
        np.std(draws, axis=0, ddof=1), list(ref["bounds_se"].values()), atol=1e-12
    )


@pytest.mark.parametrize("case", ["ae_att_point_gmm_boot50", "sim_spec_test_boot50"])
def test_point_intervals_match_r_formulas(oracle, case):
    ref = oracle(case)
    draws = np.asarray(ref["point_estimate_bootstraps"], dtype=float)
    est = ref["point_estimate"]
    ci = _point_ci(est, draws, LEVELS)
    for kind in ("nonparametric", "normal"):
        np.testing.assert_allclose(
            ci[kind].to_numpy(), _matrix(ref["point_estimate_ci"][kind]), atol=1e-10
        )
    p = _point_pvalues(est, draws)
    assert p["nonparametric"] == pytest.approx(ref["p_value"]["nonparametric"])
    assert p["parametric"] == pytest.approx(ref["p_value"]["parametric"], abs=1e-10)
    assert np.std(draws, ddof=1) == pytest.approx(ref["point_estimate_se"], abs=1e-12)
    if "j_test_bootstraps" in ref:
        j = np.asarray(ref["j_test_bootstraps"], dtype=float)
        j_stat = ref["j_test"]["J-statistic"]
        assert np.mean(j >= j_stat) == pytest.approx(ref["j_test"]["Bootstrapped p-value"])


def test_coef_ci_shape():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(40, 3))
    ci = _coef_ci(np.zeros(3), draws, ["a", "b", "c"], LEVELS)
    assert ci["normal"].shape == (3, 6) and list(ci["nonparametric"].index) == ["a", "b", "c"]
    assert (ci["normal"]["0.99 lower"] < ci["normal"]["0.9 lower"]).all()


# -- our own bootstrap: distributions comparable to R's draws --------------------


@pytest.mark.slow
def test_bounds_bootstrap_distribution_matches_r(oracle):
    ref = oracle("ae_att_linear_u_boot50")
    r_draws = _matrix(ref["bounds_bootstraps"])
    r = pymte.ivmte(pymte.load_ae(), bootstraps=50, seed=1, **AE_ARGS)
    assert r.bootstraps == 50 and r.bounds_bootstraps.shape == (50, 2)
    # Means agree within three standard errors of a 50-draw mean; spreads within a factor 1.5.
    se_mean = r_draws.std(axis=0, ddof=1) / np.sqrt(50)
    assert np.all(np.abs(r.bounds_bootstraps.mean(axis=0) - r_draws.mean(axis=0)) < 3 * se_mean)
    ratio = r.bounds_se / r_draws.std(axis=0, ddof=1)
    assert np.all((ratio > 1 / 1.5) & (ratio < 1.5))
    assert (
        set(r.bounds_ci) == {"backward", "forward"}
        and r.p_value["forward"] <= r.p_value["backward"]
    )
    assert r.propensity_se is not None and len(r.propensity_se) == 3
    assert "Bootstrapped confidence intervals (backward)" in r.summary()


def test_point_bootstrap_matches_r_distribution(oracle):
    ref = oracle("sim_spec_test_boot50")
    sim = pymte.load_sim_data()
    with pytest.warns(UserWarning, match="point identified"):
        r = pymte.ivmte(
            sim, ivlike="y ~ d + C(z)", target="ate", m0="~ u", m1="~ u", m0_dec=True,
            m1_dec=True, propensity="d ~ C(z)", bootstraps=50, seed=3,
        )  # fmt: skip
    assert r.point_estimate == pytest.approx(ref["point_estimate"], abs=1e-6)
    ratio = r.point_estimate_se / ref["point_estimate_se"]
    assert 1 / 1.6 < ratio < 1.6
    assert r.mtr_bootstraps.shape == (50, 4) and len(r.mtr_se) == 4
    assert r.j_test["df"] == 1 and 0 <= r.j_test["bootstrap_p_value"] <= 1
    assert r.p_value["nonparametric"] == 0.0
    assert "Bootstrapped J-test p-value" in r.summary()


def test_specification_test_runs_when_criterion_positive(oracle):
    ref = oracle("sim_lp_spec_test_boot50")
    sim = pymte.load_sim_data()
    r = pymte.ivmte(
        sim, ivlike="y ~ d + C(z)", target="ate", m0="~ u", m1="~ u", propensity="d ~ C(z)",
        point=False, bootstraps=50, seed=2,
    )  # fmt: skip
    assert r.criterion > 0 and r.specification_p_value is not None
    assert abs(r.specification_p_value - ref["specification_p_value"]) <= 0.2
    assert "Bootstrapped specification test p-value" in r.summary()


def test_subsampling_and_argument_checks():
    sim = pymte.load_sim_data()
    r = pymte.ivmte(
        sim, ivlike="y ~ d + z + d*z", target="late", late_from={"z": 1}, late_to={"z": 3},
        m0="~ u + I(u^2) + I(u^3) + x", m1="~ u + I(u^2) + I(u^3) + x", propensity="d ~ z + x",
        bootstraps=4, bootstraps_m=1000, bootstraps_replace=False, seed=0,
    )  # fmt: skip
    assert r.bounds_bootstraps.shape == (4, 2)
    with pytest.raises(ValueError, match="at least 2"):
        pymte.ivmte(sim, ivlike="y ~ d + C(z)", target="ate", m0="~ u", m1="~ u",
                    propensity="d ~ C(z)", bootstraps=1)  # fmt: skip
    with pytest.raises(ValueError, match="cannot exceed"):
        pymte.ivmte(sim, ivlike="y ~ d + C(z)", target="ate", m0="~ u", m1="~ u",
                    propensity="d ~ C(z)", point=True, bootstraps=2, bootstraps_m=6000,
                    bootstraps_replace=False)  # fmt: skip
