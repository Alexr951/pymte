import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES

from pymte import load_ae, load_sim_data
from pymte.testdata import gendist_basic, gendist_covariates, gendist_mosquito, gendist_splines


def test_ae_shape_and_columns():
    ae = load_ae()
    assert ae.shape == (209_133, 8)
    assert list(ae.columns) == [
        "worked",
        "hours",
        "morekids",
        "samesex",
        "yob",
        "black",
        "hisp",
        "other",
    ]
    assert set(ae["morekids"].unique()) == {0, 1}


def test_sim_data_shape_and_support():
    sim = load_sim_data()
    assert sim.shape == (5_000, 4)
    assert list(sim.columns) == ["y", "d", "z", "x"]
    assert set(sim["z"].unique()) == {0, 1, 2, 3}
    assert sim["x"].between(1, 10).all()


@pytest.mark.parametrize(
    ("frame", "fixture"),
    [
        (lambda: gendist_basic()["data_dist"], "dist_basic_dist"),
        (lambda: gendist_basic()["data_full"], "dist_basic_full"),
        (lambda: gendist_covariates()["data_dist"], "dist_covariates_dist"),
        (lambda: gendist_covariates()["data_full"], "dist_covariates_full"),
        (lambda: gendist_splines()["data_dist"], "dist_splines_dist"),
        (lambda: gendist_splines()["data_full"], "dist_splines_full"),
        (gendist_mosquito, "dist_mosquito"),
    ],
)
def test_synthetic_populations_match_r(frame, fixture):
    """The test populations are the R package's, row for row."""
    py = frame()
    r = pd.read_csv(FIXTURES / f"{fixture}.csv")
    assert list(py.columns) == list(r.columns)
    assert py.shape == r.shape
    np.testing.assert_allclose(py.to_numpy(dtype=float), r.to_numpy(dtype=float), atol=1e-10)
