import numpy as np
import pytest

from ivmte.splines import USpline


def test_matches_splines2_basis_and_integral(oracle):
    for spec in oracle("splines"):
        sp = USpline(spec["degree"], np.atleast_1d(spec["knots"]), spec["intercept"])
        u = np.asarray(spec["u"])
        np.testing.assert_allclose(sp.basis(u), np.asarray(spec["basis"]), atol=1e-12)
        np.testing.assert_allclose(sp.integral(0.0, u), np.asarray(spec["integral"]), atol=1e-12)


def test_column_count_and_intercept_convention():
    assert USpline(1, [0.5], intercept=True).n_basis == 3
    assert USpline(0, [0.2, 0.5, 0.8], intercept=True).n_basis == 4
    assert USpline(2, [0.3, 0.6]).n_basis == 4
    assert USpline(2, [0.3, 0.6]).basis([0.1]).shape == (1, 4)


def test_boundary_and_outside_knots_are_dropped():
    assert USpline(1, [0.0, 0.4, 1.0, 1.5]).knots == (0.4,)


def test_basis_sums_to_one_with_intercept():
    sp = USpline(3, [0.25, 0.5, 0.75], intercept=True)
    u = np.linspace(0, 1, 11)
    np.testing.assert_allclose(sp.basis(u).sum(axis=1), 1.0)


def test_integral_is_difference_of_antiderivatives():
    sp = USpline(2, [0.3, 0.6])
    full = sp.integral(0.0, 1.0)
    parts = sp.integral(0.0, 0.45) + sp.integral(0.45, 1.0)
    np.testing.assert_allclose(full, parts, atol=1e-14)


def test_invalid_degree():
    with pytest.raises(ValueError):
        USpline(-1, [])
