import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from pymte import load_ae
from pymte.mtr import MTRSpec


@pytest.fixture(scope="module")
def ae():
    return load_ae().head(500)


@pytest.mark.parametrize(
    ("formula", "names", "exponents"),
    [
        ("~ u + yob", ("Intercept", "u", "yob"), (0, 1, 0)),
        (
            "~ u + I(u^2) + I(u^3) + yob + u*yob",
            ("Intercept", "u", "I(u ** 2)", "I(u ** 3)", "yob", "u:yob"),
            (0, 1, 2, 3, 0, 1),
        ),
        (
            "~ u + I(yob == 55) + I(yob == 60)",
            ("Intercept", "u", "I(yob == 55)", "I(yob == 60)"),
            (0, 1, 0, 0),
        ),
        ("0 + yob:u + yob:I(u**2)", ("yob:u", "yob:I(u ** 2)"), (1, 2)),
    ],
)
def test_polynomial_terms(ae, formula, names, exponents):
    spec = MTRSpec.from_formula(formula, ae)
    assert spec.names == names
    assert spec.exponents == exponents
    assert not spec.splines


def test_spline_terms_and_names(ae):
    spec = MTRSpec.from_formula("~ uSplines(degree = 2, knots = c(.1, .3, .5, .7))*yob", ae)
    assert spec.poly_names == ("Intercept", "yob")
    assert len(spec.splines) == 1
    block = spec.splines[0]
    assert block.spline.degree == 2 and block.spline.knots == (0.1, 0.3, 0.5, 0.7)
    assert block.names == ("1", "yob")
    assert spec.names[2:8] == tuple(f"uS1.{b}:1" for b in range(1, 7))
    assert spec.names[8:] == tuple(f"uS1.{b}:yob" for b in range(1, 7))
    assert spec.n_coef == 2 + 12


def test_two_splines_indexed_in_term_order():
    data = pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
    spec = MTRSpec.from_formula(
        "~ 0 + x:uSplines(degree=0, knots=c(0.2, 0.5, 0.8), intercept=True)"
        " + uSplines(degree=1, knots=c(0.4), intercept=True) + I(u^2)",
        data,
    )
    # Main effects come before interactions, as in R's terms().
    assert spec.names == (
        "I(u ** 2)",
        "uS1.1:1",
        "uS1.2:1",
        "uS1.3:1",
        "uS2.1:x",
        "uS2.2:x",
        "uS2.3:x",
        "uS2.4:x",
    )


def test_uname(ae):
    spec = MTRSpec.from_formula("~ v + I(v^2) + yob + v*yob", ae, uname="v")
    assert spec.exponents == (0, 1, 2, 0, 1)
    assert "v" in spec.names and "u" not in spec.covariates


@pytest.mark.parametrize("formula", ["~ log(u) + yob", "~ I((yob*u)^2)", "~ u:I(u^2)", "~ exp(u)"])
def test_non_monomial_u_is_rejected(ae, formula):
    with pytest.raises(ValueError, match="must enter as a monomial"):
        MTRSpec.from_formula(formula, ae)


def test_gamma_matches_numerical_integration(ae):
    data = ae.head(5)
    spec = MTRSpec.from_formula(
        "~ u + I(u^2) + yob + uSplines(degree=2, knots=c(.3, .6))*yob", data
    )
    rng = np.random.default_rng(0)
    lb, ub, w = rng.uniform(0, 0.5, 5), rng.uniform(0.5, 1, 5), rng.normal(size=5)
    gamma = spec.gamma(data, lb, ub, w)
    for i in range(5):
        row = data.iloc[[i]]
        for j in range(spec.n_coef):
            val, _ = quad(lambda t, j=j, row=row: spec.design(row, t)[0, j], lb[i], ub[i])
            assert gamma[i, j] == pytest.approx(val * w[i], abs=1e-6)


def test_gamma_rows_restricts_and_design_evaluates(ae):
    data = ae.head(6)
    spec = MTRSpec.from_formula("~ u + yob", data)
    rows = np.array([True, False, True, False, True, False])
    g = spec.gamma(data, 0.0, 0.5, 2.0, rows=rows)
    assert g.shape == (3, 3)
    np.testing.assert_allclose(g[:, 1], 2.0 * 0.5**2 / 2)
    d = spec.design(data, np.full(6, 0.25))
    np.testing.assert_allclose(d[:, 1], 0.25)
    np.testing.assert_allclose(d[:, 2], data["yob"].to_numpy())


def test_from_columns_matches_formula(ae):
    from pymte.splines import USpline

    data = ae.head(50)
    by_formula = MTRSpec.from_formula(
        "~ u + I(u^2) + yob + u:yob + uSplines(degree=2, knots=c(.3, .6)):yob", data
    )
    by_columns = MTRSpec.from_columns(
        [(0, None), (1, None), (2, None), (0, "yob"), (1, "yob"), (USpline(2, [0.3, 0.6]), "yob")],
        data,
    )
    assert by_columns.names == by_formula.names
    assert by_columns.covariates == ("yob",)
    u = np.linspace(0, 1, 7)
    np.testing.assert_allclose(
        by_columns.design(data.head(7), u), by_formula.design(data.head(7), u)
    )
    np.testing.assert_allclose(
        by_columns.gamma(data, 0.1, 0.8, 1.0), by_formula.gamma(data, 0.1, 0.8, 1.0)
    )
    with pytest.raises(ValueError, match="not found"):
        MTRSpec.from_columns([(1, "nope")], data)
