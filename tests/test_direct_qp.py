"""The regression approach on the mosquito population, checked by hand.

This mirrors ``test_direct_qp.R`` of the R package: the point identified
case recovers the true MTR coefficients and ATE, and a misspecified
(collinear) specification is bounded by a quadratically constrained
program rebuilt with CVXPY on the same grid. The R test needs Gurobi; here
Clarabel solves both the estimator's problems and the hand-built ones.
"""

import cvxpy as cp
import numpy as np
import pytest

import pymte
from pymte.testdata import gendist_mosquito

COEF_M0 = [0.9, -1.1, 0.3]
COEF_M1 = [0.35, -0.3, -0.05]
TRUE_ATE = -0.8 / 3


def mono_integral(x, k):
    return x ** (k + 1) / (k + 1)


@pytest.fixture(scope="module")
def dtm():
    return gendist_mosquito()


@pytest.fixture(scope="module")
def point(dtm):
    with pytest.warns(UserWarning, match="point identified via linear regression"):
        return pymte.ivmte(
            data=dtm,
            propensity="d ~ 0 + C(z)",
            m0="~ 1 + u + I(u^2)",
            m1="~ 1 + u + I(u^2)",
            criterion_tol=0,
            target="ate",
            treat="d",
            outcome="ey",
        )


def test_point_identified_case(point, dtm):
    np.testing.assert_allclose(point.propensity.phat, dtm["pz"], atol=1e-6)
    np.testing.assert_allclose(point.mtr_coef.to_numpy(), COEF_M0 + COEF_M1, atol=1e-6)
    assert point.point_estimate == pytest.approx(TRUE_ATE, abs=1e-6)


@pytest.fixture(scope="module")
def design(dtm):
    """Design of the regression approach built by hand: IPW integrals of the monomials."""
    d = dtm["d"].to_numpy(dtype=float)
    p = dtm["pz"].to_numpy(dtype=float)
    b0 = [(mono_integral(1, k) - mono_integral(p, k)) / (1 - p) for k in range(3)]
    b1 = [(mono_integral(p, k) - mono_integral(0, k)) / p for k in range(3)]
    return np.column_stack([b * (1 - d) for b in b0] + [b * d for b in b1])


def test_mtr_component_construction(point, design, dtm):
    fit = np.linalg.lstsq(design, dtm["ey"].to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(point.mtr_coef.to_numpy(), fit, atol=1e-8)


def test_misspecified_bounds(dtm, design):
    tol = 0.5
    data = dtm.assign(x=1)
    result = pymte.ivmte(
        data=data,
        propensity="d ~ 0 + C(z)",
        m0="~ 1 + u + I(u^2)",
        m1="~ 1 + u + I(u^2) + x",
        point=True,
        criterion_tol=tol,
        target="ate",
        treat="d",
        outcome="ey",
        initgrid_nu=3,
        audit_nu=3,
        m0_inc=True,
        m1_inc=True,
        solver="clarabel",
    )
    y = data["ey"].to_numpy()
    mis_x = np.column_stack([design, data["d"].to_numpy(dtype=float)])
    u = np.arange(0, 1.01, 0.25)
    a0 = np.column_stack([np.ones(5), u, u**2])
    a1 = np.column_stack([a0, np.ones(5)])
    a = np.vstack([np.hstack([a0, np.zeros((5, 4))]), np.hstack([np.zeros((5, 3)), a1])])
    mono = np.vstack(
        [
            np.hstack([a0[1:] - a0[:-1], np.zeros((4, 4))]),
            np.hstack([np.zeros((4, 3)), a1[1:] - a1[:-1]]),
        ]
    )
    theta = cp.Variable(7)
    shape = [a @ theta >= y.min(), a @ theta <= y.max(), mono @ theta >= 0]
    ssr = cp.sum_squares(mis_x @ theta - y)
    crit = cp.Problem(cp.Minimize(ssr), shape)
    crit.solve(solver="CLARABEL")
    assert result.audit.criterion == pytest.approx(crit.value / len(y), rel=1e-6)
    tau = np.array([-1, -1 / 2, -1 / 3, 1, 1 / 2, 1 / 3, 1])
    np.testing.assert_allclose(result.gstar.to_numpy(), tau, atol=1e-12)
    bounds = []
    for sense in (cp.Minimize, cp.Maximize):
        prob = cp.Problem(sense(tau @ theta), [*shape, ssr <= crit.value * (1 + tol)])
        prob.solve(solver="CLARABEL")
        bounds.append(prob.value)
    assert result.bounds == pytest.approx(tuple(bounds), abs=1e-5)
