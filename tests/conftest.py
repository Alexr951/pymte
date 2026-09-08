import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

FIXTURES = Path(__file__).parent / "fixtures" / "oracle"


def load_oracle(name: str) -> dict:
    """Load a JSON fixture produced by tests/fixtures/run_oracle.R."""
    with (FIXTURES / f"{name}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def oracle():
    return load_oracle


def hand_bounds(gamma, beta, rows, gstar, criterion_tol):
    """Solve the moment-approach LPs built by hand, as the R test suite does.

    ``gamma`` (S x J) and ``beta`` (S) are the IV-like moments, ``rows`` a list
    of ``(matrix, sense, rhs)`` shape restrictions on the J coefficients with
    ``sense`` in {"<=", ">="}, ``gstar`` (J) the target. The variables are the
    2S non-negative slacks followed by the J coefficients. Returns the minimum
    criterion and the bounds on the target, from scipy's HiGHS interface
    without going through the package.
    """
    s, j = gamma.shape
    a_eq = np.hstack([-np.eye(s), np.eye(s), gamma])
    blocks, rhs = [], []
    for mat, sense, b in rows:
        sign = 1.0 if sense == "<=" else -1.0
        blocks.append(sign * np.hstack([np.zeros((len(mat), 2 * s)), mat]))
        rhs.append(sign * np.broadcast_to(b, len(mat)))
    a_ub, b_ub = np.vstack(blocks), np.concatenate(rhs)
    bounds = [(0, None)] * (2 * s) + [(None, None)] * j
    ones = np.concatenate([np.ones(2 * s), np.zeros(j)])
    crit = linprog(ones, a_ub, b_ub, a_eq, beta, bounds, method="highs").fun
    a_top = np.vstack([ones, a_ub])
    b_top = np.concatenate([[(1 + criterion_tol) * crit], b_ub])
    c = np.concatenate([np.zeros(2 * s), gstar])
    lower = linprog(c, a_top, b_top, a_eq, beta, bounds, method="highs").fun
    upper = -linprog(-c, a_top, b_top, a_eq, beta, bounds, method="highs").fun
    return crit, lower, upper


@pytest.fixture(scope="session")
def hand_lp():
    return hand_bounds


def grid_designs(audit_grid, design0, design1):
    """MTR bases on R's audit grid and their differences in consecutive ``u``.

    ``audit_grid`` is the ``audit_grid`` entry of an oracle fixture (the
    covariate rows and ``u`` values R used); ``design0``/``design1`` map a
    frame with the covariates and ``u`` to the basis matrix of each arm.
    """
    x = pd.DataFrame(audit_grid["audit_x"])
    u = np.asarray(audit_grid["audit_u"], dtype=float)
    nx, nu = len(x), len(u)
    grid = x.iloc[np.repeat(np.arange(nx), nu)].reset_index(drop=True).assign(u=np.tile(u, nx))
    mono0, mono1 = design0(grid), design1(grid)
    hi = np.array([c * nu + k for c in range(nx) for k in range(1, nu)])
    return mono0, mono1, mono0[hi] - mono0[hi - 1], mono1[hi] - mono1[hi - 1]
