import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pymte.lp import LinearConstraints, run_lp

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
    ``sense`` in {"<=", ">="}, ``gstar`` (J) the target. Returns the minimum
    criterion and the bounds on the target.
    """
    s, j = gamma.shape
    n = 2 * s + j
    a_eq = sp.csr_matrix(np.hstack([-np.eye(s), np.eye(s), gamma]))
    blocks, rhs = [], []
    for mat, sense, b in rows:
        sign = 1.0 if sense == "<=" else -1.0
        blocks.append(sign * np.hstack([np.zeros((len(mat), 2 * s)), mat]))
        rhs.append(sign * np.broadcast_to(b, len(mat)))
    a_ub = sp.csr_matrix(np.vstack(blocks))
    b_ub = np.concatenate(rhs)
    lb = np.concatenate([np.zeros(2 * s), np.full(j, -np.inf)])
    model = LinearConstraints(n, a_ub, b_ub, a_eq, beta, lb, None)
    ones = np.concatenate([np.ones(2 * s), np.zeros(j)])
    crit = run_lp(ones, model, "min", "highs").obj
    a_top = sp.csr_matrix(sp.vstack([sp.csr_matrix(ones), a_ub]))
    b_top = np.concatenate([[(1 + criterion_tol) * crit], b_ub])
    bounded = LinearConstraints(n, a_top, b_top, a_eq, beta, lb, None)
    c = np.concatenate([np.zeros(2 * s), gstar])
    lower = run_lp(c, bounded, "min", "highs").obj
    upper = run_lp(c, bounded, "max", "highs").obj
    return crit, lower, upper


@pytest.fixture(scope="session")
def hand_lp():
    return hand_bounds
