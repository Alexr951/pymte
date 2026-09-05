# Installation

```bash
pip install ivmte
```

Python 3.10 or later is required. The package depends on NumPy, SciPy,
pandas, statsmodels, formulaic and CVXPY.

## Solvers

Partial identification requires an optimiser. Two open-source solvers are
installed automatically and used by default:

- **HiGHS** (via `scipy.optimize.linprog`) for the linear programs of the
  moment approach.
- **Clarabel** (via CVXPY) for the quadratically constrained programs of the
  regression approach.

Commercial solvers can be used instead when a licence is available. They are
selected automatically when importable, in the same order of preference as
the R package (Gurobi, then MOSEK), or explicitly with `solver="gurobi"` or
`solver="mosek"`:

```bash
pip install "ivmte[gurobi]"
pip install "ivmte[mosek]"
```

## Optional extras

```bash
pip install "ivmte[plots]"   # matplotlib, for plot_mtr, plot_mte, plot_weights
pip install "ivmte[docs]"    # build the documentation locally
pip install "ivmte[dev]"     # pytest, ruff, mypy
```
