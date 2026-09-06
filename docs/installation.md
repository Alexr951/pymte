# Installation

```bash
pip install pymte
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
pip install "pymte[gurobi]"
pip install "pymte[mosek]"
```

## Optional extras

```bash
pip install "pymte[plots]"   # matplotlib, for plot_mtr, plot_mte, plot_weights
pip install "pymte[docs]"    # build the documentation locally
pip install "pymte[dev]"     # pytest, ruff, mypy
```
