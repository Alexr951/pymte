---
file_format: mystnb
kernelspec:
  name: python3
---

# The audit procedure

Shape restrictions are linear inequalities that must hold at every point
$(u, x)$. Imposing them on a fine grid from the start makes the optimisation
problems large, so the package follows the R implementation:

1. impose the restrictions on a small *initial grid*;
2. solve for the minimum criterion and the bounds;
3. check the restrictions at both bounding solutions on a finer *audit
   grid*;
4. add the violated grid points to the constraint set and go back to step 2.

The loop stops when no violations remain (the usual case after one to three
rounds), after `audit_max` rounds, or when the same violations persist for
three rounds.

## Grid construction

The grid in $u$ is the first `initgrid_nu` (`audit_nu`) points of the
base-2 Halton sequence plus the end points 0 and 1, so it is deterministic
and nested. The grid in the covariates is a sample of `initgrid_nx`
(`audit_nx`) distinct covariate rows; when the data have fewer distinct
rows, as in the examples, the whole support is used and the procedure is
deterministic. Otherwise pass `seed` for reproducibility, or pass the grids
explicitly with `initgrid_x`, `initgrid_u`, `audit_x`, `audit_u`.

```{code-cell} python
from pymte.audit import rhalton

rhalton(5)
```

## Tuning

| argument | default | meaning |
|---|---|---|
| `initgrid_nu`, `initgrid_nx` | 20, 20 | size of the initial grid |
| `audit_nu`, `audit_nx` | 25, 2500 | size of the audit grid |
| `audit_add` | 100 | maximum number of violated points added per round |
| `audit_max` | 25 | maximum number of rounds |
| `audit_tol` | 1e-6 | violations smaller than this are ignored |
| `criterion_tol` | 1e-4 | relative slack on the criterion in the bound problems |

When more than `audit_add` points are violated, the worst violation in
every (restriction, covariate cell) group is added first, then the second
worst in every group, and so on.

## Diagnostics

```{code-cell} python
import pymte

ae = pymte.load_ae()
r = pymte.ivmte(
    ae,
    ivlike="worked ~ morekids + samesex + morekids*samesex",
    target="att",
    m0="~ u + uSplines(degree=1, knots=[.2, .4, .6, .8]) + yob",
    m1="~ uSplines(degree=2, knots=[.1, .3, .5, .7]) * yob",
    propensity="morekids ~ samesex + yob",
)
print("\n".join(r.messages))
```

`r.audit` holds the number of rounds, the final constraint set, solver
status codes and runtimes, and any violations left at termination:

```{code-cell} python
r.audit.audit_count, r.audit.constraints.n, r.audit.status
```

## Infeasible problems

If the restrictions cannot all be satisfied, the criterion problem is
infeasible and an error names the likely cause. The remedy is to relax the
bounds or monotonicity restrictions. A bound problem that is unbounded
signals an initial grid that is too coarse; the estimator then enlarges the
initial grid by half and retries, up to three times, before giving up.
