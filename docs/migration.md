# Migrating from the R package

`pymte.ivmte()` takes the same arguments as `ivmte::ivmte()` with dots
replaced by underscores. Formulas are strings, R vectors become Python
lists or dicts, and results are attributes of the returned object instead
of list elements. Numerical results agree with the R package to the
tolerances documented in the test suite.

## Formulas

| R | Python | note |
|---|---|---|
| `m0 = ~ u + I(u^2) + x` | `m0="~ u + I(u**2) + x"` | `I(u^2)` is also accepted |
| `factor(z)` | `C(z)` | formulaic syntax |
| `(yob == 55)` | `I(yob == 55)` | boolean terms need `I()` |
| `uSplines(degree = 1, knots = c(.2, .4))` | `uSplines(degree=1, knots=[.2, .4])` | `c(...)` is accepted too |
| `ivlike = c(y ~ d, y ~ d \| z)` | `ivlike=["y ~ d", "y ~ d | z"]` | |
| `propensity = d ~ z + x` | `propensity="d ~ z + x"` | |
| `propensity = p` (a column) | `propensity="p", treat="d"` | |

## Arguments

| R | Python |
|---|---|
| `target = "att"` | `target="att"` |
| `late.from = c(z = 1)` | `late_from={"z": 1}` |
| `late.to`, `late.X` | `late_to`, `late_x` |
| `genlate.lb`, `genlate.ub` | `genlate_lb`, `genlate_ub` |
| `target.weight0 = c(0, f, 0)` | `target_weight0=[0, f, 0]` |
| `target.knots0 = c(k1, k2)` | `target_knots0=[k1, k2]` |
| `uname = v` | `uname="v"` |
| `m0.lb`, `m0.ub`, `m1.lb`, `m1.ub`, `mte.lb`, `mte.ub` | `m0_lb`, ... |
| `m0.inc`, `m0.dec`, `m1.inc`, `m1.dec`, `mte.inc`, `mte.dec` | `m0_inc`, ... |
| `equal.coef = ~ 0 + x` | `equal_coef="~ 0 + x"` |
| `components = l(c(intercept, x), c(d), )` | `components=[["intercept", "x"], ["d"], None]` |
| `subset = l(x <= 9, 1 == 1, z %in% c(1, 3))` | `subset=["x <= 9", None, "z in [1, 3]"]` |
| `link = "probit"` | `link="probit"` |
| `outcome = "y"` | `outcome="y"` |
| `solver = "gurobi"` | `solver="gurobi"` (also `"mosek"`, `"highs"`, `"clarabel"`) |
| `solver.options = list(...)` | `solver_options={...}` |
| `criterion.tol` | `criterion_tol` |
| `initgrid.nx`, `initgrid.nu`, `audit.nx`, `audit.nu` | `initgrid_nx`, ... |
| `initgrid.x`, `initgrid.u`, `audit.x`, `audit.u` | `initgrid_x`, ... |
| `audit.add`, `audit.max`, `audit.tol` | `audit_add`, `audit_max`, `audit_tol` |
| `point`, `point.eyeweight` | `point`, `point_eyeweight` |
| `bootstraps`, `bootstraps.m`, `bootstraps.replace` | `bootstraps`, `bootstraps_m`, `bootstraps_replace` |
| `levels`, `ci.type`, `specification.test` | `levels`, `ci_type`, `specification_test` |
| `noisy` | `noisy` |
| (R's global RNG) | `seed` |

Not available: `direct` other than the default least squares, `soft`,
`rescale`, `debug`, `smallreturnlist`, the deprecated `lpsolver*` arguments
and the CPLEX backend.

## Results

| R | Python |
|---|---|
| `r$bounds` | `r.bounds` (a tuple) |
| `r$point.estimate` | `r.point_estimate` |
| `r$mtr.coef` | `r.mtr_coef` (a Series) |
| `r$gstar.coef$min.g0`, `$max.g1`, ... | `r.gstar_coef` (DataFrame with columns `min`, `max`, `criterion`) |
| `r$gstar$g0`, `r$gstar$g1` | `r.gstar` (one Series over both arms) |
| `r$gstar.weights` | `r.target_gammas.weights` |
| `r$s.set` | `r.ivlike` |
| `r$moments` | `r.moments` |
| `r$propensity$model` | `r.propensity` |
| `r$audit.count`, `r$audit.criterion`, `r$audit.grid` | `r.audit.audit_count`, `r.criterion`, `r.audit` |
| `r$splines.dict` | `r.specs[0].splines`, `r.specs[1].splines` |
| `r$messages` | `r.messages` |
| `r$bounds.ci`, `r$p.value`, `r$specification.p.value` | `r.bounds_ci`, `r.p_value`, `r.specification_p_value` |
| `r$point.estimate.ci`, `r$j.test` | `r.point_estimate_ci`, `r.j_test` |
| `summary(r)`, `print(r)` | `r.summary()`, `print(r)` |

Coefficient names carry the `[m0]`/`[m1]` prefixes of the R package.
Spline coefficients are named `uS{j}.{b}:{interaction}` (R: `u0S{j}.{b}`).

## Behavioural differences

- **LATE.** R redefined `late` in July 2022; the R vignette's LATE numbers
  use the earlier pointwise weights, which both R and this package now call
  `avglate`.
- **Grids.** The covariate grid is sampled with NumPy's generator (`seed`),
  R uses its own. Results agree when the whole covariate support fits in
  `audit_nx`, which is the case in the examples; otherwise pass explicit
  grids to compare implementations.
- **Audit rounds.** The number of rounds depends on which optimal vertex
  the solver returns and can differ between solvers; the bounds agree.
- **Regression approach.** Point identified regressions run without a
  commercial solver. The quadratically constrained bounds use Clarabel.
- **Spline intercept.** As in the current R source (and unlike CRAN 1.4.0),
  `uSplines` drops the first basis function unless `intercept=True`.
