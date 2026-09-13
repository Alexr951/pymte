# FAQ and troubleshooting

## The criterion problem is infeasible

The shape restrictions cannot all hold at once, for example `m0_lb=0.2` together with `m1_ub=0.1` and `mte_inc=True`. Relax or remove restrictions until the problem is feasible. The error message names the likely conflict. A positive `criterion_tol` does not help here because the restrictions enter as hard constraints.

## A bound problem is unbounded

The initial grid is too coarse for the MTR specification: some direction in the coefficient space is unrestricted on the grid points. The estimator enlarges the initial grid by half and retries up to three times. If it still fails, increase `initgrid_nu` or `initgrid_nx`, or add bounds on the MTRs.

## The audit does not finish

With many covariate cells and a flexible specification, each round can add up to `audit_add` points and the loop may hit `audit_max`. Raise `audit_max`, or start from a larger initial grid so that fewer rounds are needed. A warning reports how many violations remain; if they are all of the order of the solver tolerance, raise `audit_tol`.

## Which solver should I use?

The defaults, HiGHS for linear programs and Clarabel for quadratically constrained problems, reproduce the published results of the R package and need no licence. Gurobi and MOSEK are picked automatically when their Python packages are installed and can be faster on large problems; pass `solver="highs"` to keep the open-source solver in that case. Solver options go through `solver_options` unchanged.

## My bounds differ from the R package in the sixth decimal

Small differences arise from the propensity score fit (R's `glm` stops at a looser tolerance than statsmodels) and from the solver. Differences at the level of `1e-6` or below are expected; anything larger points to a different specification, most often the LATE definition (see {doc}`migration`) or a different covariate grid.

## The number of audit rounds differs from the R vignette

The vignette was produced with Gurobi. The solution at a bound is not unique, so different solvers return different vertices and therefore different violations on the audit grid. The bounds agree; the round count may not.

## `criterion_tol=0` gives a wider interval in R

With a zero tolerance the feasible set of the regression approach is the set of least-squares minimisers, along which the target is constant. The width reported by Gurobi in that case comes from its feasibility tolerance on the quadratic constraint. Use a positive `criterion_tol`.

## Estimation is slow

The Angrist and Evans example (209,133 observations) takes about a second without bootstrap and about half a minute with 50 bootstrap replicates. Most of the time goes into re-fitting the propensity score and the moments on each replicate. Use `bootstraps_m` smaller than the sample size, or run replicates in parallel by calling `ivmte()` on your own resamples.

## Where is the `u` variable?

`u` is the unobserved resistance to treatment, normalised to be uniform on [0, 1]. It is never a column of the data; it appears only in the `m0` and `m1` formulas. If your data already have a column named `u`, choose another name with `uname`.
