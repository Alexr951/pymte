# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- The modules and functions now follow the layout and names of the R
  package: `mst` (estimator, GMM, bootstrap inference and the result),
  `mtr` (`polyparse`, `gen_gamma`), `wweights`, `sweights`, `ivlike`,
  `design`, `lp` (with the solver interface), `monobound` and `audit`. The
  previous `estimate`, `point`, `bootstrap`, `results`, `weights`, `shape`
  and `solvers` modules are gone; `MTRSpec.from_formula` is `polyparse`,
  `fit_propensity`/`propensity_from_column` are `propensity`, and the
  package exports the same functions as the R namespace.
- A collinear regression with `point=True` falls through to the bounds, as
  in R, instead of raising.
- The summary reports `Audit reached audit_max (N)` when the audit stopped
  with violations left, as R does, instead of claiming success.
- The argument checks of the R package are enforced: IV-like formulas must
  share one outcome, the treatment variable cannot enter `m0`/`m1`, `treat`
  must agree with the propensity formula, and the variables of `late_from`
  and `late_to` must be in the propensity model. As in R, a warning names
  the IV-like specifications without the treatment variable when moments
  turn out to be dependent.

### Added

- `solver_options_criterion`, `solver_options_bounds` and
  `solver_presolve`.
- The synthetic populations of the R test suite (`pymte.testdata`) and the
  closed-form helpers of its tests (`pymte.testfunctions_covariates`,
  `pymte.testfunctions_splines`), together with the ports of
  `test_single`, `test_covariates`, `test_splines` and `test_direct_qp`,
  which rebuild the IV-like estimands, Gamma moments and bounds by hand.

## [0.1.0] - 2026-09-05

First version, ported from R ivmte 1.4.0 (GitHub HEAD, 2024-08-27).

### Added

- Package skeleton, packaging metadata and documentation scaffold.
- The `AE` (Angrist and Evans 1998 subsample) and `ivmteSimData` datasets,
  exported unchanged from the R package.
- Plots of the MTRs, the MTE and the weights (matplotlib, optional).
- Explicit term lists as an alternative to formulas for the MTRs.
- MTR specifications (`MTRSpec`): polynomial and B-spline terms in the
  unobservable, interactions with covariates, exact integrals against
  weights in `u`. Spline bases (`USpline`) reproduce `splines2::bSpline` and
  `splines2::ibs` with boundary knots 0 and 1.
- Propensity score models with logit, probit and linear links, or a
  user-supplied score column.
- Target weights for the ATE, ATT, ATU, LATE, average LATE, generalised LATE
  and custom piecewise-constant weights; IV-like estimands from OLS and TSLS
  regressions with `components` and `subset`, and their moment
  representation.
- Partial identification by linear programming (moment approach, HiGHS) and
  quadratically constrained programming (regression approach, Clarabel),
  with shape restrictions, `equal_coef`, and the audit procedure of the R
  package. Point identification by two-step GMM, OLS and equality-constrained
  least squares.
- The `ivmte()` estimator and the `IVMTEResult` container with `summary()`
  and `to_dict()`.
- Bootstrap inference: nonparametric bootstrap, m-out-of-n and subsampling;
  backward and forward confidence regions and p-values for bounds;
  percentile and normal intervals for point estimates and coefficients;
  bootstrapped Hansen J p-value; the misspecification test of the R package
  for the partially identified moment approach.

### Not ported

- `direct` norms other than least squares, `soft`, `rescale`, `debug`,
  `smallreturnlist`, the deprecated `lpsolver*` arguments and the CPLEX
  backend.
