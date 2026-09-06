# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
