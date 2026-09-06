# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Package skeleton, packaging metadata and documentation scaffold.
- The `AE` (Angrist and Evans 1998 subsample) and `ivmteSimData` datasets,
  exported unchanged from the R package.
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
