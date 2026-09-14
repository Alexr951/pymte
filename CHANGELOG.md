# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [1.0.2] - 2026-09-13

### Fixed

- The `late` and `avglate` targets raise a `ValueError` when the propensity scores at `late_from` and `late_to` coincide after clipping to [0, 1], so the LATE interval is empty. Previously this produced a numpy `RuntimeWarning` and infinite weights.

### Changed

- The public API is now the set of names in `pymte.__all__`: `ivmte`, `IVMTEResult`, `propensity`, `Propensity`, `MTRSpec`, `USpline`, `load_ae`, `load_sim_data`, `plot_mtr`, `plot_mte` and `plot_weights`. The building blocks of the estimator (`ivmte_estimate`, `polyparse`, `design`, the `lp`, `mst`, `mtr`, `wweights`, `sweights`, `ivlike`, `monobound` and `audit` functions) remain importable from the package and their modules, but they are no longer exported by `from pymte import *` and are not covered by semantic versioning.

## [1.0.1] - 2026-09-13

### Fixed

- The licence metadata reads `GPL-3.0-only` and the distribution includes the `NOTICE` file.

## [1.0.0] - 2026-09-13

First public release, ported from R ivmte 1.4.0 (GitHub HEAD, 2024-08-27).
