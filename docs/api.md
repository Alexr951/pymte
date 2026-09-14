# API reference

The public API consists of the names below: the estimator and its result, the MTR and propensity score specifications, the datasets and the plots. These are the names exported by `pymte` and covered by semantic versioning.

## Estimation

```{eval-rst}
.. currentmodule:: pymte

.. autosummary::
   :toctree: generated
   :nosignatures:

   ivmte
   IVMTEResult
```

## Specifications and inputs

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   MTRSpec
   USpline
   propensity.propensity
   Propensity
   load_ae
   load_sim_data
```

## Plots

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_mtr
   plot_mte
   plot_weights
```

## Building blocks

The estimator is assembled from the following functions, which follow the R package: `mst` holds the estimator, `mtr` the MTR specifications, `wweights` and `sweights` the target and IV-like weights, `lp` the optimisation problems, `monobound` the shape restrictions and `audit` the audit procedure. The names are those of the R package in snake case. They are importable from their modules and listed here for users of the R package, but they are not part of the public API and may change between minor versions.

```{eval-rst}
.. currentmodule:: pymte

.. autosummary::
   :toctree: generated
   :nosignatures:

   mst.ivmte_estimate
   mtr.polyparse
   mtr.gen_gamma
   mtr.gen_gamma_splines
   mst.gen_target
   mst.gen_s_set
   mst.gmm_estimate
   mst.moment_matrix
   mst.bound_ci
   mst.bound_pvalue
   wweights.wate1
   wweights.watt1
   wweights.watu1
   wweights.wlate1
   wweights.wgenlate1
   wweights.gen_weight
   design.design
   ivlike.iv_estimate
   ivlike.piv
   sweights.olsj
   sweights.tsls
   monobound.gengrid
   monobound.genbound_a
   monobound.genmono_a
   monobound.combinemonobound
   monobound.genmonobound_a
   audit.audit
   audit.AuditResult
   audit.select_violations
   audit.rhalton
   lp.lp_setup
   lp.lp_setup_equal_coef
   lp.lp_setup_criterion
   lp.lp_setup_bound
   lp.lp_setup_criterion_boot
   lp.criterion_min
   lp.bound
   lp.qp_setup
   lp.qp_setup_criterion
   lp.qp_setup_bound
   lp.run_lp
   lp.run_qcqp
```

## Test data

The synthetic populations of the R package's test suite, for experiments and for the tests in `tests/`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   testdata.gendist_basic
   testdata.gendist_covariates
   testdata.gendist_splines
   testdata.gendist_mosquito
```
