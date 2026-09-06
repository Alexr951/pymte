# API reference

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
   Propensity
   fit_propensity
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

The estimator is assembled from the following functions, which can be used
on their own.

```{eval-rst}
.. currentmodule:: pymte

.. autosummary::
   :toctree: generated
   :nosignatures:

   weights.conventional_weights
   weights.custom_target_gammas
   weights.target_gammas_from_weights
   ivlike.fit_ivlike
   ivlike.build_moments
   shape.build_grids
   shape.shape_constraints
   shape.halton
   lp.L1Criterion
   lp.LSCriterion
   lp.build_constraints
   lp.solve_criterion
   lp.solve_bound
   audit.run_audit
   audit.AuditResult
   point.gmm
   point.least_squares
   bootstrap.bound_ci
   bootstrap.bound_pvalue
   bootstrap.point_ci
   solvers.solve_lp
   solvers.solve_qcqp
```
