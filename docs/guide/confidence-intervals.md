---
file_format: mystnb
kernelspec:
  name: python3
---

# Confidence intervals

Inference uses the nonparametric bootstrap: the whole estimation, including
the propensity score, is repeated on `bootstraps` resamples of the data.
Set `bootstraps_m` for an m-out-of-n bootstrap and `bootstraps_replace=False`
for subsampling. The audit grid is held fixed across replicates, as in the
R package.

```{code-cell} python
import ivmte

sim = ivmte.load_sim_data()
r = ivmte.ivmte(
    sim,
    target="late",
    late_from={"z": 1},
    late_to={"z": 3},
    m0="~ u + I(u**2) + I(u**3) + x",
    m1="~ u + I(u**2) + I(u**3) + x",
    ivlike="y ~ d + z + d*z",
    propensity="d ~ z + x",
    bootstraps=50,
    seed=1,
)
print(r.summary())
```

## Bounds

For a partially identified target the package reports the *backward* and
*forward* confidence regions of the R package. With $\hat\theta = (lb, ub)$,
$L_b = \sqrt{m}(lb_b - lb)$ and $U_b = \sqrt{m}(ub_b - ub)$ over the
bootstrap draws, the backward region at level $\alpha$ is

$$
\Big[\, lb + \tfrac{1}{\sqrt n} Q_{L}\big(\tfrac{1-\alpha}{2}\big),\;
      ub + \tfrac{1}{\sqrt n} Q_{U}\big(\tfrac{1+\alpha}{2}\big) \Big],
$$

and the forward region swaps the quantiles and signs. Quantiles are of type
1 (inverse empirical distribution function). Both regions are stored;
`ci_type` picks the one shown in the summary.

```{code-cell} python
r.bounds_ci["forward"]
```

The p-value for a zero target inverts the region: it is one minus the
largest level at which the region still excludes zero.

```{code-cell} python
r.p_value, r.bounds_se
```

## Point estimates

For a point identified target the summary shows percentile intervals; a
normal approximation using the bootstrap standard error is also stored.

```{code-cell} python
p = ivmte.ivmte(
    sim,
    target="ate",
    m0="~ u",
    m1="~ u",
    ivlike="y ~ d + C(z)",
    propensity="d ~ C(z)",
    bootstraps=50,
    seed=1,
)
p.point_estimate_ci["normal"]
```

Standard errors and intervals are also available for every MTR coefficient
and every propensity score coefficient:

```{code-cell} python
p.mtr_se.round(4)
```

```{code-cell} python
p.propensity_ci["nonparametric"].round(3)
```

## Failed draws

A resample can be unusable, for example when a factor level disappears or
a bound problem has no solution. Such draws are discarded and redrawn; the
count is reported in `bootstraps_failed`. Bootstrap draws in R and Python
come from different random number generators, so individual replicates
differ between the two implementations while the intervals agree up to
simulation noise.
