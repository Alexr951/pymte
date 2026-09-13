---
file_format: mystnb
kernelspec:
  name: python3
---

# Specification tests

Both approaches allow a test of whether the MTR specification is compatible with the IV-like estimands.

## Point identified: Hansen's J test

With more independent moments than MTR coefficients, two-step GMM yields the J statistic $n\,\bar g(\hat\theta)' \hat\Omega^{-1} \bar g(\hat\theta)$ with degrees of freedom equal to the number of overidentifying restrictions. The asymptotic p-value is always reported; with `bootstraps > 0` the statistic is also bootstrapped with the moments recentred at their sample values (Hall and Horowitz 1996), which gives a p-value that does not rely on the first-step estimation being negligible.

```{code-cell} python
import pymte

sim = pymte.load_sim_data()
r = pymte.ivmte(
    sim,
    target="ate",
    m0="~ u",
    m1="~ u",
    ivlike="y ~ d + C(z) + d:C(z)",
    propensity="d ~ C(z)",
    point=True,
    bootstraps=50,
    seed=1,
)
r.j_test
```

## Partially identified: misspecification test

When the minimum criterion in the sample is positive, the IV-like moments cannot all be matched under the shape restrictions and the specification may be rejected. Following the test of Bugni, Canay and Shi (2015) as implemented in the R package, each bootstrap replicate minimises its own criterion over the coefficients that come within `criterion_tol` of the sample's minimum criterion, and the p-value is the share of replicates whose statistic is at least the sample criterion.

```{code-cell} python
b = pymte.ivmte(
    sim,
    target="ate",
    m0="~ u",
    m1="~ u",
    ivlike="y ~ d + C(z)",
    propensity="d ~ C(z)",
    point=False,
    bootstraps=50,
    seed=1,
)
b.criterion, b.specification_p_value
```

The test is skipped when the sample criterion is zero (the moments are matched exactly) and for the regression approach, whose least-squares identified set is never empty, so its minimum criterion carries no evidence about specification. Set `specification_test=False` to skip it in any case, which saves one linear program per replicate.
