---
file_format: mystnb
kernelspec:
  name: python3
---

# Plotting MTRs, MTEs and weights

Three helpers draw the estimated objects with matplotlib (install with `pip install "pymte[plots]"`). Each returns the axes it drew on and accepts `ax=` to draw into an existing figure.

## MTR functions

In the partially identified case the MTRs at the lower and upper bound are shown; with covariates in the specification, pass their values with `at`.

```{code-cell} python
import pymte
import matplotlib.pyplot as plt

ae = pymte.load_ae()
r = pymte.ivmte(
    ae,
    target="att",
    m0="~ 0 + uSplines(degree=2, knots=[1/3, 2/3])",
    m1="~ 0 + uSplines(degree=2, knots=[1/3, 2/3])",
    m0_inc=True,
    m1_inc=True,
    mte_dec=True,
    ivlike="worked ~ morekids + samesex + morekids*samesex",
    propensity="morekids ~ samesex",
)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
pymte.plot_mtr(r, ax=axes[0])
pymte.plot_mte(r, ax=axes[1])
plt.tight_layout()
```

The coefficients behind the curves are in `r.gstar_coef`; the point identified case draws the single estimated curve from `r.mtr_coef`.

## Weights

`plot_weights` shows the sample-average weight that the target parameter and each IV-like estimand place on $m_1(u)$. Where the IV-like weights are zero the data carry no information about the MTR, which is what makes extrapolation to the target parameter depend on the shape restrictions.

```{code-cell} python
fig, ax = plt.subplots(figsize=(6, 3.2))
pymte.plot_weights(r, ax=ax)
```
