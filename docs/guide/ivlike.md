---
file_format: mystnb
kernelspec:
  name: python3
---

# IV-like estimands

The moment approach matches the MTR functions to *IV-like estimands*:
coefficients of linear regressions of the outcome that can be written as
$\beta_s = E[s(D, X, Z)\,Y]$ for a known weight $s$. Mogstad, Santos and
Torgovitsky (2018, Proposition 1) show that each such coefficient is a
linear function of the MTR coefficients, which yields the moment
conditions the estimator uses. Adding estimands can only shrink the
identified set; which ones to use is the researcher's choice, and is the
main difference from the regression approach, which uses the whole
conditional mean of the outcome.

## Specifying regressions

`ivlike` takes one formula or a list of formulas. Without a `|` the
regression is estimated by OLS; the part after `|` lists the instruments
for two-stage least squares (exogenous covariates must be repeated there,
as in R):

```{code-cell} python
import pymte

sim = pymte.load_sim_data()
spec = dict(
    target="ate",
    m0="~ uSplines(degree=1, knots=[.25, .5, .75]) + x",
    m1="~ uSplines(degree=1, knots=[.25, .5, .75]) + x",
    propensity="d ~ z + x",
)
r = pymte.ivmte(
    sim,
    ivlike=[
        "y ~ I(z == 1) + I(z == 2) + I(z == 3) + x",
        "y ~ d + x",
        "y ~ d | z",
    ],
    **spec,
)
r
```

Every coefficient of every regression is a moment by default. The number
of linearly independent moments is reported and compared with the number of
MTR coefficients to decide between point and partial identification.

```{code-cell} python
r.ivlike.names, r.ivlike.beta.round(4)
```

## Selecting coefficients

`components` names the coefficients to use from each regression, with
`"intercept"` for the constant and `None` for all of them:

```{code-cell} python
pymte.ivmte(
    sim,
    ivlike=["y ~ I(z == 1) + I(z == 2) + I(z == 3) + x", "y ~ d + x", "y ~ d | z"],
    components=[["intercept", "x"], ["d"], None],
    **spec,
).bounds
```

## Estimating on subsamples

`subset` gives one row-selection expression per regression, evaluated with
:meth:`pandas.DataFrame.eval`; the corresponding moments then average over
that subsample:

```{code-cell} python
pymte.ivmte(
    sim,
    ivlike=["y ~ z + x", "y ~ d + x", "y ~ d | z"],
    subset=["x <= 9", None, "z in [1, 3]"],
    target="ate",
    m0="~ uSplines(degree=3, knots=[.25, .5, .75]) + x",
    m1="~ uSplines(degree=3, knots=[.25, .5, .75]) + x",
    propensity="d ~ z + x",
).bounds
```

## The weights and moments

The fitted regressions and their weights are available for inspection.
Each moment's ``gamma0``/``gamma1`` rows are the integrals of the MTR bases
against $s(0, X, Z)\,1\{u > p\}$ and $s(1, X, Z)\,1\{u \le p\}$:

```{code-cell} python
import pandas as pd

pd.DataFrame(r.ivlike.gamma1, index=r.ivlike.names, columns=r.specs[1].names).round(3)
```
