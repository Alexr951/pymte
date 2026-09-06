---
file_format: mystnb
kernelspec:
  name: python3
---

# Specifying the MTR functions

The marginal treatment response (MTR) functions $m_0(u, x)$ and $m_1(u, x)$
are the objects being estimated. Each is specified by a one-sided formula in
the unobservable $u$ (uniformly distributed on $[0, 1]$ by normalisation)
and the covariates in the data. The formulas are passed as `m0` and `m1`.

## Polynomials in $u$

The simplest specifications are polynomials in $u$ whose coefficients may
depend on covariates:

```python
m0 = "~ u + yob"
m1 = "~ u + I(u**2) + I(u**3) + yob + u:yob"
```

Formulas follow the syntax of [formulaic](https://matthewwardrop.github.io/formulaic/),
which is close to R's: `+` adds terms, `:` interacts, `a*b` expands to
`a + b + a:b`, `0 +` or `-1` removes the intercept, and `I(...)` protects a
Python expression. The R spelling `I(u^2)` is accepted as a convenience and
rewritten to `I(u**2)`.

The unobservable must enter every term as a monomial: `u`, `I(u**k)`, or
one of these interacted with covariates (`x:u`, `x:I(u**3)`). Anything else,
such as `log(u)` or `I((x*u)**2)`, is rejected with an error, because the
package integrates the MTR analytically in $u$ and needs to know the exact
form of the $u$-dependence.

## Splines in $u$

Nonparametric specifications use B-splines in $u$ through the `uSplines`
term:

```python
m0 = "~ u + uSplines(degree=1, knots=[.2, .4, .6, .8]) + yob"
m1 = "~ uSplines(degree=2, knots=[.1, .3, .5, .7]) * yob"
```

`degree` is required; `knots` lists the interior knots (the boundary knots
are always 0 and 1); `intercept=True` keeps the first basis function so that
the block spans constants. Splines may be interacted with covariates, in
which case every basis function is multiplied by the covariate. See
{doc}`splines` for the basis convention and how coefficients are named.

## Covariates, factors and booleans

Covariates enter through the same formula syntax. Categorical variables are
wrapped in `C()`:

```python
m1 = "~ u + C(yob)"  # a dummy for each year of birth
m1 = "~ u + I(yob == 55)"  # a single indicator
```

## Renaming the unobservable

If `u` is already a column in the data, choose another name with `uname`:

```python
ivmte(..., m0="~ v + I(v**2) + yob", m1="~ v + yob", uname="v")
```

## Specifying terms without formulas

The same specification can be given as an explicit list of terms, each a
u-part (an integer exponent or a `USpline`) times a data column (`None` for
a constant). This is convenient when the specification is generated
programmatically:

```{code-cell} python
from pymte import USpline

terms = [(0, None), (1, None), (0, "yob"), (USpline(degree=1, knots=[.2, .4, .6, .8]), None)]
# equivalent to "~ u + yob + uSplines(degree=1, knots=[.2, .4, .6, .8])"
```

Term lists are accepted wherever a formula is, including `m0` and `m1` in
`ivmte()`.

## Working with specifications directly

`MTRSpec` parses a formula against a data frame and exposes the structure
the estimator uses: coefficient names, the exponent of $u$ in each
polynomial term, and the spline blocks.

```{code-cell} python
import pymte

ae = pymte.load_ae()
spec = pymte.MTRSpec.from_formula("~ u + I(u**2) + yob + u:yob", ae)
spec.names, spec.exponents
```

Its two numerical methods are the building blocks of everything else in the
package. `design(data, u)` evaluates the basis at given $(u, x)$ points, so
that `design @ theta` is the MTR; `gamma(data, lb, ub, weight)` returns, for
each observation, the weighted integral of every basis function over
$[lb_i, ub_i]$:

```{code-cell} python
import numpy as np

rows = ae.head(3)
spec.design(rows, u=[0.1, 0.5, 0.9]).round(3)
```

```{code-cell} python
spec.gamma(rows, lb=0.0, ub=0.5, weight=1.0).round(4)
```
