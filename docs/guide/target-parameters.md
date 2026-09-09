---
file_format: mystnb
kernelspec:
  name: python3
---

# Target parameters

Every target parameter is a weighted average of the two marginal treatment
response functions,

$$
\beta^\star = E\left[\int_0^1 \big(m_1(u, X)\,\omega_1^\star(u, X, Z) + m_0(u, X)\,\omega_0^\star(u, X, Z)\big)\,du\right],
$$

with weights that are known or identified from the data. For a treatment
effect the two weights are negatives of each other,
$\omega_0^\star = -\omega_1^\star$, so that the parameter is a weighted average
of the MTE $m_1 - m_0$; the weights on $m_1$ are tabulated below. The
package supports the conventional parameters and arbitrary
piecewise-constant weights in $u$.

| `target`    | $\omega_1^\star$                                           | description |
|-------------|------------------------------------------------------------|-------------|
| `"ate"`     | $1$                                                        | average treatment effect |
| `"att"`     | $1\{u \le p(X,Z)\}/P(D=1)$                                 | on the treated |
| `"atu"`     | $1\{u > p(X,Z)\}/P(D=0)$                                   | on the untreated |
| `"late"`    | $1\{p(X,z_0) < u \le p(X,z_1)\}/\lvert E[p(X,z_1)] - E[p(X,z_0)]\rvert$ | LATE from `late_from` to `late_to` |
| `"avglate"` | $1\{p(X,z_0) < u \le p(X,z_1)\}/\lvert p(X,z_1) - p(X,z_0)\rvert$ | population average of the covariate-specific LATEs |
| `"genlate"` | $1\{\underline u < u \le \bar u\}/(\bar u - \underline u)$ | generalised LATE between `genlate_lb` and `genlate_ub` |

$\omega_0^\star = -\omega_1^\star$ in all cases. The LATE weights are those of
Imbens and Angrist (1994) written in the selection model: the compliers
from $z_0$ to $z_1$ are the units with $p(X, z_0) < U \le p(X, z_1)$. The
generalised LATE of Heckman and Vytlacil (2005) replaces the two
propensity scores by chosen values of $u$, which allows a LATE to be
extrapolated beyond the support of the instrument.

```{code-cell} python
import pymte

sim = pymte.load_sim_data()
common = dict(
    ivlike="y ~ d + z + d*z",
    m0="~ u + I(u**2) + I(u**3) + x",
    m1="~ u + I(u**2) + I(u**3) + x",
    propensity="d ~ z + x",
)
pymte.ivmte(sim, target="late", late_from={"z": 1}, late_to={"z": 3}, **common).bounds
```

## Conditioning on covariates

`late_x` restricts the LATE or generalised LATE to a covariate cell. The
propensity score is then evaluated at the fixed covariate values and the
expectation is taken over that cell only. No smoothing is done, so the
conditioning variables must be discrete and the cell must contain
observations:

```{code-cell} python
pymte.ivmte(sim, target="late", late_from={"z": 1}, late_to={"z": 3}, late_x={"x": 2}, **common).bounds
```

```{code-cell} python
pymte.ivmte(sim, target="genlate", genlate_lb=0.2, genlate_ub=0.42, **common).bounds
```

## Policy relevant treatment effects

Mogstad and Torgovitsky (2018, Table 2) express the policy relevant
treatment effects of Heckman and Vytlacil (2001) as weights of the same
form. A policy that raises every propensity score by $\alpha$ has

$$
\omega_1^\star(u, X, Z) = \frac{\mathbf 1\{u \le p(X, Z) + \alpha\} - \mathbf 1\{u \le p(X, Z)\}}{\alpha},
$$

which is piecewise constant in $u$ with knots at $p(X, Z)$ and
$p(X, Z) + \alpha$ and can be passed through the custom weights described
below. The same holds for a proportional change in the propensity score
and for a shift in one component of the instrument.

```{note}
The R package redefined `late` in July 2022. Before that, `late` used the
pointwise weights now called `avglate`, and the numbers in the R vignette
were produced with the old definition. Use `target="avglate"` to reproduce
them; the two coincide when `late_x` fixes all covariates entering the
propensity score.
```

## Custom weights

Any weight that is piecewise constant in $u$ can be passed directly. Knots
split $[0, 1]$ into pieces and each piece receives a weight; both can be
numbers or functions of covariates, whose argument names must match column
names. The following replicates the conditional LATE above by hand:

```{code-cell} python
import pandas as pd

prop = pymte.propensity("d ~ z + x", sim)
px = (sim["x"] == 2).mean()

def p_at(x, z):
    return float(prop.predict(pd.DataFrame({"x": [x], "z": [z]}))[0])

def weight1(x):
    return 0.0 if x != 2 else 1.0 / ((p_at(2, 3) - p_at(2, 1)) * px)

def weight0(x):
    return -weight1(x)

def knot1(x):
    return p_at(x, 1)

def knot2(x):
    return p_at(x, 3)

custom = pymte.ivmte(
    sim,
    target_knots0=[knot1, knot2],
    target_knots1=[knot1, knot2],
    target_weight0=[0, weight0, 0],
    target_weight1=[0, weight1, 0],
    **common,
)
custom.bounds
```

No sign convention is imposed on custom weights: for a treatment effect the
`m0` weights are the negatives of the `m1` weights, as above.

## Inspecting the weights

The integrated weights are returned as `gstar`, one entry per MTR
coefficient, so that the target equals `gstar @ theta`:

```{code-cell} python
custom.gstar.round(4)
```
