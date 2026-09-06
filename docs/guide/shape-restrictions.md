---
file_format: mystnb
kernelspec:
  name: python3
---

# Shape restrictions

Bounds on a target parameter can be tightened considerably by restrictions
on the MTR functions that are implied by economic reasoning. Three kinds are
supported, all imposed pointwise in $u$ (and in the covariates) through
linear inequalities on the MTR coefficients.

## Bounds on the MTRs

`m0_lb`, `m0_ub`, `m1_lb`, `m1_ub` bound $m_0$ and $m_1$; `mte_lb`,
`mte_ub` bound the marginal treatment effect $m_1 - m_0$. By default $m_0$
and $m_1$ are restricted to the observed range of the outcome, which for a
binary outcome means $[0, 1]$; the MTE is unrestricted unless bounds are
given.

## Monotonicity

`m0_inc`, `m0_dec`, `m1_inc`, `m1_dec` impose that the MTRs are weakly
increasing or decreasing in $u$; `mte_inc`, `mte_dec` do the same for the
MTE. Setting both `inc` and `dec` forces constancy in $u$.

```{code-cell} python
import pymte

ae = pymte.load_ae()
common = dict(
    ivlike="worked ~ morekids + samesex + morekids*samesex",
    target="att",
    m0="~ u + uSplines(degree=1, knots=[.2, .4, .6, .8]) + yob",
    m1="~ uSplines(degree=2, knots=[.1, .3, .5, .7]) * yob",
    propensity="morekids ~ samesex + yob",
)
pymte.ivmte(ae, **common).bounds
```

```{code-cell} python
pymte.ivmte(ae, m0_inc=True, m1_inc=True, mte_dec=True, **common).bounds
```

## Equality of coefficients across arms

`equal_coef` names terms whose coefficients must be the same in $m_0$ and
$m_1$, for instance a covariate effect that does not vary with treatment:

```{code-cell} python
sim = pymte.load_sim_data()
r = pymte.ivmte(
    sim,
    outcome="y",
    target="ate",
    m0="~ x + u",
    m1="~ x + u",
    equal_coef="~ 0 + x",
    propensity="d ~ x + C(z)",
)
r.mtr_coef.round(6)
```

## How the restrictions are imposed

The restrictions cannot be checked at every $u$, so they are imposed on a
grid and then verified on a finer grid by the audit procedure described in
{doc}`audit`. In the point identified case the restrictions are not
imposed at all; a warning says so.
