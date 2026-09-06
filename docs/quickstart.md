---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart

The introductory example of the R package uses the Angrist and Evans (1998)
data: the effect of having a third child on whether the mother worked,
instrumented by whether the first two children have the same sex.

```{code-cell} python
import ivmte

ae = ivmte.load_ae()
ae.head()
```

## A partially identified model

We specify linear MTR functions in the unobservable $u$ with an additive
year-of-birth effect, take as IV-like estimands the coefficients of an OLS
regression of `worked` on `morekids`, `samesex` and their interaction, fit a
logit propensity score, and ask for the average treatment effect on the
treated.

```{code-cell} python
r = ivmte.ivmte(
    ae,
    target="att",
    m0="~ u + yob",
    m1="~ u + yob",
    ivlike="worked ~ morekids + samesex + morekids*samesex",
    propensity="morekids ~ samesex + yob",
)
r
```

Four IV-like moments cannot pin down six MTR coefficients, so the result is
a pair of bounds. They match the R package to seven digits. The result
object carries everything the estimator computed:

```{code-cell} python
r.bounds, r.moments, r.audit.audit_count
```

```{code-cell} python
r.gstar_coef.round(4)
```

The `messages` attribute is the progress log R prints with `noisy = TRUE`:

```{code-cell} python
print("\n".join(r.messages))
```

## A point identified model

With `m0 = m1 = "~ u"` the four moments identify the four coefficients, and
the estimator switches to GMM:

```{code-cell} python
p = ivmte.ivmte(
    ae,
    target="att",
    m0="~ u",
    m1="~ u",
    ivlike="worked ~ morekids + samesex + morekids*samesex",
    propensity="morekids ~ samesex",
)
p
```

```{code-cell} python
p.mtr_coef.round(5)
```

## The regression approach

Naming the outcome fits the MTRs to its conditional means directly, with no
IV-like estimands:

```{code-cell} python
q = ivmte.ivmte(
    ae,
    target="att",
    m0="~ u + yob",
    m1="~ u + yob",
    outcome="worked",
    propensity="morekids ~ samesex + yob",
)
q.point_estimate
```

The user guide covers each ingredient: {doc}`guide/mtr-specification`,
{doc}`guide/target-parameters`, {doc}`guide/ivlike`,
{doc}`guide/shape-restrictions` and {doc}`guide/audit`.
