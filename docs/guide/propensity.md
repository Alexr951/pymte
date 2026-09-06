---
file_format: mystnb
kernelspec:
  name: python3
---

# The propensity score

The MTE framework normalises the unobservable so that treatment is
$D = 1\{u \le p(X, Z)\}$, where $p(X, Z) = P(D = 1 \mid X, Z)$ is the
propensity score. Every weight and every moment in the estimator is a
function of $p$, so the propensity score is estimated first.

## Fitting a model

Pass a two-sided formula as `propensity`; the left-hand side names the
treatment variable. The link is `"logit"` (default), `"probit"` or
`"linear"`. Fitted values from the linear probability model are truncated to
$[0, 1]$.

```python
ivmte(..., propensity="morekids ~ samesex + yob", link="probit")
```

The formula uses the same syntax as the MTR specifications, so instruments
with several values can be entered as factors, `"d ~ C(z) + x"`.

```{code-cell} python
import ivmte

ae = ivmte.load_ae()
prop = ivmte.fit_propensity(ae, "morekids ~ samesex + yob")
dict(zip(prop.names, prop.params.round(5)))
```

The fitted object predicts on new data, which is how the `late` target
evaluates the score at the `late_from` and `late_to` instrument values:

```{code-cell} python
import pandas as pd

prop.predict(pd.DataFrame({"samesex": [0, 1], "yob": [50, 50]}))
```

## Supplying a score

If the propensity score has been estimated elsewhere, pass its column name
as `propensity` and name the treatment variable with `treat`:

```python
ivmte(..., propensity="phat", treat="d")
```

Targets that need to evaluate the score at counterfactual instrument values
(`late`, `avglate`) require a fitted model and are not available with a
supplied column.
