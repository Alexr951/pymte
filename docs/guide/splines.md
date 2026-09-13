---
file_format: mystnb
kernelspec:
  name: python3
---

# B-splines in the unobservable

## Basis convention

`uSplines(degree, knots, intercept=False)` builds a B-spline basis on
$[0, 1]$ with boundary knots fixed at 0 and 1. With $K$ interior knots the
basis has `degree + K + 1` functions; unless `intercept=True` the first one
is dropped, leaving `degree + K` columns. This matches `splines2::bSpline`
and `splines2::ibs` in R, which the R package uses, so spline coefficients
carry the same meaning in both implementations. Knots at or outside the
boundaries are ignored.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from pymte import USpline

u = np.linspace(0, 1, 401)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, sp in zip(axes, [USpline(0, [0.25, 0.5, 0.75], intercept=True), USpline(2, [1/3, 2/3])]):
    ax.plot(u, sp.basis(u))
    knots = ", ".join(f"{k:.2g}" for k in sp.knots)
    ax.set_title(f"degree {sp.degree}, knots {knots}, intercept={sp.intercept}", fontsize=10)
    ax.set_xlabel("u")
plt.tight_layout()
```

Degree 0 gives piecewise constants, which together with `intercept=True`
is the natural nonparametric specification for a step function in $u$.

## Exact integrals

All moments in the estimator are integrals of the MTR against a weight in
$u$. For spline terms these are obtained from the antiderivative of the
basis, so no numerical quadrature is involved:

```{code-cell} python
sp = USpline(2, [0.3, 0.6])
# Each basis function integrates to (ub - lb) / n_full over the whole interval
# only in the piecewise-constant case; in general the integrals differ.
sp.integral(lb=0.0, ub=1.0)
```

## Coefficient names

Spline coefficients are named `uS{j}.{b}:{interaction}`: `j` indexes the
distinct spline specifications in the formula in order of appearance (main
effects before interactions), `b` is the basis function (1-based), and the
interaction is the covariate column the basis is multiplied by, or `1` for
none. The R package uses the same scheme with an additional arm prefix
(`u0S1.2:yob`); in results the arm appears as `[m0]`/`[m1]` instead.

```{code-cell} python
import pymte

ae = pymte.load_ae()
pymte.polyparse("~ uSplines(degree=2, knots=[.1, .3, .5, .7]) * yob", ae).names
```
