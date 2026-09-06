# Theory primer

The options of `ivmte()` are statements about the model below. The notation follows
Mogstad, Santos and Torgovitsky (2018), hereafter MST; Mogstad and
Torgovitsky (2018) survey the same material at greater length.

## The selection model

There is a binary treatment $D$, an outcome $Y = D Y_1 + (1 - D) Y_0$,
covariates $X$ and an instrument $Z$. Treatment follows a threshold
crossing rule,

$$
D = \mathbf 1\{U \le p(X, Z)\}, \qquad U \mid X, Z \sim \text{Uniform}[0, 1],
$$

where $p(X, Z) = P(D = 1 \mid X, Z)$ is the propensity score and $U$ is the
unobserved resistance to treatment. Together with independence of
$(Y_0, Y_1, U)$ from $Z$ given $X$, this is the model of Heckman and
Vytlacil (2005) and is equivalent to the monotonicity condition of Imbens
and Angrist (1994). Units with small $U$ take treatment whenever the
instrument offers it; units with large $U$ do not.

## From LATE to MTE

The LATE identified by moving the instrument from $z_0$ to $z_1$ is the
average effect for units with $p(x, z_0) < U \le p(x, z_1)$: the compliers.
Any parameter for a different population, for instance the average effect
on all treated units, requires effects for values of $U$ that no complier
group covers. The marginal treatment effect

$$
\text{MTE}(u, x) = E[Y_1 - Y_0 \mid U = u, X = x]
$$

is the object that lets one reason about such extrapolation: every
conventional parameter is a weighted average of the MTE with a known
weight in $u$.

## Marginal treatment response functions

The package works with the two *marginal treatment response* (MTR)
functions

$$
m_0(u, x) = E[Y_0 \mid U = u, X = x], \qquad m_1(u, x) = E[Y_1 \mid U = u, X = x],
$$

so that $\text{MTE} = m_1 - m_0$. The user specifies $m_0$ and $m_1$ as
linear combinations of known basis functions in $u$ and $x$,

$$
m_d(u, x) = \sum_{j} \theta_{d j}\, b_{d j}(u, x),
$$

polynomials in $u$ with covariate-dependent coefficients, or B-splines in
$u$ interacted with covariates ({doc}`guide/mtr-specification`). Because
the bases are known functions of $u$, every integral below is computed
exactly.

## Target parameters

A target parameter is

$$
\beta^\star = E\left[\int_0^1 m_1(u, X)\,\omega_1^\star(u, X, Z)\,du\right]
            + E\left[\int_0^1 m_0(u, X)\,\omega_0^\star(u, X, Z)\,du\right]
$$

with weights that follow from the definition of the parameter, for
example $\omega_1^\star = \mathbf 1\{u \le p(X, Z)\} / P(D = 1)$ and
$\omega_0^\star = -\omega_1^\star$ for the ATT. Substituting the basis
expansion,

$$
\beta^\star = \sum_j \theta_{0j}\, \gamma^\star_{0j} + \sum_j \theta_{1j}\, \gamma^\star_{1j},
\qquad
\gamma^\star_{dj} = E\left[\int_0^1 b_{dj}(u, X)\,\omega_d^\star(u, X, Z)\,du\right],
$$

so the target is linear in the MTR coefficients $\theta$. The package
returns the vector $\gamma^\star$ as `gstar` ({doc}`guide/target-parameters`).

## IV-like estimands

MST show that the same representation holds for any *IV-like estimand*:
a coefficient $\beta_s$ from a linear regression of $Y$ on functions of
$(D, X, Z)$, possibly instrumented, which can be written as
$\beta_s = E[s(D, Z)\, Y]$ for a known weight $s$. For such an estimand,

$$
\beta_s = \sum_j \theta_{0j}\, \gamma_{0j}(s) + \sum_j \theta_{1j}\, \gamma_{1j}(s),
$$

with $\gamma_{1j}(s) = E[s(1, Z)\int_0^{p(X,Z)} b_{1j}(u, X)\,du]$ and
$\gamma_{0j}(s) = E[s(0, Z)\int_{p(X,Z)}^1 b_{0j}(u, X)\,du]$. The data thus
provide a system of linear equations in $\theta$, one per IV-like estimand
({doc}`guide/ivlike`).

## Point and partial identification

If the system $\Gamma \theta = \beta$ has as many linearly independent
equations as unknowns, $\theta$ and hence $\beta^\star$ are point
identified; the package estimates $\theta$ by two-step GMM (or by least
squares in the regression approach) and reports a point estimate with
bootstrap intervals ({doc}`guide/confidence-intervals`).

Otherwise the identified set for $\beta^\star$ is an interval obtained by
minimising and maximising $\gamma^{\star\prime}\theta$ over the
coefficients that fit the data. With sampling noise the system rarely has
an exact solution, so MST define the minimum criterion

$$
\hat Q = \min_\theta \sum_s \big|\hat\Gamma_s \theta - \hat\beta_s\big|
$$

and compute bounds over $\{\theta : \sum_s |\hat\Gamma_s \theta - \hat\beta_s| \le (1 + \kappa)\hat Q\}$
for a small tolerance $\kappa$ (`criterion_tol`). Both problems are linear
programs. The *regression approach* replaces the IV-like moments by the
least-squares fit of the MTRs to the outcome, which turns the criterion
problem into a quadratic program and the bound problems into linearly
constrained problems with one quadratic constraint.

## Shape restrictions and the audit

Bounds on the MTRs or the MTE, and monotonicity in $u$, are linear
inequalities in $\theta$ that hold at every $(u, x)$. They can shrink the
identified set considerably. They are imposed on a grid of points and
verified on a finer grid by an iterative *audit*: points where a bounding
solution violates a restriction are added to the constraint set and the
problem is solved again ({doc}`guide/shape-restrictions`, {doc}`guide/audit`).

## Inference

Confidence regions for the bounds and intervals for point estimates come
from the nonparametric bootstrap; a bootstrap misspecification test checks
whether a positive minimum criterion is compatible with sampling noise
({doc}`guide/specification-tests`).

## References

- Heckman, J. J. and E. Vytlacil (2005). Structural Equations, Treatment
  Effects, and Econometric Policy Evaluation. *Econometrica* 73(3), 669-738.
- Imbens, G. W. and J. D. Angrist (1994). Identification and Estimation of
  Local Average Treatment Effects. *Econometrica* 62(2), 467-475.
- Mogstad, M., A. Santos and A. Torgovitsky (2018). Using Instrumental
  Variables for Inference About Policy Relevant Treatment Parameters.
  *Econometrica* 86(5), 1589-1619.
- Mogstad, M. and A. Torgovitsky (2018). Identification and Extrapolation
  of Causal Effects with Instrumental Variables. *Annual Review of
  Economics* 10, 577-613.
- Shea, J. and A. Torgovitsky (2023). ivmte: An R Package for Extrapolating
  Instrumental Variable Estimates Away From Compliers. *Observational
  Studies* 9(2), 1-42.
