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
unobserved resistance to treatment: a unit takes treatment when its
resistance is below the propensity score, so smaller values of $U$ mean a
stronger latent willingness to be treated. The uniform distribution is a
normalisation, not an assumption. The model maintains that $Z$ is
independent of $(Y_0, Y_1, U)$ given $X$, which combines exogeneity of the
instrument with the exclusion restriction. Vytlacil (2002) showed that
under this independence the threshold-crossing rule is equivalent to the
monotonicity condition of Imbens and Angrist (1994): a shift in the
instrument moves every unit towards treatment or every unit away from it.

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
conventional parameter is a weighted average of the MTE with a weight in
$u$ that is either known or identified from the data.

With a continuous instrument the MTE is identified pointwise by the local
IV estimand of Heckman and Vytlacil (1999, 2005),

$$
\text{MTE}(u, x) = \frac{\partial}{\partial u}\, E[Y \mid p(X, Z) = u, X = x],
$$

for every $u$ in the interior of the support of the propensity score.
With a discrete instrument, the common case, the data only reveal
integrals of the MTE over intervals $(p(x, z_0), p(x, z_1)]$, and any
target that puts weight outside those intervals is not point identified
without further assumptions. This is the problem the moment-based
framework addresses: it makes the assumptions explicit through the
parameterisation of the MTRs and reports bounds when they do not suffice
for point identification.

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

MST (Proposition 1) show that the same representation holds for any
*IV-like estimand*: a cross moment $\beta_s = E[s(D, X, Z)\, Y]$ for a known
or identified function $s$, which includes every coefficient of a linear
regression of $Y$ on functions of $(D, X, Z)$, with or without
instruments. For such an estimand,

$$
\beta_s = \sum_j \theta_{0j}\, \gamma_{0j}(s) + \sum_j \theta_{1j}\, \gamma_{1j}(s),
$$

with $\gamma_{1j}(s) = E[s(1, X, Z)\int_0^{p(X,Z)} b_{1j}(u, X)\,du]$ and
$\gamma_{0j}(s) = E[s(0, X, Z)\int_{p(X,Z)}^1 b_{0j}(u, X)\,du]$. The weight
of an OLS coefficient is the corresponding row of the projection,
$s(d, x, z) = e_j' E[\tilde X \tilde X']^{-1} \tilde x(d)$ with $\tilde X$ the
regressor vector; TSLS coefficients have the analogous two-stage weight.
The data thus provide a system of linear equations in $\theta$, one per
IV-like estimand ({doc}`guide/ivlike`).

## Point and partial identification

If the system $\Gamma \theta = \beta$ has at least as many linearly
independent equations as unknowns, $\theta$ and hence $\beta^\star$ are
point identified; the package estimates $\theta$ by two-step GMM (or by
least squares in the regression approach) and reports a point estimate
with bootstrap intervals ({doc}`guide/confidence-intervals`). In the point
identified case the parameter space is all of $\mathbb R^K$: shape
restrictions are not imposed.

Otherwise the identified set for $\beta^\star$ is an interval obtained by
minimising and maximising $\gamma^{\star\prime}\theta$ over the
coefficients that fit the data. With sampling noise the system rarely has
an exact solution, so MST define the minimum criterion

$$
\hat Q = \min_\theta \sum_s \big|\hat\Gamma_s \theta - \hat\beta_s\big|
$$

and compute bounds over $\{\theta : \sum_s |\hat\Gamma_s \theta - \hat\beta_s| \le (1 + \kappa)\hat Q\}$
for a small tolerance $\kappa$ (`criterion_tol`). Both problems are linear
programs. MST give conditions under which the resulting interval is a
consistent estimate of the identified set; the tolerance is part of that
argument, which is why the default is positive.

The *regression approach* (Brinch, Mogstad and Wiswall 2017; Shea and
Torgovitsky 2023) replaces the IV-like moments by the conditional mean of
the outcome. Since

$$
E[Y \mid D = d, X = x, Z = z] = \int_0^1 m_d(u, x)\,\frac{\mathbf 1\{u \in I_d(p(x, z))\}}{P(D = d \mid X = x, Z = z)}\,du,
$$

with $I_1(p) = [0, p]$ and $I_0(p) = (p, 1]$, the MTR coefficients can be
fit by least squares of $Y$ on the generated regressors obtained by
integrating each basis function against these weights. The criterion is
then $\hat Q = \min_\theta \tfrac{1}{n}\sum_i (Y_i - \hat\Gamma_i'\theta)^2$, a
quadratic program, and the bound problems are linear objectives under one
quadratic constraint. The regression identified set is contained in the
moment identified set for any choice of IV-like estimands, and the
researcher need not choose estimands at all. Its minimum criterion is
never zero, however, so it cannot support a specification test.

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

- Andrews, D. W. K. and S. Han (2009). Invalidity of the Bootstrap and the
  m Out of n Bootstrap for Confidence Interval Endpoints Defined by Moment
  Inequalities. *Econometrics Journal* 12, S172-S199.
- Brinch, C. N., M. Mogstad and M. Wiswall (2017). Beyond LATE with a
  Discrete Instrument. *Journal of Political Economy* 125(4), 985-1039.
- Bugni, F. A., I. A. Canay and X. Shi (2015). Specification Tests for
  Partially Identified Models Defined by Moment Inequalities. *Journal of
  Econometrics* 185(1), 259-282.
- Cornelissen, T., C. Dustmann, A. Raute and U. Schönberg (2016). From LATE
  to MTE: Alternative Methods for the Evaluation of Policy Interventions.
  *Labour Economics* 41, 47-60.
- Hall, P. and J. L. Horowitz (1996). Bootstrap Critical Values for Tests
  Based on Generalized-Method-of-Moments Estimators. *Econometrica* 64(4),
  891-916.
- Heckman, J. J. and E. Vytlacil (1999). Local Instrumental Variables and
  Latent Variable Models for Identifying and Bounding Treatment Effects.
  *Proceedings of the National Academy of Sciences* 96(8), 4730-4734.
- Heckman, J. J. and E. Vytlacil (2005). Structural Equations, Treatment
  Effects, and Econometric Policy Evaluation. *Econometrica* 73(3), 669-738.
- Imbens, G. W. and J. D. Angrist (1994). Identification and Estimation of
  Local Average Treatment Effects. *Econometrica* 62(2), 467-475.
- Manski, C. F. and J. V. Pepper (2000). Monotone Instrumental Variables:
  With an Application to the Returns to Schooling. *Econometrica* 68(4),
  997-1010.
- Mogstad, M., A. Santos and A. Torgovitsky (2018). Using Instrumental
  Variables for Inference About Policy Relevant Treatment Parameters.
  *Econometrica* 86(5), 1589-1619.
- Mogstad, M. and A. Torgovitsky (2018). Identification and Extrapolation
  of Causal Effects with Instrumental Variables. *Annual Review of
  Economics* 10, 577-613.
- Shea, J. and A. Torgovitsky (2023). ivmte: An R Package for Extrapolating
  Instrumental Variable Estimates Away From Compliers. *Observational
  Studies* 9(2), 1-42.
- Vytlacil, E. (2002). Independence, Monotonicity, and Latent Index Models:
  An Equivalence Result. *Econometrica* 70(1), 331-341.
