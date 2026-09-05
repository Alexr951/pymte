# ivmte

Instrumental variables and marginal treatment effects in Python.

`ivmte` estimates treatment parameters such as the ATE, ATT or a policy
relevant treatment effect from instrumental variables data, using the
marginal treatment effect (MTE) framework of Heckman and Vytlacil (2005) and
the moment-based implementation of Mogstad, Santos and Torgovitsky (2018).
The user specifies parametric or nonparametric (spline) marginal treatment
response functions, optional shape restrictions such as boundedness or
monotonicity, and a set of IV-like estimands. The package returns either a
point estimate (when the model is point identified) or sharp bounds obtained
by linear or quadratically constrained programming, together with bootstrap
confidence intervals and specification tests.

This package is a port of the R package
[`ivmte`](https://github.com/jkcshea/ivmte) by Joshua Shea and Alexander
Torgovitsky, released under the GPL-3. It reproduces the R package's
estimands, options and numerical results; the API follows Python conventions.

## Installation

```bash
pip install ivmte
```

Linear programs are solved with HiGHS (through SciPy) and quadratically
constrained programs with Clarabel (through CVXPY); both are installed
automatically. MOSEK and Gurobi can be used instead when licensed:

```bash
pip install "ivmte[mosek]"
pip install "ivmte[gurobi]"
pip install "ivmte[plots]"   # matplotlib for plot_mtr, plot_mte, plot_weights
```

## Quick start

```python
import ivmte

ae = ivmte.load_ae()
```

The full user guide, a theory primer and the API reference are on
[Read the Docs](https://ivmte.readthedocs.io).

## Citing

If you use this package, please cite the papers that developed the
methodology:

- Mogstad, M., A. Santos and A. Torgovitsky (2018). Using Instrumental
  Variables for Inference About Policy Relevant Treatment Parameters.
  *Econometrica* 86(5), 1589-1619. doi:10.3982/ECTA15463.
- Shea, J. and A. Torgovitsky (2023). ivmte: An R Package for Extrapolating
  Instrumental Variable Estimates Away From Compliers. *Observational
  Studies* 9(2), 1-42.
- Heckman, J. J. and E. Vytlacil (2005). Structural Equations, Treatment
  Effects, and Econometric Policy Evaluation. *Econometrica* 73(3), 669-738.
- Mogstad, M. and A. Torgovitsky (2018). Identification and Extrapolation of
  Causal Effects with Instrumental Variables. *Annual Review of Economics*
  10, 577-613.

BibTeX entries are in the documentation under "Citing". The `AE` dataset is
derived from Angrist and Evans (1998), *American Economic Review* 88(3).

## Licence

GPL-3.0-or-later. This package is a derivative work of the R package `ivmte`
(Copyright Joshua Shea and Alexander Torgovitsky), distributed under the same
licence. See `LICENSE` for the full text.
