# pymte

Instrumental variables and marginal treatment effects in Python.

`pymte` implements the moment-based marginal treatment effect (MTE)
framework of Mogstad, Santos and Torgovitsky (2018). Given a binary
treatment, an instrument and a specification of the marginal treatment
response (MTR) functions, it computes point estimates or sharp bounds for
target parameters such as the ATE, ATT, LATE or policy relevant treatment
effects, imposing shape restrictions where desired, and provides bootstrap
inference and specification tests.

The package is a port of the R package
[ivmte](https://github.com/jkcshea/ivmte) by Joshua Shea and Alexander
Torgovitsky (GPL-3). Estimands, options and numerical results match the R
package; the interface follows Python conventions.

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart
theory
```

```{toctree}
:maxdepth: 1
:caption: User guide

guide/index
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
migration
datasets
faq
citing
```
