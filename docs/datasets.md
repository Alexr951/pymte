---
file_format: mystnb
kernelspec:
  name: python3
---

# Datasets

Two datasets ship with the package, exported unchanged from the R package.

## Angrist and Evans (1998)

`load_ae()` returns 209,133 women from the 1980 Census extract of Angrist
and Evans (1998), *American Economic Review* 88(3), restricted to mothers
aged at least 20 at first birth. The treatment `morekids` indicates a third
child; `samesex` (the first two children have the same sex) is the
instrument.

| column | meaning |
|---|---|
| `worked` | worked in the previous year |
| `hours` | weekly hours worked in the previous year |
| `morekids` | more than two children |
| `samesex` | first two children have the same sex |
| `yob` | mother's year of birth |
| `black`, `hisp`, `other` | race and ethnicity indicators |

```{code-cell} python
import pymte

ae = pymte.load_ae()
ae.describe().round(3)
```

The R package built the file from the cleaned data distributed by Ivan
Fernandez-Val; see `PROVENANCE.md` in the package's data directory.

## Simulated data

`load_sim_data()` returns the 5,000 observations of `ivmteSimData`: a
binary outcome `y`, a binary treatment `d`, an instrument `z` with values 0
to 3 and a covariate `x` with values 1 to 10. The data were generated in R
with `set.seed(1)`; the generating code is reproduced in `PROVENANCE.md`.
We ship the R draw so that results match the R package exactly.

```{code-cell} python
sim = pymte.load_sim_data()
sim.groupby("z")[["d", "y"]].mean().round(3)
```
