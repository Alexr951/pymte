# Data provenance

Both files are exported unchanged (`write.csv(..., row.names = FALSE)`, then
gzip) from version 1.4.0 of the R package `ivmte`
(https://github.com/jkcshea/ivmte, GPL-3).

## ae.csv.gz

`AE` in the R package: a subsample of the 1980 Census extract used by Angrist
and Evans (1998, *American Economic Review* 88(3), 450-477), restricted to
women who were at least 20 at first birth and to eight columns:

| column     | meaning                                                          |
|------------|------------------------------------------------------------------|
| `worked`   | worked in the previous year                                      |
| `hours`    | weekly hours worked in the previous year                         |
| `morekids` | more than two children (versus exactly two)                      |
| `samesex`  | first two children have the same sex                             |
| `yob`      | mother's year of birth                                           |
| `black`    | mother is Black                                                  |
| `hisp`     | mother is Hispanic                                               |
| `other`    | mother is neither Black nor Hispanic                             |

The R package built it with `inst/extdata/AE.R`, starting from the cleaned
file distributed by Ivan Fernandez-Val
(http://sites.bu.edu/ivanf/files/2014/03/m_d_806.dta_.zip).

## ivmte_sim_data.csv.gz

`ivmteSimData` in the R package, generated in R by the following code
(`inst/extdata/ivmteSimData.R`). We do not re-simulate it in Python because
R and NumPy random streams differ.

```r
set.seed(1)
n <- 5000
u <- runif(n)
z <- rbinom(n, 3, .5)
x <- as.numeric(cut(rnorm(n), 10))
d <- as.numeric(u < z*.25 + .01*x)
v0 <- rnorm(n) + .2*u
y0 <- as.numeric(0 + v0 + .1*x > 0)
v1 <- rnorm(n) - .2*u
y1 <- as.numeric(.5 + v1 - .3*x > 0)
y <- d*y1 + (1-d)*y0
ivmteSimData <- data.frame(y, d, z, x)
```
