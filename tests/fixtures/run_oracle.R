# Reference results from the R package ivmte, used as parity fixtures for the
# Python port. Run with
#
#     Rscript run_oracle.R <output directory>
#
# Requires ivmte (>= 1.4.0), lpSolveAPI, splines2 and jsonlite. Every model is
# solved with lpSolveAPI; the quadratically constrained problems of the
# regression approach need Gurobi or MOSEK and are therefore not generated
# here (their published values are hard-coded in the Python tests).

suppressPackageStartupMessages({
  library(ivmte)
  library(jsonlite)
  library(splines2)
})

cli <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(cli) >= 1) cli[1] else "oracle-out"
FILTER <- if (length(cli) >= 2) cli[2] else ""
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
SOLVER <- "lpsolveapi"
SEED <- 1234L

## ---------------------------------------------------------------------------
## Helpers
## ---------------------------------------------------------------------------

named <- function(v) {
  if (is.null(v)) return(NULL)
  if (is.matrix(v)) return(list(rownames = rownames(v), colnames = colnames(v),
                                values = unname(v)))
  if (is.null(names(v))) return(unname(v))
  as.list(v)
}

col_means <- function(g) {
  if (is.null(g)) return(NULL)
  if (is.matrix(g)) return(named(colMeans(g)))
  named(g)
}

extract <- function(r) {
  out <- list()
  if (!is.null(r$bounds)) {
    out$bounds <- unname(r$bounds)
    out$audit_count <- r$audit.count
    out$audit_criterion <- r$audit.criterion
    out$audit_criterion_status <- r$audit.criterion.status
    out$gstar_coef <- lapply(r$gstar.coef, named)
    out$audit_grid <- list(audit_x = r$audit.grid$audit.x,
                           audit_u = r$audit.grid$audit.u,
                           violations = r$audit.grid$violations)
    out$solver <- r$solver
  }
  if (!is.null(r$point.estimate)) {
    out$point_estimate <- unname(r$point.estimate)
    out$mtr_coef <- named(r$mtr.coef)
    out$j_test <- named(r$j.test)
    out$redundant <- r$redundant
    if (is.list(r$moments)) {
      out$moments_count <- r$moments$count
      out$moments_criterion <- named(r$moments$criterion)
    }
    out$ssr <- r$SSR
  }
  if (is.numeric(r$moments)) out$moments <- r$moments
  if (!is.null(r$gstar)) {
    out$gstar <- list(g0 = named(r$gstar$g0), g1 = named(r$gstar$g1), n = r$gstar$n)
  }
  if (!is.null(r$gstar.weights)) {
    out$gstar_weights <- lapply(r$gstar.weights, function(w) {
      list(lb = head(w$lb, 30), ub = head(w$ub, 30), multiplier = head(w$multiplier, 30))
    })
  }
  if (!is.null(r$s.set)) {
    out$s_set <- lapply(r$s.set, function(s) {
      list(ivspec = s$ivspec, beta = unname(s$beta), n = s$n,
           g0 = col_means(s$g0), g1 = col_means(s$g1))
    })
  }
  if (!is.null(r$propensity)) {
    m <- r$propensity$model
    out$propensity <- list(
      coef = if (!is.null(m)) named(coef(m)) else NULL,
      phat_head = head(unname(r$propensity$phat), 50),
      phat_mean = mean(r$propensity$phat))
  }
  if (!is.null(r$splines.dict)) out$splines_dict <- r$splines.dict
  ## bootstrap fields
  for (f in c("bounds.se", "bounds.bootstraps", "bounds.ci", "p.value",
              "specification.p.value", "bootstraps", "bootstraps.failed",
              "point.estimate.se", "point.estimate.bootstraps",
              "point.estimate.ci", "mtr.se", "mtr.bootstraps", "j.test.bootstraps")) {
    if (!is.null(r[[f]])) {
      v <- r[[f]]
      key <- gsub("\\.", "_", f)
      if (is.list(v) && !is.data.frame(v)) out[[key]] <- lapply(v, named)
      else out[[key]] <- named(v)
    }
  }
  out$messages <- r$messages
  out
}

run_case <- function(name, args) {
  cat(sprintf("== %s\n", name))
  args$solver <- SOLVER
  args$noisy <- FALSE
  warnings <- character(0)
  set.seed(SEED)
  res <- withCallingHandlers(
    tryCatch(do.call(ivmte, args), error = function(e) e),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  out <- list(name = name, seed = SEED, warnings = warnings)
  if (inherits(res, "error")) {
    out$error <- conditionMessage(res)
    cat("   ERROR:", out$error, "\n")
  } else {
    out <- c(out, extract(res))
    if (!is.null(res$bounds)) {
      cat(sprintf("   bounds [%.8g, %.8g]  audits %d\n", res$bounds[1], res$bounds[2],
                  res$audit.count))
    } else {
      cat(sprintf("   point %.8g\n", res$point.estimate))
    }
  }
  write_json(out, file.path(outdir, paste0(name, ".json")), auto_unbox = TRUE,
             digits = NA, null = "null", na = "null", pretty = TRUE)
  invisible(res)
}

## ---------------------------------------------------------------------------
## Datasets (exported unchanged)
## ---------------------------------------------------------------------------

write.csv(AE, file.path(outdir, "ae.csv"), row.names = FALSE)
write.csv(ivmteSimData, file.path(outdir, "ivmte_sim_data.csv"), row.names = FALSE)

## ---------------------------------------------------------------------------
## Spline bases from splines2 (boundary knots 0 and 1, as in ivmte)
## ---------------------------------------------------------------------------

spline_specs <- list(
  list(degree = 1, knots = c(.2, .4, .6, .8), intercept = FALSE),
  list(degree = 2, knots = c(.1, .3, .5, .7), intercept = FALSE),
  list(degree = 1, knots = c(.25, .5, .75), intercept = FALSE),
  list(degree = 3, knots = c(.25, .5, .75), intercept = FALSE),
  list(degree = 2, knots = c(1 / 3, 2 / 3), intercept = FALSE),
  list(degree = 2, knots = c(.3, .6), intercept = FALSE),
  list(degree = 0, knots = c(.2, .5, .8), intercept = TRUE),
  list(degree = 1, knots = c(.4), intercept = TRUE),
  list(degree = 0, knots = c(.5), intercept = FALSE),
  list(degree = 3, knots = numeric(0), intercept = TRUE)
)
ugrid <- c(seq(0, 1, by = 0.01), 0.123456789, 0.987654321, 1 / 3, 2 / 3)
splines_out <- lapply(spline_specs, function(s) {
  basis <- splines2::bSpline(ugrid, knots = s$knots, degree = s$degree,
                             intercept = s$intercept, Boundary.knots = c(0, 1))
  integral <- splines2::ibs(ugrid, knots = s$knots, degree = s$degree,
                            intercept = s$intercept, Boundary.knots = c(0, 1))
  list(degree = s$degree, knots = s$knots, intercept = s$intercept, u = ugrid,
       basis = unname(unclass(basis)), integral = unname(unclass(integral)))
})
write_json(splines_out, file.path(outdir, "splines.json"), digits = NA, pretty = TRUE,
           auto_unbox = TRUE)

## Halton sequence used for the u grids
write_json(list(halton_25 = rhalton(25), halton_20 = rhalton(20), halton_5 = rhalton(5),
                halton_4 = rhalton(4), halton_1 = rhalton(1)),
           file.path(outdir, "halton.json"), digits = NA, pretty = TRUE)

## ---------------------------------------------------------------------------
## Vignette examples (Angrist-Evans data)
## ---------------------------------------------------------------------------

ae_ivlike <- worked ~ morekids + samesex + morekids * samesex
ae_prop <- morekids ~ samesex + yob

run_case("ae_att_linear_u", list(
  data = AE, target = "att", m0 = ~ u + yob, m1 = ~ u + yob,
  ivlike = ae_ivlike, propensity = ae_prop))

run_case("ae_att_linear_u_probit", list(
  data = AE, target = "att", m0 = ~ u + yob, m1 = ~ u + yob,
  ivlike = ae_ivlike, propensity = ae_prop, link = "probit"))

run_case("ae_att_linear_u_linear_link", list(
  data = AE, target = "att", m0 = ~ u + yob, m1 = ~ u + yob,
  ivlike = ae_ivlike, propensity = ae_prop, link = "linear"))

run_case("ae_att_regression_point", list(
  data = AE, target = "att", m0 = ~ u + yob, m1 = ~ u + yob,
  outcome = "worked", propensity = ae_prop))

run_case("ae_att_poly", list(
  data = AE, ivlike = ae_ivlike, target = "att",
  m0 = ~ u + I(u^2) + yob + u * yob,
  m1 = ~ u + I(u^2) + I(u^3) + yob + u * yob,
  propensity = ae_prop))

run_case("ae_ate_poly", list(
  data = AE, ivlike = ae_ivlike, target = "ate",
  m0 = ~ u + I(u^2) + yob + u * yob,
  m1 = ~ u + I(u^2) + I(u^3) + yob + u * yob,
  propensity = ae_prop))

run_case("ae_atu_poly", list(
  data = AE, ivlike = ae_ivlike, target = "atu",
  m0 = ~ u + I(u^2) + yob + u * yob,
  m1 = ~ u + I(u^2) + I(u^3) + yob + u * yob,
  propensity = ae_prop))

run_case("ae_att_poly_uname_v", list(
  data = AE, ivlike = ae_ivlike, target = "att",
  m0 = ~ v + I(v^2) + yob + v * yob,
  m1 = ~ v + I(v^2) + I(v^3) + yob + v * yob,
  uname = "v", propensity = ae_prop))

run_case("ae_att_boolean_terms", list(
  data = AE, ivlike = ae_ivlike, target = "att",
  m0 = ~ u + yob, m1 = ~ u + (yob == 55) + (yob == 60),
  propensity = ae_prop))

run_case("ae_att_splines", list(
  data = AE, ivlike = ae_ivlike, target = "att",
  m0 = ~ u + uSplines(degree = 1, knots = c(.2, .4, .6, .8)) + yob,
  m1 = ~ uSplines(degree = 2, knots = c(.1, .3, .5, .7)) * yob,
  propensity = ae_prop))

run_case("ae_att_splines_shape", list(
  data = AE, ivlike = ae_ivlike, target = "att",
  m0 = ~ u + uSplines(degree = 1, knots = c(.2, .4, .6, .8)) + yob,
  m1 = ~ uSplines(degree = 2, knots = c(.1, .3, .5, .7)) * yob,
  m1.inc = TRUE, m0.inc = TRUE, mte.dec = TRUE,
  propensity = ae_prop))

run_case("ae_att_splines_noint_shape", list(
  data = AE, ivlike = ae_ivlike, target = "att",
  m0 = ~ 0 + uSplines(degree = 2, knots = c(1 / 3, 2 / 3)),
  m1 = ~ 0 + uSplines(degree = 2, knots = c(1 / 3, 2 / 3)),
  m1.inc = TRUE, m0.inc = TRUE, mte.dec = TRUE,
  propensity = morekids ~ samesex))

run_case("ae_att_point_gmm", list(
  data = AE, target = "att", m0 = ~ u, m1 = ~ u,
  ivlike = ae_ivlike, propensity = morekids ~ samesex, point = TRUE))

## ---------------------------------------------------------------------------
## Vignette examples (simulated data)
## ---------------------------------------------------------------------------

sim_cubic <- ~ u + I(u^2) + I(u^3) + x

run_case("sim_equal_coef_regression_point", list(
  data = ivmteSimData, outcome = "y", target = "ate",
  m0 = ~ x + u, m1 = ~ x + u, equal.coef = ~ 0 + x,
  propensity = d ~ x + factor(z)))

run_case("sim_late", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z, target = "late",
  late.from = c(z = 1), late.to = c(z = 3),
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x))

run_case("sim_late_x2", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z, target = "late",
  late.from = c(z = 1), late.to = c(z = 3), late.X = c(x = 2),
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x))

run_case("sim_late_x8", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z, target = "late",
  late.from = c(z = 1), late.to = c(z = 3), late.X = c(x = 8),
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x))

run_case("sim_genlate", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z, target = "genlate",
  genlate.lb = .2, genlate.ub = .42,
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x))

r_genlate_x2 <- run_case("sim_genlate_x2", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z, target = "genlate",
  genlate.lb = .2, genlate.ub = .42, late.X = c(x = 2),
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x))

## Custom weights replicating the conditional LATE with x = 2
pmodel <- r_genlate_x2$propensity$model
xeval <- 2
px <- mean(ivmteSimData$x == xeval)
z.from <- 1
z.to <- 3
weight1 <- function(x) {
  if (x != xeval) return(0)
  p.from <- predict(pmodel, newdata = data.frame(x = xeval, z = z.from), type = "response")
  p.to <- predict(pmodel, newdata = data.frame(x = xeval, z = z.to), type = "response")
  1 / ((p.to - p.from) * px)
}
weight0 <- function(x) -weight1(x)
knot1 <- function(x) predict(pmodel, newdata = data.frame(x = x, z = z.from), type = "response")
knot2 <- function(x) predict(pmodel, newdata = data.frame(x = x, z = z.to), type = "response")

run_case("sim_custom_weights_late_x2", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z,
  target.knots0 = c(knot1, knot2), target.knots1 = c(knot1, knot2),
  target.weight0 = c(0, weight0, 0), target.weight1 = c(0, weight1, 0),
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x))

sim_multi_ivlike <- c(y ~ (z == 1) + (z == 2) + (z == 3) + x, y ~ d + x, y ~ d | z)
sim_spl1 <- ~ uSplines(degree = 1, knots = c(.25, .5, .75)) + x

run_case("sim_multi_ivlike", list(
  data = ivmteSimData, ivlike = sim_multi_ivlike, target = "ate",
  m0 = sim_spl1, m1 = sim_spl1, propensity = d ~ z + x))

run_case("sim_multi_ivlike_components", list(
  data = ivmteSimData, ivlike = sim_multi_ivlike, target = "ate",
  components = l(c(intercept, x), c(d), ),
  m0 = sim_spl1, m1 = sim_spl1, propensity = d ~ z + x))

sim_spl3 <- ~ uSplines(degree = 3, knots = c(.25, .5, .75)) + x
run_case("sim_multi_ivlike_subset", list(
  data = ivmteSimData, ivlike = c(y ~ z + x, y ~ d + x, y ~ d | z),
  subset = l(x <= 9, 1 == 1, z %in% c(1, 3)), target = "ate",
  m0 = sim_spl3, m1 = sim_spl3, propensity = d ~ z + x))

run_case("sim_point_gmm", list(
  data = ivmteSimData, ivlike = y ~ d + factor(z), target = "ate",
  m0 = ~ u, m1 = ~ u, propensity = d ~ factor(z), point = TRUE))

run_case("sim_point_gmm_eyeweight", list(
  data = ivmteSimData, ivlike = y ~ d + factor(z), target = "ate",
  m0 = ~ u, m1 = ~ u, propensity = d ~ factor(z), point = TRUE,
  point.eyeweight = TRUE))

run_case("sim_point_lp_tol0", list(
  data = ivmteSimData, ivlike = y ~ d + factor(z), target = "ate",
  m0 = ~ u, m1 = ~ u, propensity = d ~ factor(z), point = FALSE,
  criterion.tol = 0))

run_case("sim_point_regression_ols", list(
  data = ivmteSimData, outcome = "y", target = "ate",
  m0 = ~ u, m1 = ~ u, propensity = d ~ factor(z), point = TRUE))

run_case("sim_point_regression_equal_coef", list(
  data = ivmteSimData, outcome = "y", target = "ate",
  m0 = ~ u, m1 = ~ u, propensity = d ~ factor(z),
  equal.coef = ~ 0 + u, point = TRUE))

run_case("sim_point_gmm_many_moments", list(
  data = ivmteSimData, ivlike = y ~ d + factor(z) + d * factor(z), target = "ate",
  m0 = ~ u, m1 = ~ u, propensity = d ~ factor(z), point = TRUE))

run_case("sim_shape_ignored_point", list(
  data = ivmteSimData, ivlike = y ~ d + factor(z), target = "ate",
  m0 = ~ u, m1 = ~ u, m0.dec = TRUE, m1.dec = TRUE, propensity = d ~ factor(z)))

## ---------------------------------------------------------------------------
## Bootstrap examples (RNG dependent; compared distributionally)
## ---------------------------------------------------------------------------

run_case("ae_att_linear_u_boot50", list(
  data = AE, target = "att", m0 = ~ u + yob, m1 = ~ u + yob,
  ivlike = ae_ivlike, propensity = ae_prop, bootstraps = 50))

run_case("ae_att_point_gmm_boot50", list(
  data = AE, target = "att", m0 = ~ u, m1 = ~ u,
  ivlike = ae_ivlike, propensity = morekids ~ samesex, point = TRUE,
  bootstraps = 50))

run_case("sim_spec_test_boot50", list(
  data = ivmteSimData, ivlike = y ~ d + factor(z), target = "ate",
  m0 = ~ u, m1 = ~ u, m0.dec = TRUE, m1.dec = TRUE,
  propensity = d ~ factor(z), bootstraps = 50))

run_case("sim_late_boot50", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z, target = "late",
  late.from = c(z = 1), late.to = c(z = 3),
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x, bootstraps = 50))

run_case("sim_late_boot50_m2000_subsample", list(
  data = ivmteSimData, ivlike = y ~ d + z + d * z, target = "late",
  late.from = c(z = 1), late.to = c(z = 3),
  m0 = sim_cubic, m1 = sim_cubic, propensity = d ~ z + x, bootstraps = 50,
  bootstraps.m = 2000, bootstraps.replace = FALSE))

## ---------------------------------------------------------------------------
## testthat scenarios (internal simulated distributions). Explicit grids are
## passed so that the Python tests can use exactly the same grid points.
## ---------------------------------------------------------------------------

dtcf <- ivmte:::gendistCovariates()$data.full
dtc <- ivmte:::gendistCovariates()$data.dist
write.csv(dtcf, file.path(outdir, "dist_covariates_full.csv"), row.names = FALSE)
write.csv(dtc, file.path(outdir, "dist_covariates_dist.csv"), row.names = FALSE)
support_c <- unique(dtcf[, c("x1", "x2")])
rownames(support_c) <- NULL

run_case("tt_single", list(
  ivlike = ey ~ 1 + d + x1 + x2, data = dtcf, components = l(c(d, x1)),
  subset = l(z2 %in% c(2, 3)), propensity = d ~ x1 + x2 + z1 + z2, link = "logit",
  m0 = ~ x1 + x2:u + x2:I(u^2), m1 = ~ x1 + x1:x2 + u + x1:u + x2:I(u^2),
  uname = "u", target = "late",
  late.from = c(z1 = 1, z2 = 2), late.to = c(z1 = 0, z2 = 3), late.X = c(x1 = 0, x2 = 1),
  criterion.tol = 0.01, initgrid.nu = 4, audit.nu = 5,
  initgrid.x = support_c[1:2, ], audit.x = support_c[1:5, ]))

run_case("tt_covariates", list(
  ivlike = c(ey ~ d, ey ~ d + x1, ey ~ d + x1 + x2,
             ey ~ d + x1 + x2 | x1 + x2 + z1 + z2, ey ~ d | factor(z2)),
  data = dtcf, components = l(d, d, c(d, x1, x2), d, d),
  subset = l(, , z2 %in% c(2, 3), , z2 %in% c(2, 3)),
  propensity = ~ p, m0 = ~ x1 + x2:u + x2:I(u^2), m1 = ~ x1 + x1:x2 + u + x1:u + x2:I(u^2),
  uname = "u", target = "genlate", genlate.lb = 0.2, genlate.ub = 0.7,
  criterion.tol = 0.01, initgrid.nu = 1, audit.nu = 5,
  initgrid.x = support_c[1:2, ], audit.x = support_c[1:3, ],
  m0.inc = TRUE, m1.inc = TRUE, mte.dec = TRUE, treat = "d"))

dtsf <- ivmte:::gendistSplines()$data.full
dts <- ivmte:::gendistSplines()$data.dist
write.csv(dtsf, file.path(outdir, "dist_splines_full.csv"), row.names = FALSE)
write.csv(dts, file.path(outdir, "dist_splines_dist.csv"), row.names = FALSE)
support_s <- data.frame(x = unique(dtsf$x))

spl_m1 <- ~ x + uSplines(degree = 2, knots = c(0.3, 0.6), intercept = FALSE)
spl_m0 <- ~ 0 + x:uSplines(degree = 0, knots = c(0.2, 0.5, 0.8), intercept = TRUE) +
  uSplines(degree = 1, knots = c(0.4), intercept = TRUE) + I(u^2)

run_case("tt_splines", list(
  ivlike = c(ey ~ d, ey ~ d + x, ey ~ d + x | z + x), data = dtsf,
  components = l(c(intercept, d), d, c(d, x)), propensity = ~ p, treat = "d",
  m1 = spl_m1, m0 = spl_m0, uname = "u", target = "att",
  criterion.tol = 0.01, initgrid.nu = 25, audit.nu = 25,
  initgrid.x = support_s, audit.x = support_s,
  m1.ub = 55, m0.lb = 0, mte.inc = TRUE))

weight11 <- function(z) 1 / (1 + z)
weight01 <- function(z, x) -1 / (1 + z + abs(x))
weight02 <- function(z) -(1 / (1 + z)) * 1.33
knots01 <- function(z, x) 0.3 + 0.3 * z + 0.1 * x

run_case("tt_splines_custom_weights", list(
  ivlike = c(ey ~ d, ey ~ d + x, ey ~ d + x | z + x), data = dtsf,
  components = l(c(intercept, d), d, c(d, x)), propensity = ~ p, treat = "d",
  m1 = spl_m1, m0 = spl_m0, uname = "u",
  target.weight0 = c(weight01, weight02), target.weight1 = weight11,
  target.knots0 = knots01,
  criterion.tol = 0.01, initgrid.x = support_s, audit.x = support_s,
  m1.ub = 55, m0.lb = 0, mte.inc = TRUE, mte.ub = 10))

cat("done\n")
