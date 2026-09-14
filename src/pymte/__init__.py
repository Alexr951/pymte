"""Instrumental variables: extrapolation by marginal treatment effects.

This package is a Python port of the R package ``ivmte`` by Joshua Shea and
Alexander Torgovitsky. It implements the moment-based marginal treatment
effect framework of Mogstad, Santos and Torgovitsky (2018) for point and
partial identification of treatment parameters. The modules and functions
follow the layout and the names of the R package.
"""

from pymte.audit import audit, rhalton
from pymte.datasets import load_ae, load_sim_data
from pymte.design import design
from pymte.ivlike import iv_estimate
from pymte.lp import (
    bound,
    criterion_min,
    lp_setup,
    lp_setup_bound,
    lp_setup_criterion,
    lp_setup_criterion_boot,
    qp_setup,
    qp_setup_bound,
    qp_setup_criterion,
)
from pymte.mst import (
    IVMTEResult,
    bound_ci,
    bound_pvalue,
    gen_s_set,
    gen_target,
    gmm_estimate,
    ivmte,
    ivmte_estimate,
)
from pymte.mtr import MTRSpec, gen_gamma, polyparse
from pymte.plots import plot_mte, plot_mtr, plot_weights
from pymte.propensity import Propensity, propensity
from pymte.splines import USpline

__all__ = [
    "IVMTEResult",
    "MTRSpec",
    "Propensity",
    "USpline",
    "audit",
    "bound",
    "bound_ci",
    "bound_pvalue",
    "criterion_min",
    "design",
    "gen_gamma",
    "gen_s_set",
    "gen_target",
    "gmm_estimate",
    "iv_estimate",
    "ivmte",
    "ivmte_estimate",
    "load_ae",
    "load_sim_data",
    "lp_setup",
    "lp_setup_bound",
    "lp_setup_criterion",
    "lp_setup_criterion_boot",
    "plot_mte",
    "plot_mtr",
    "plot_weights",
    "polyparse",
    "propensity",
    "qp_setup",
    "qp_setup_bound",
    "qp_setup_criterion",
    "rhalton",
]
__version__ = "1.0.1"
