"""Instrumental variables: extrapolation by marginal treatment effects.

This package is a Python port of the R package ``ivmte`` by Joshua Shea and
Alexander Torgovitsky. It implements the moment-based marginal treatment
effect framework of Mogstad, Santos and Torgovitsky (2018) for point and
partial identification of treatment parameters.
"""

from pymte.datasets import load_ae, load_sim_data
from pymte.estimate import ivmte
from pymte.mtr import MTRSpec
from pymte.plots import plot_mte, plot_mtr, plot_weights
from pymte.propensity import Propensity, fit_propensity
from pymte.results import IVMTEResult
from pymte.splines import USpline

__all__ = [
    "IVMTEResult",
    "MTRSpec",
    "Propensity",
    "USpline",
    "fit_propensity",
    "ivmte",
    "load_ae",
    "load_sim_data",
    "plot_mte",
    "plot_mtr",
    "plot_weights",
]
__version__ = "0.1.0.dev0"
