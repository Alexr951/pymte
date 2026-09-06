"""Instrumental variables: extrapolation by marginal treatment effects.

This package is a Python port of the R package ``ivmte`` by Joshua Shea and
Alexander Torgovitsky. It implements the moment-based marginal treatment
effect framework of Mogstad, Santos and Torgovitsky (2018) for point and
partial identification of treatment parameters.
"""

from ivmte.datasets import load_ae, load_sim_data
from ivmte.estimate import ivmte
from ivmte.mtr import MTRSpec
from ivmte.propensity import Propensity, fit_propensity
from ivmte.results import IVMTEResult
from ivmte.splines import USpline

__all__ = [
    "IVMTEResult",
    "MTRSpec",
    "Propensity",
    "USpline",
    "fit_propensity",
    "ivmte",
    "load_ae",
    "load_sim_data",
]
__version__ = "0.1.0.dev0"
