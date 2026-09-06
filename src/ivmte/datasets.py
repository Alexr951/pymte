"""Example datasets shipped with the package."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd


def _read(name: str) -> pd.DataFrame:
    with files("ivmte").joinpath("data").joinpath(name).open("rb") as fh:
        return pd.read_csv(fh, compression="gzip")


def load_ae() -> pd.DataFrame:
    """Load the Angrist and Evans (1998) subsample used in the R package.

    Returns
    -------
    pandas.DataFrame
        209,133 rows and 8 columns: ``worked``, ``hours``, ``morekids``,
        ``samesex``, ``yob``, ``black``, ``hisp`` and ``other``.

    Notes
    -----
    The data are women aged at least 20 at first birth from the 1980 Census
    extract of Angrist and Evans (1998), restricted to the columns needed for
    the examples. They are exported unchanged from the R package ``ivmte``;
    see ``ivmte/data/PROVENANCE.md`` for details.

    References
    ----------
    Angrist, J. D. and W. N. Evans (1998). Children and Their Parents' Labor
    Supply: Evidence from Exogenous Variation in Family Size. *American
    Economic Review* 88(3), 450-477.
    """
    return _read("ae.csv.gz")


def load_sim_data() -> pd.DataFrame:
    """Load the simulated dataset ``ivmteSimData`` from the R package.

    Returns
    -------
    pandas.DataFrame
        5,000 rows and 4 columns: binary outcome ``y``, binary treatment
        ``d``, instrument ``z`` in {0, 1, 2, 3} and covariate ``x`` in 1..10.

    Notes
    -----
    The data were generated in R with ``set.seed(1)``; the generating code is
    reproduced in ``ivmte/data/PROVENANCE.md``. We ship the R draw itself, so
    results match the R package exactly; NumPy and R random streams differ.
    """
    return _read("ivmte_sim_data.csv.gz")
