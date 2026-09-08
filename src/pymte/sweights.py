"""Weights of IV-like estimands.

By Mogstad, Santos and Torgovitsky (2018) every OLS or TSLS coefficient
``beta_k`` equals ``E[s_k(D, Z) Y]`` for a known weight ``s_k``. These
functions return ``s_k(0, Z_i)`` and ``s_k(1, Z_i)`` for the selected
coefficients, one column per component.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def olsj(
    x: NDArray[np.float64],
    x0: NDArray[np.float64],
    x1: NDArray[np.float64],
    components: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """OLS weights ``s_j(d, x) = e_j' E[XX']^{-1} x(d)``.

    Parameters
    ----------
    x : numpy.ndarray
        Regressor matrix.
    x0, x1 : numpy.ndarray
        ``x`` with the treatment fixed to 0 and to 1.
    components : sequence of int
        Column positions of the coefficients to use.

    Returns
    -------
    tuple of numpy.ndarray
        ``(s0, s1)``, each of shape ``(n, len(components))``.
    """
    n = len(x)
    exx_inv = np.linalg.inv(x.T @ x / n)
    cpos = list(components)
    return (x0 @ exx_inv)[:, cpos], (x1 @ exx_inv)[:, cpos]


def tsls(
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    z0: NDArray[np.float64],
    z1: NDArray[np.float64],
    components: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """TSLS weights ``s_j(d, z) = e_j' (Pi E[ZX'])^{-1} Pi z(d)`` with ``Pi = E[XZ'] E[ZZ']^{-1}``.

    Parameters
    ----------
    x : numpy.ndarray
        Regressor matrix.
    z, z0, z1 : numpy.ndarray
        Instrument matrix and its versions with the treatment fixed to 0 and 1.
    components : sequence of int
        Column positions of the coefficients to use.

    Returns
    -------
    tuple of numpy.ndarray
        ``(s0, s1)``, each of shape ``(n, len(components))``.
    """
    n = len(x)
    exz = x.T @ z / n
    pi = exz @ np.linalg.inv(z.T @ z / n)
    w = np.linalg.solve(pi @ exz.T, pi)  # K x L
    cpos = list(components)
    return (z0 @ w.T)[:, cpos], (z1 @ w.T)[:, cpos]
