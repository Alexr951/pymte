"""B-spline bases in the unobservable ``u`` and their exact integrals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class USpline:
    """A B-spline basis on [0, 1] with fixed boundary knots 0 and 1.

    Parameters
    ----------
    degree : int
        Polynomial degree of the pieces. Degree 0 gives piecewise constants.
    knots : sequence of float
        Interior knots. Values at or outside the boundaries are dropped, as in
        the R package.
    intercept : bool, default False
        Whether to keep the first basis function. With ``intercept=False`` the
        basis has ``degree + len(knots)`` columns and does not span constants,
        matching ``splines2::bSpline`` and ``splines::bs``.

    Notes
    -----
    The basis and its antiderivative are taken from
    :class:`scipy.interpolate.BSpline` on the knot vector
    ``[0] * (degree + 1) + knots + [1] * (degree + 1)``. This reproduces the
    column convention of ``splines2::bSpline``/``splines2::ibs`` used by the
    R package, including the ``intercept`` handling.
    """

    degree: int
    knots: tuple[float, ...] = ()
    intercept: bool = False
    _spline: BSpline = field(init=False, repr=False, compare=False)
    _integral: BSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.degree) != self.degree or self.degree < 0:
            raise ValueError("uSplines: 'degree' must be a non-negative integer")
        knots = tuple(sorted(float(k) for k in self.knots if 0 < k < 1))
        object.__setattr__(self, "degree", int(self.degree))
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "intercept", bool(self.intercept))
        k = self.degree
        t = np.r_[np.zeros(k + 1), knots, np.ones(k + 1)]
        n = len(t) - k - 1
        spline = BSpline(t, np.eye(n), k, extrapolate=False)
        object.__setattr__(self, "_spline", spline)
        object.__setattr__(self, "_integral", spline.antiderivative())

    @property
    def n_basis(self) -> int:
        """Number of basis functions (columns)."""
        return self.degree + len(self.knots) + int(self.intercept)

    def _drop_first(self, mat: NDArray[np.float64]) -> NDArray[np.float64]:
        return mat if self.intercept else mat[..., 1:]

    def basis(self, u: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the basis functions at ``u``.

        Parameters
        ----------
        u : array_like
            Points in [0, 1].

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(u), n_basis)``.
        """
        x = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        # The last interval is closed at 1 so that the basis sums to one there.
        out = np.asarray(self._spline(np.where(x >= 1.0, np.nextafter(1.0, 0.0), x)), dtype=float)
        return self._drop_first(out)

    def integral(self, lb: ArrayLike, ub: ArrayLike) -> NDArray[np.float64]:
        """Integrate each basis function from ``lb`` to ``ub``.

        Parameters
        ----------
        lb, ub : array_like
            Integration limits in [0, 1], broadcast against each other.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n, n_basis)`` with ``n`` the broadcast length.
        """
        lo, hi = np.broadcast_arrays(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))
        lo = np.clip(lo, 0.0, 1.0)
        hi = np.clip(hi, 0.0, 1.0)
        out = np.asarray(self._integral(hi) - self._integral(lo), dtype=float)
        return self._drop_first(out)
