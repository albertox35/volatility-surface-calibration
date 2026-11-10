"""Condiciones de no-arbitraje estatico sobre una superficie de volatilidad.

«Prevencion de arbitraje» no es una propiedad que se afirme: son dos condiciones
concretas y verificables sobre la varianza total implicita
:math:`w(k, T) = \\sigma_{BS}^2(k,T)\\, T`, con :math:`k = \\log(K/F)`.

**Mariposa (butterfly).** La densidad implicita de riesgo neutral tiene que ser no
negativa. Gatheral demuestra que eso equivale a :math:`g(k) \\geq 0`, con

.. math::

    g(k) = \\left(1 - \\frac{k\\, w'(k)}{2 w(k)}\\right)^2
         - \\frac{w'(k)^2}{4}\\left(\\frac{1}{w(k)} + \\frac{1}{4}\\right)
         + \\frac{w''(k)}{2}

porque la densidad es
:math:`p(k) = g(k)\\,\\varphi(d_2)/\\sqrt{2\\pi\\,w(k)}`. El signo de ``g`` **es**
el signo de la densidad: un ``g`` negativo significa una densidad negativa, es
decir una mariposa con precio negativo. Arbitraje puro.

**Calendario (calendar spread).** La varianza total tiene que ser no decreciente
en el vencimiento para cada moneyness fijo. Si no, un diferencial de calendario
tiene precio negativo.

Estas funciones operan sobre cualquier :math:`w` suave, no solo sobre SVI: sirven
igual para SABR, para una interpolacion o para una superficie cotizada.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = [
    "ArbitrageReport",
    "g_function",
    "implied_density",
    "check_butterfly",
    "check_calendar",
    "numerical_derivatives",
]


@dataclass(frozen=True)
class ArbitrageReport:
    """Resultado de verificar las condiciones de no-arbitraje."""

    butterfly_free: bool
    calendar_free: bool
    min_g: float
    """Minimo de la funcion ``g``; negativo implica arbitraje de mariposa."""

    worst_butterfly_k: float
    """Log-moneyness donde ``g`` alcanza su minimo."""

    min_calendar_slope: float
    """Minima pendiente de la varianza total en T; negativa implica arbitraje."""

    n_butterfly_violations: int = 0
    n_calendar_violations: int = 0

    @property
    def arbitrage_free(self) -> bool:
        """``True`` si la superficie pasa ambas condiciones."""
        return self.butterfly_free and self.calendar_free

    def __bool__(self) -> bool:
        return self.arbitrage_free

    def summary(self) -> str:
        """Resumen legible del diagnostico."""
        estado = "libre de arbitraje" if self.arbitrage_free else "CON ARBITRAJE"
        return (
            f"{estado}: mariposa {'ok' if self.butterfly_free else 'VIOLADA'} "
            f"(min g = {self.min_g:.6f} en k = {self.worst_butterfly_k:.4f}, "
            f"{self.n_butterfly_violations} nodos), "
            f"calendario {'ok' if self.calendar_free else 'VIOLADO'} "
            f"(min pendiente = {self.min_calendar_slope:.3e}, "
            f"{self.n_calendar_violations} nodos)"
        )


def g_function(
    k: np.ndarray,
    w: np.ndarray,
    dw_dk: np.ndarray,
    d2w_dk2: np.ndarray,
) -> np.ndarray:
    """Funcion ``g`` de Gatheral; su signo es el signo de la densidad implicita.

    Parameters
    ----------
    k:
        Log-moneyness ``log(K/F)``.
    w:
        Varianza total implicita, estrictamente positiva.
    dw_dk, d2w_dk2:
        Primera y segunda derivada de ``w`` respecto a ``k``.

    Returns
    -------
    numpy.ndarray
        ``g(k)``, del mismo tamanio que la entrada.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    dw = np.asarray(dw_dk, dtype=float)
    d2w = np.asarray(d2w_dk2, dtype=float)
    if np.any(w <= 0.0):
        raise ValueError("la varianza total implicita debe ser estrictamente positiva")

    termino1 = (1.0 - k * dw / (2.0 * w)) ** 2
    termino2 = (dw**2 / 4.0) * (1.0 / w + 0.25)
    return termino1 - termino2 + d2w / 2.0


def implied_density(
    k: np.ndarray,
    w: np.ndarray,
    dw_dk: np.ndarray,
    d2w_dk2: np.ndarray,
) -> np.ndarray:
    """Densidad implicita de riesgo neutral en log-moneyness.

    .. math:: p(k) = \\frac{g(k)}{\\sqrt{2\\pi w(k)}}
                     \\exp\\left(-\\frac{d_2(k)^2}{2}\\right)

    Que esta densidad integre a uno y sea no negativa es la forma mas directa de
    comprobar que una superficie es admisible.
    """
    g = g_function(k, w, dw_dk, d2w_dk2)
    w = np.asarray(w, dtype=float)
    k = np.asarray(k, dtype=float)
    d2 = -k / np.sqrt(w) - np.sqrt(w) / 2.0
    return g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d2**2)


def numerical_derivatives(
    w_fn: Callable[[np.ndarray], np.ndarray], k: np.ndarray, h: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivadas de ``w`` por diferencias centradas de cuarto orden.

    Se usa un esquema de cinco puntos en lugar de tres porque ``g`` depende de la
    segunda derivada, y un esquema de segundo orden introduce un error que puede
    fabricar violaciones de no-arbitraje inexistentes.

    Returns
    -------
    w, dw_dk, d2w_dk2
    """
    k = np.asarray(k, dtype=float)
    w_m2 = np.asarray(w_fn(k - 2.0 * h), dtype=float)
    w_m1 = np.asarray(w_fn(k - h), dtype=float)
    w_0 = np.asarray(w_fn(k), dtype=float)
    w_p1 = np.asarray(w_fn(k + h), dtype=float)
    w_p2 = np.asarray(w_fn(k + 2.0 * h), dtype=float)

    dw = (-w_p2 + 8.0 * w_p1 - 8.0 * w_m1 + w_m2) / (12.0 * h)
    d2w = (-w_p2 + 16.0 * w_p1 - 30.0 * w_0 + 16.0 * w_m1 - w_m2) / (12.0 * h**2)
    return w_0, dw, d2w


def check_butterfly(
    w_fn: Callable[[np.ndarray], np.ndarray],
    k_grid: np.ndarray | None = None,
    tol: float = -1e-10,
) -> tuple[bool, float, float, int]:
    """Verifica la condicion de mariposa sobre una malla de moneyness.

    Parameters
    ----------
    w_fn:
        Funcion ``k -> w(k)`` de varianza total implicita.
    k_grid:
        Malla de log-moneyness. Por omision cubre ``[-1.5, 1.5]``, que para
        vencimientos de mercado abarca con holgura los strikes cotizados.
    tol:
        Tolerancia negativa para absorber el error de redondeo.

    Returns
    -------
    libre, min_g, k_del_minimo, n_violaciones
    """
    if k_grid is None:
        k_grid = np.linspace(-1.5, 1.5, 601)
    k_grid = np.asarray(k_grid, dtype=float)
    w, dw, d2w = numerical_derivatives(w_fn, k_grid)
    g = g_function(k_grid, w, dw, d2w)
    idx = int(np.argmin(g))
    violaciones = int(np.sum(g < tol))
    return bool(g[idx] >= tol), float(g[idx]), float(k_grid[idx]), violaciones


def check_calendar(
    w_surface: Callable[[np.ndarray, float], np.ndarray],
    maturities: np.ndarray,
    k_grid: np.ndarray | None = None,
    tol: float = -1e-10,
) -> tuple[bool, float, int]:
    """Verifica que la varianza total no decrezca en el vencimiento.

    Parameters
    ----------
    w_surface:
        Funcion ``(k, T) -> w(k, T)``.
    maturities:
        Vencimientos a comprobar, en orden creciente.
    k_grid:
        Malla de log-moneyness.

    Returns
    -------
    libre, minima_pendiente, n_violaciones
    """
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 201)
    k_grid = np.asarray(k_grid, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    if maturities.size < 2:
        raise ValueError("se requieren al menos dos vencimientos")
    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError("los vencimientos deben venir en orden estrictamente creciente")

    superficie = np.array([w_surface(k_grid, float(t)) for t in maturities])
    pendientes = np.diff(superficie, axis=0) / np.diff(maturities)[:, None]
    minima = float(np.min(pendientes))
    violaciones = int(np.sum(pendientes < tol))
    return bool(minima >= tol), minima, violaciones
