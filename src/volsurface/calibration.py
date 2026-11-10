"""Calibracion por minimos cuadrados no lineales con restricciones.

El problema **no** es «minimizacion de restricciones no lineales», que no
significa nada: es un problema de **minimos cuadrados no lineales con
restricciones de caja**, y eso determina el algoritmo. Se usa la region de
confianza reflectiva de SciPy (``trust-constr`` interior para cajas, es decir
``method="trf"`` en ``least_squares``), que respeta las cotas en cada iterado en
lugar de proyectar al final.

Dos decisiones que cambian el resultado mas que el optimizador:

**Se ajusta la varianza total, no la volatilidad.** La varianza total es la
cantidad en la que SVI es lineal por tramos asintoticamente y en la que se
expresan las condiciones de no-arbitraje. Ajustar volatilidades pondera de forma
implicita los vencimientos cortos.

**Se pondera por vega.** Un error de un punto de volatilidad vale dinero muy
distinto en el dinero que en el ala, porque la vega difiere en un orden de
magnitud. Ajustar sin ponderar equivale a declarar que todos los strikes
importan igual, lo que ningun operador aceptaria. La ponderacion por vega es la
opcion por omision, y se puede desactivar de forma explicita.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from .sabr import SABRParams, hagan_implied_volatility
from .svi import SVIParams

__all__ = ["CalibrationResult", "calibrate_svi_slice", "calibrate_sabr_slice", "black_vega"]


@dataclass(frozen=True)
class CalibrationResult:
    """Resultado de una calibracion."""

    params: object
    rmse_volatility: float
    """Error cuadratico medio en puntos de volatilidad."""

    max_abs_error: float
    """Maximo error absoluto en volatilidad."""

    n_iterations: int
    success: bool
    message: str
    arbitrage_free: bool | None = None
    min_g: float | None = None

    def __str__(self) -> str:
        arb = "" if self.arbitrage_free is None else (
            f", {'libre de arbitraje' if self.arbitrage_free else 'CON ARBITRAJE'}"
            f" (min g = {self.min_g:.2e})"
        )
        return (
            f"RMSE {self.rmse_volatility * 100:.4f} puntos de vol, "
            f"max {self.max_abs_error * 100:.4f}, "
            f"{self.n_iterations} iteraciones{arb}"
        )


def black_vega(
    forward: float, strike: np.ndarray, maturity: float, vol: np.ndarray
) -> np.ndarray:
    """Vega de Black respecto a la volatilidad, sin descontar.

    Se usa solo como peso relativo, asi que el factor de descuento es irrelevante
    y se omite.
    """
    strike = np.asarray(strike, dtype=float)
    vol = np.asarray(vol, dtype=float)
    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * vol**2 * maturity) / (vol * sqrt_t)
    return forward * norm.pdf(d1) * sqrt_t


def _weights(
    forward: float,
    strikes: np.ndarray,
    maturity: float,
    vols: np.ndarray,
    vega_weighted: bool,
) -> np.ndarray:
    if not vega_weighted:
        return np.ones_like(np.asarray(strikes, dtype=float))
    vega = black_vega(forward, strikes, maturity, vols)
    # Normalizado para que la escala del residuo no dependa del nivel del forward.
    return vega / np.max(vega)


def calibrate_svi_slice(
    log_moneyness: np.ndarray,
    market_volatility: np.ndarray,
    maturity: float,
    forward: float = 1.0,
    vega_weighted: bool = True,
    initial: SVIParams | None = None,
    max_nfev: int = 5000,
) -> CalibrationResult:
    """Calibra un corte SVI crudo a volatilidades cotizadas.

    Las cotas de los parametros no son arbitrarias: codifican las condiciones
    **necesarias** de admisibilidad (``b >= 0``, ``|rho| < 1``, ``sigma > 0``),
    de modo que el optimizador no puede visitar una region donde la varianza
    total seria negativa.

    Parameters
    ----------
    log_moneyness:
        ``log(K/F)`` de cada cotizacion.
    market_volatility:
        Volatilidades implicitas cotizadas.
    maturity:
        Plazo del corte.
    forward:
        Forward del subyacente, necesario para la ponderacion por vega.
    vega_weighted:
        Ponderar los residuos por vega.
    initial:
        Punto de partida. Si es ``None`` se usa una heuristica basada en el
        nivel y la pendiente observados.

    Returns
    -------
    CalibrationResult
        Incluye el diagnostico de no-arbitraje del ajuste resultante, porque un
        ajuste con error pequenio que admite arbitraje es inservible.
    """
    k = np.asarray(log_moneyness, dtype=float)
    vol = np.asarray(market_volatility, dtype=float)
    if k.shape != vol.shape:
        raise ValueError("log_moneyness y market_volatility deben tener la misma forma")
    if k.size < 5:
        raise ValueError(f"SVI tiene 5 parametros; se necesitan al menos 5 cotizaciones, hay {k.size}")
    if maturity <= 0.0:
        raise ValueError("el vencimiento debe ser positivo")
    if np.any(vol <= 0.0):
        raise ValueError("las volatilidades cotizadas deben ser positivas")

    w_market = vol**2 * maturity
    pesos = _weights(forward, forward * np.exp(k), maturity, vol, vega_weighted)

    if initial is None:
        nivel = float(np.min(w_market))
        initial = SVIParams(a=0.5 * nivel, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    x0 = np.array([initial.a, initial.b, initial.rho, initial.m, initial.sigma])

    # Cotas que codifican las condiciones necesarias de admisibilidad.
    lower = np.array([-np.inf, 0.0, -0.999, -2.0, 1e-6])
    upper = np.array([np.inf, 10.0, 0.999, 2.0, 5.0])

    def residuos(x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = x
        y = k - m
        w = a + b * (rho * y + np.sqrt(y**2 + sigma**2))
        # Penalizar varianzas negativas en lugar de dejar que produzcan nan.
        w = np.where(w > 1e-12, w, 1e-12)
        return pesos * (w - w_market) / maturity

    salida = least_squares(
        residuos, x0, bounds=(lower, upper), method="trf", max_nfev=max_nfev
    )
    params = SVIParams(
        a=float(salida.x[0]),
        b=float(salida.x[1]),
        rho=float(salida.x[2]),
        m=float(salida.x[3]),
        sigma=float(salida.x[4]),
    )

    ajustada = params.implied_volatility(k, maturity)
    err = ajustada - vol
    libre, min_g = params.is_butterfly_free()
    return CalibrationResult(
        params=params,
        rmse_volatility=float(np.sqrt(np.mean(err**2))),
        max_abs_error=float(np.max(np.abs(err))),
        n_iterations=int(salida.nfev),
        success=bool(salida.success),
        message=str(salida.message),
        arbitrage_free=libre,
        min_g=min_g,
    )


def calibrate_sabr_slice(
    strikes: np.ndarray,
    market_volatility: np.ndarray,
    forward: float,
    maturity: float,
    beta: float = 0.5,
    vega_weighted: bool = True,
    max_nfev: int = 5000,
) -> CalibrationResult:
    """Calibra ``alpha``, ``rho`` y ``nu`` de SABR con ``beta`` fijo.

    ``beta`` **no se calibra**, y no por comodidad: ``beta`` y ``rho`` controlan
    ambos la asimetria del smile y estan fuertemente correlacionados, asi que
    ajustar los cuatro parametros a la vez deja el problema practicamente no
    identificado. La practica de mercado es fijar ``beta`` por tipo de subyacente
    y calibrar los otros tres.

    Returns
    -------
    CalibrationResult
    """
    strikes = np.asarray(strikes, dtype=float)
    vol = np.asarray(market_volatility, dtype=float)
    if strikes.shape != vol.shape:
        raise ValueError("strikes y market_volatility deben tener la misma forma")
    if strikes.size < 3:
        raise ValueError("se calibran 3 parametros; se necesitan al menos 3 cotizaciones")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta debe estar en [0, 1]")

    pesos = _weights(forward, strikes, maturity, vol, vega_weighted)
    # Punto de partida: alpha desde la volatilidad en el dinero.
    idx_atm = int(np.argmin(np.abs(strikes - forward)))
    alpha0 = float(vol[idx_atm] * forward ** (1.0 - beta))
    x0 = np.array([alpha0, -0.3, 0.4])
    lower = np.array([1e-8, -0.999, 0.0])
    upper = np.array([5.0, 0.999, 5.0])

    def residuos(x: np.ndarray) -> np.ndarray:
        p = SABRParams(alpha=float(x[0]), beta=beta, rho=float(x[1]), nu=float(x[2]))
        ajustada = hagan_implied_volatility(strikes, forward, maturity, p)
        return pesos * (ajustada - vol)

    salida = least_squares(
        residuos, x0, bounds=(lower, upper), method="trf", max_nfev=max_nfev
    )
    params = SABRParams(
        alpha=float(salida.x[0]), beta=beta, rho=float(salida.x[1]), nu=float(salida.x[2])
    )
    ajustada = hagan_implied_volatility(strikes, forward, maturity, params)
    err = ajustada - vol
    libre, min_g = params.is_butterfly_free(forward, maturity)
    return CalibrationResult(
        params=params,
        rmse_volatility=float(np.sqrt(np.mean(err**2))),
        max_abs_error=float(np.max(np.abs(err))),
        n_iterations=int(salida.nfev),
        success=bool(salida.success),
        message=str(salida.message),
        arbitrage_free=libre,
        min_g=min_g,
    )
