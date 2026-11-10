"""SVI y SSVI: la parametrizacion estandar de superficies de volatilidad.

Omitir SVI en una libreria de superficies de volatilidad es omitir el estandar de
facto del sector. Su valor no es ajustar bien --muchas funciones lo hacen-- sino
que **existen condiciones cerradas y suficientes para ausencia de arbitraje
estatico** sobre sus parametros (Gatheral y Jacquier, 2014). Eso convierte el
problema de «ajustar y rezar» en uno de optimizacion con restricciones.

**SVI crudo** parametriza la varianza total implicita de un solo vencimiento:

.. math::

    w(k) = a + b\\left(\\rho(k-m) + \\sqrt{(k-m)^2 + \\sigma^2}\\right)

Cinco parametros con lectura directa: ``a`` el nivel, ``b`` la pendiente total de
las alas, ``rho`` la asimetria, ``m`` el desplazamiento horizontal y ``sigma`` la
curvatura en el vertice.

**SSVI** (*surface SVI*) extiende la idea a toda la superficie con una sola
funcion ``phi`` de la varianza total en el dinero:

.. math::

    w(k,\\theta) = \\frac{\\theta}{2}\\left(1 + \\rho\\varphi(\\theta)k +
        \\sqrt{(\\varphi(\\theta)k + \\rho)^2 + 1 - \\rho^2}\\right)

Es la version con la que se trabaja en produccion, porque las condiciones de
no-arbitraje son globales y no por vencimiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arbitrage import check_butterfly, check_calendar

__all__ = ["SVIParams", "SSVIParams", "svi_total_variance", "svi_derivatives"]


@dataclass(frozen=True)
class SVIParams:
    """Parametros del SVI crudo para un vencimiento.

    Parameters
    ----------
    a:
        Nivel de varianza total.
    b:
        Pendiente conjunta de las alas; debe ser no negativa.
    rho:
        Asimetria, en ``(-1, 1)``.
    m:
        Desplazamiento del vertice en log-moneyness.
    sigma:
        Curvatura en el vertice; debe ser positiva.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0.0:
            raise ValueError("b debe ser no negativo: una pendiente de alas negativa no tiene sentido")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho debe estar en el intervalo abierto (-1, 1)")
        if self.sigma <= 0.0:
            raise ValueError("sigma debe ser positivo")

    # ------------------------------------------------------------ evaluacion

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Varianza total implicita ``w(k)``."""
        return svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)

    def implied_volatility(self, k: np.ndarray, maturity: float) -> np.ndarray:
        """Volatilidad implicita de Black-Scholes."""
        if maturity <= 0.0:
            raise ValueError("el vencimiento debe ser positivo")
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / maturity)

    def derivatives(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``w``, ``w'`` y ``w''`` en forma analitica."""
        return svi_derivatives(k, self.a, self.b, self.rho, self.m, self.sigma)

    # ----------------------------------------------------------- restricciones

    @property
    def minimum_variance(self) -> float:
        """Minimo de ``w`` sobre todo ``k``, alcanzado en el vertice.

        .. math:: w_{\\min} = a + b\\sigma\\sqrt{1-\\rho^2}

        Que sea no negativo es condicion **necesaria**: una varianza total
        negativa no corresponde a ninguna volatilidad.
        """
        return float(self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2))

    def satisfies_necessary_conditions(self) -> tuple[bool, str]:
        """Condiciones necesarias de admisibilidad, en forma cerrada.

        Son baratas y se evaluan antes de cualquier malla. No son suficientes
        para descartar arbitraje de mariposa, para lo cual hay que mirar ``g``.

        Returns
        -------
        cumple, motivo
        """
        if self.minimum_variance < 0.0:
            return False, f"varianza total minima negativa: {self.minimum_variance:.6f}"
        # Las pendientes asintoticas de las alas son b(1 +/- rho); ambas deben
        # ser no negativas para que w no se haga negativo lejos del vertice.
        if self.b * (1.0 + self.rho) < 0.0 or self.b * (1.0 - self.rho) < 0.0:
            return False, "pendiente asintotica de un ala negativa"
        return True, "cumple las condiciones necesarias"

    def is_butterfly_free(self, k_grid: np.ndarray | None = None) -> tuple[bool, float]:
        """Verifica la condicion de mariposa evaluando ``g`` con derivadas analiticas.

        Returns
        -------
        libre, min_g
        """
        if k_grid is None:
            k_grid = np.linspace(-2.0, 2.0, 801)
        from .arbitrage import g_function

        w, dw, d2w = self.derivatives(np.asarray(k_grid, dtype=float))
        if np.any(w <= 0.0):
            return False, float("-inf")
        g = g_function(k_grid, w, dw, d2w)
        return bool(np.min(g) >= -1e-10), float(np.min(g))


def svi_total_variance(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    """Varianza total implicita del SVI crudo.

    Examples
    --------
    >>> import numpy as np
    >>> float(svi_total_variance(np.array([0.0]), 0.04, 0.1, -0.3, 0.0, 0.2)[0])
    0.0591...
    """
    k = np.asarray(k, dtype=float)
    centrado = k - m
    return a + b * (rho * centrado + np.sqrt(centrado**2 + sigma**2))


def svi_derivatives(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Varianza total y sus dos primeras derivadas, en forma analitica.

    Se derivan a mano en lugar de numericamente porque ``g`` depende de ``w''``,
    y un error de diferenciacion numerica puede fabricar violaciones de
    no-arbitraje que no existen o, peor, ocultar las que si.

    Con :math:`y = k - m` y :math:`R = \\sqrt{y^2 + \\sigma^2}`:

    .. math::

        w = a + b(\\rho y + R), \\qquad
        w' = b\\left(\\rho + \\frac{y}{R}\\right), \\qquad
        w'' = \\frac{b\\,\\sigma^2}{R^3}

    Returns
    -------
    w, dw_dk, d2w_dk2
    """
    k = np.asarray(k, dtype=float)
    y = k - m
    r = np.sqrt(y**2 + sigma**2)
    w = a + b * (rho * y + r)
    dw = b * (rho + y / r)
    d2w = b * sigma**2 / r**3
    return w, dw, d2w


@dataclass(frozen=True)
class SSVIParams:
    """SSVI con familia de ley de potencia para ``phi``.

    .. math:: \\varphi(\\theta) = \\frac{\\eta}{\\theta^{\\gamma}(1+\\theta)^{1-\\gamma}}

    Parameters
    ----------
    rho:
        Asimetria global, en ``(-1, 1)``.
    eta:
        Escala de la curvatura; positiva.
    gamma:
        Exponente de la ley de potencia, en ``(0, 1)``.
    """

    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho debe estar en (-1, 1)")
        if self.eta <= 0.0:
            raise ValueError("eta debe ser positivo")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma debe estar en (0, 1)")

    def phi(self, theta: float | np.ndarray) -> np.ndarray:
        """Funcion ``phi`` de la ley de potencia."""
        theta = np.asarray(theta, dtype=float)
        if np.any(theta <= 0.0):
            raise ValueError("theta debe ser positivo")
        return self.eta / (theta**self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    def total_variance(self, k: np.ndarray, theta: float) -> np.ndarray:
        """Varianza total implicita ``w(k, theta)``."""
        k = np.asarray(k, dtype=float)
        p = float(self.phi(theta))
        pk = p * k
        return 0.5 * theta * (
            1.0 + self.rho * pk + np.sqrt((pk + self.rho) ** 2 + 1.0 - self.rho**2)
        )

    def implied_volatility(self, k: np.ndarray, theta: float, maturity: float) -> np.ndarray:
        """Volatilidad implicita de Black-Scholes."""
        if maturity <= 0.0:
            raise ValueError("el vencimiento debe ser positivo")
        return np.sqrt(self.total_variance(k, theta) / maturity)

    def to_svi(self, theta: float) -> SVIParams:
        """Corte a vencimiento fijo, expresado como SVI crudo.

        La identificacion se obtiene igualando las dos formas:

        .. math::

            a = \\frac{\\theta}{2}\\left(1 - \\rho^2\\right), \\quad
            b = \\frac{\\theta\\varphi}{2}, \\quad
            m = -\\frac{\\rho}{\\varphi}, \\quad
            \\sigma = \\frac{\\sqrt{1-\\rho^2}}{\\varphi}

        Tener la equivalencia explicita permite reutilizar en SSVI todo el
        aparato de derivadas analiticas y diagnostico del SVI crudo.
        """
        p = float(self.phi(theta))
        return SVIParams(
            a=0.5 * theta * (1.0 - self.rho**2),
            b=0.5 * theta * p,
            rho=self.rho,
            m=-self.rho / p,
            sigma=float(np.sqrt(1.0 - self.rho**2) / p),
        )

    # --------------------------------------- condiciones de Gatheral-Jacquier

    def butterfly_conditions(self, theta: float) -> tuple[bool, dict[str, float]]:
        """Condiciones cerradas y suficientes para ausencia de arbitraje de mariposa.

        Gatheral y Jacquier (2014), teorema 4.2: la superficie SSVI esta libre de
        arbitraje de mariposa si, para todo ``theta``,

        .. math::

            \\theta\\varphi(\\theta)\\left(1+|\\rho|\\right) < 4, \\qquad
            \\theta\\varphi(\\theta)^2\\left(1+|\\rho|\\right) \\leq 4

        Su valor practico es que son **restricciones sobre los parametros**, no
        una comprobacion sobre una malla: se pueden imponer directamente al
        optimizador durante la calibracion.

        Returns
        -------
        cumple, diagnostico
            El diagnostico contiene los dos valores y sus holguras.
        """
        p = float(self.phi(theta))
        factor = 1.0 + abs(self.rho)
        c1 = theta * p * factor
        c2 = theta * p**2 * factor
        return (c1 < 4.0 and c2 <= 4.0), {
            "theta_phi_factor": c1,
            "theta_phi2_factor": c2,
            "holgura_condicion_1": 4.0 - c1,
            "holgura_condicion_2": 4.0 - c2,
        }

    def check_surface(self, thetas: np.ndarray, maturities: np.ndarray):
        """Diagnostico completo de no-arbitraje sobre la superficie.

        Comprueba mariposa en cada vencimiento con la funcion ``g`` --la
        condicion **definitoria**, no solo la suficiente-- y la monotonia de la
        varianza total en el vencimiento.

        Returns
        -------
        ArbitrageReport
        """
        from .arbitrage import ArbitrageReport

        thetas = np.asarray(thetas, dtype=float)
        maturities = np.asarray(maturities, dtype=float)
        if thetas.size != maturities.size:
            raise ValueError("thetas y maturities deben tener la misma longitud")

        min_g, peor_k, violaciones = np.inf, 0.0, 0
        for th in thetas:
            libre, g, k, n = check_butterfly(lambda k, th=float(th): self.total_variance(k, th))
            violaciones += n
            if g < min_g:
                min_g, peor_k = g, k

        cal_libre, min_pend, cal_viol = check_calendar(
            lambda k, t: self.total_variance(k, float(np.interp(t, maturities, thetas))),
            maturities,
        )
        return ArbitrageReport(
            butterfly_free=bool(min_g >= -1e-10),
            calendar_free=cal_libre,
            min_g=float(min_g),
            worst_butterfly_k=float(peor_k),
            min_calendar_slope=float(min_pend),
            n_butterfly_violations=int(violaciones),
            n_calendar_violations=int(cal_viol),
        )
