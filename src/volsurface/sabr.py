"""SABR con la expansion asintotica de Hagan (2002).

SABR es el modelo de referencia para el smile de tipos de interes y, en general,
para cualquier mercado donde se cotice por delta. Su utilidad practica no viene
de la dinamica sino de que Hagan derivo una **formula asintotica cerrada** para
la volatilidad implicita, lo que hace la calibracion casi instantanea.

El precio de esa comodidad es que la formula es una aproximacion de primer orden
en ``T``: para vencimientos largos o volatilidad de volatilidad alta **admite
arbitraje de mariposa**, en particular en el ala izquierda con strikes bajos. No
es un defecto de implementacion, es una limitacion conocida del desarrollo
asintotico, y por eso este modulo expone la verificacion de ``g`` sobre el smile
resultante en lugar de suponer que es admisible.

Convencion habitual de ``beta``: 1 para dinamica lognormal, 0 para normal
(*Bachelier*), 0.5 para el caso CIR. En la practica ``beta`` se fija por criterio
de mercado y se calibran ``alpha``, ``rho`` y ``nu``, porque ``beta`` y ``rho``
estan fuertemente correlacionados y ajustar los cuatro a la vez deja el problema
mal planteado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SABRParams", "hagan_implied_volatility"]

# Umbral de conmutacion entre la formula cerrada de z/x(z) y su desarrollo en
# serie. No es arbitrario: es el punto donde se cruzan los dos errores.
#
#   - Por debajo, la formula cerrada sufre cancelacion catastrofica, porque
#     x(z) -> z y el logaritmo resta dos cantidades casi iguales. Su error
#     relativo crece como eps/|z|: en |z| = 1e-10 vale ya ~1e-6.
#   - Por encima, el desarrollo de segundo orden deja un error O(z^3).
#
# En 1e-4 el error de la serie es ~1e-12 y el de cancelacion ~1e-12. Con un
# umbral de 1e-10 --la eleccion ingenua-- el empalme deja un salto de 3e-8 en
# volatilidad justo en el strike en el dinero, que es el mas consultado.
_SERIES_THRESHOLD = 1e-4


@dataclass(frozen=True)
class SABRParams:
    """Parametros del modelo SABR.

    Parameters
    ----------
    alpha:
        Nivel de volatilidad inicial; positivo.
    beta:
        Exponente de elasticidad, en ``[0, 1]``. Se fija, no se calibra.
    rho:
        Correlacion entre el forward y su volatilidad, en ``(-1, 1)``.
    nu:
        Volatilidad de la volatilidad; no negativa.
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha debe ser positivo")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta debe estar en [0, 1]")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho debe estar en (-1, 1)")
        if self.nu < 0.0:
            raise ValueError("nu debe ser no negativo")

    def implied_volatility(self, strike: np.ndarray, forward: float, maturity: float) -> np.ndarray:
        """Volatilidad implicita lognormal por la formula de Hagan."""
        return hagan_implied_volatility(strike, forward, maturity, self)

    def atm_volatility(self, forward: float, maturity: float) -> float:
        """Volatilidad en el dinero, donde la formula se simplifica.

        .. math::

            \\sigma_{ATM} = \\frac{\\alpha}{F^{1-\\beta}}\\left[1 +
                \\left(\\frac{(1-\\beta)^2\\alpha^2}{24F^{2-2\\beta}}
                     + \\frac{\\rho\\beta\\nu\\alpha}{4F^{1-\\beta}}
                     + \\frac{(2-3\\rho^2)\\nu^2}{24}\\right)T\\right]
        """
        f_pow = forward ** (1.0 - self.beta)
        correccion = (
            (1.0 - self.beta) ** 2 * self.alpha**2 / (24.0 * f_pow**2)
            + self.rho * self.beta * self.nu * self.alpha / (4.0 * f_pow)
            + (2.0 - 3.0 * self.rho**2) * self.nu**2 / 24.0
        )
        return float(self.alpha / f_pow * (1.0 + correccion * maturity))

    def total_variance(self, k: np.ndarray, forward: float, maturity: float) -> np.ndarray:
        """Varianza total implicita en log-moneyness, para los diagnosticos."""
        k = np.asarray(k, dtype=float)
        strikes = forward * np.exp(k)
        vol = self.implied_volatility(strikes, forward, maturity)
        return vol**2 * maturity

    def is_butterfly_free(
        self, forward: float, maturity: float, k_grid: np.ndarray | None = None
    ) -> tuple[bool, float]:
        """Comprueba si el smile de Hagan admite arbitraje de mariposa.

        Es una comprobacion necesaria, no una formalidad: la expansion asintotica
        **puede** violar la condicion para plazos largos o ``nu`` alto.

        Returns
        -------
        libre, min_g
        """
        from .arbitrage import check_butterfly

        if k_grid is None:
            k_grid = np.linspace(-1.2, 1.2, 481)
        libre, min_g, _, _ = check_butterfly(
            lambda k: self.total_variance(k, forward, maturity), k_grid
        )
        return libre, min_g


def hagan_implied_volatility(
    strike: np.ndarray, forward: float, maturity: float, params: SABRParams
) -> np.ndarray:
    """Volatilidad implicita lognormal de SABR (Hagan et al., 2002).

    .. math::

        \\sigma(K,F) = \\frac{\\alpha}{(FK)^{(1-\\beta)/2}\\left[1
            + \\frac{(1-\\beta)^2}{24}\\log^2\\frac{F}{K}
            + \\frac{(1-\\beta)^4}{1920}\\log^4\\frac{F}{K}\\right]}
            \\cdot \\frac{z}{x(z)} \\cdot \\left[1 + \\Gamma\\, T\\right]

    con :math:`z = \\frac{\\nu}{\\alpha}(FK)^{(1-\\beta)/2}\\log\\frac{F}{K}` y
    :math:`x(z) = \\log\\frac{\\sqrt{1-2\\rho z+z^2}+z-\\rho}{1-\\rho}`.

    El cociente :math:`z/x(z)` tiende a 1 cuando :math:`z\\to 0`, pero evaluarlo
    directamente cerca del dinero produce ``0/0``. Aqui se usa el desarrollo
    :math:`z/x(z) \\approx 1 + \\rho z/2 + O(z^2)` en una vecindad del cero, que
    es lo que mantiene la formula estable en el strike donde mas se la consulta.

    Returns
    -------
    numpy.ndarray
    """
    strike = np.atleast_1d(np.asarray(strike, dtype=float))
    if forward <= 0.0:
        raise ValueError("el forward debe ser positivo")
    if np.any(strike <= 0.0):
        raise ValueError("los strikes deben ser positivos")
    if maturity <= 0.0:
        raise ValueError("el vencimiento debe ser positivo")

    a, b, rho, nu = params.alpha, params.beta, params.rho, params.nu
    one_minus_b = 1.0 - b
    log_fk = np.log(forward / strike)
    fk_pow = (forward * strike) ** (one_minus_b / 2.0)

    denominador = fk_pow * (
        1.0
        + one_minus_b**2 / 24.0 * log_fk**2
        + one_minus_b**4 / 1920.0 * log_fk**4
    )

    z = (nu / a) * fk_pow * log_fk
    # z / x(z), con desarrollo de segundo orden cerca del dinero.
    #
    #   x(z) = z - rho z^2/2 + (3rho^2-1) z^3/6 + O(z^4)
    #   z/x(z) = 1 + rho z/2 + (2 - 3 rho^2) z^2/12 + O(z^3)
    #
    # El termino cuadratico no es un lujo: con solo el lineal, el empalme con la
    # formula cerrada deja un escalon visible en la segunda derivada del smile,
    # y por tanto en la densidad implicita.
    cerca = np.abs(z) < _SERIES_THRESHOLD
    z_seguro = np.where(cerca, 1.0, z)
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z_seguro + z_seguro**2) + z_seguro - rho) / (1.0 - rho))
    serie = 1.0 + rho * z / 2.0 + (2.0 - 3.0 * rho**2) * z**2 / 12.0
    ratio = np.where(cerca, serie, z_seguro / x_z)

    correccion = (
        one_minus_b**2 * a**2 / (24.0 * fk_pow**2)
        + rho * b * nu * a / (4.0 * fk_pow)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    )

    vol = (a / denominador) * ratio * (1.0 + correccion * maturity)
    return vol if vol.size > 1 else vol
