"""Modelo de Heston con la rama correcta del logaritmo complejo.

El modelo de Heston tiene una funcion caracteristica cerrada, asi que las
opciones europeas se valuan por inversion de Fourier sin simular. Pero la
formulacion original de Heston (1993) tiene un defecto numerico conocido: para
vencimientos largos o parametros ciertos, **cruza el corte de rama del logaritmo
complejo** y el precio salta de forma discontinua. La calibracion entonces no
converge, y el sintoma se confunde facilmente con un problema del optimizador.

Albrecher y coautores (2007) lo bautizaron *the little Heston trap* y mostraron
que existe una formulacion algebraicamente equivalente que **nunca** cruza el
corte. La diferencia esta en un solo signo:

- Formulacion con trampa: :math:`g_1 = \\frac{\\kappa - \\rho\\xi iu + d}
  {\\kappa - \\rho\\xi iu - d}` y exponenciales :math:`e^{+d\\tau}`.
- Formulacion estable: :math:`g_2 = 1/g_1` y exponenciales
  :math:`e^{-d\\tau}`, que decaen.

Este modulo implementa **las dos**, y no por completitud: la version con trampa
existe para poder **demostrar** el fallo en la bateria de pruebas. Una libreria
que solo trae la version correcta pide que se le crea; esta lo ensenia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

__all__ = ["HestonParams", "characteristic_function", "price_call", "price_put"]


@dataclass(frozen=True)
class HestonParams:
    """Parametros del modelo de Heston.

    .. math::

        dS_t = (r-q)S_t\\,dt + \\sqrt{v_t}\\,S_t\\,dW^1_t

        dv_t = \\kappa(\\theta - v_t)\\,dt + \\xi\\sqrt{v_t}\\,dW^2_t,
        \\qquad d\\langle W^1,W^2\\rangle_t = \\rho\\,dt

    Parameters
    ----------
    v0:
        Varianza instantanea inicial.
    kappa:
        Velocidad de reversion a la media.
    theta:
        Varianza de largo plazo.
    xi:
        Volatilidad de la volatilidad.
    rho:
        Correlacion entre el subyacente y su varianza; negativa en renta
        variable, que es lo que genera la asimetria del smile.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 <= 0.0 or self.theta <= 0.0:
            raise ValueError("v0 y theta deben ser positivos")
        if self.kappa <= 0.0:
            raise ValueError("kappa debe ser positivo para que haya reversion a la media")
        if self.xi < 0.0:
            raise ValueError("xi debe ser no negativo")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho debe estar en (-1, 1)")

    @property
    def is_numerically_degenerate(self) -> bool:
        """``True`` si ``xi`` es tan pequenio que la formula pierde precision.

        La funcion caracteristica divide entre ``xi**2``. Por debajo de
        ``xi ~ 1e-5`` eso significa dividir entre ``1e-10`` o menos, y la
        cancelacion catastrofica degrada el precio: con ``xi = 1e-7`` el error
        frente a Black-Scholes es de 2e-3, mientras que con ``xi = 1e-4`` es de
        2e-8. El limite ``xi -> 0`` **no** se alcanza numericamente por esta via;
        para volatilidad de volatilidad nula hay que usar Black-Scholes
        directamente, que es la solucion exacta de ese caso.
        """
        return self.xi < 1e-5

    @property
    def feller_condition(self) -> float:
        """Holgura de la condicion de Feller ``2*kappa*theta - xi^2``.

        Si es positiva, el proceso de varianza no alcanza cero. Si es negativa el
        modelo sigue siendo valido y la funcion caracteristica tambien, pero la
        varianza toca cero con probabilidad positiva, lo que afecta a los
        esquemas de simulacion. Se expone porque es informacion que conviene
        tener a la vista durante la calibracion.
        """
        return float(2.0 * self.kappa * self.theta - self.xi**2)

    @property
    def satisfies_feller(self) -> bool:
        """``True`` si se cumple la condicion de Feller."""
        return self.feller_condition >= 0.0


def characteristic_function(
    u: complex | np.ndarray,
    params: HestonParams,
    maturity: float,
    rate: float = 0.0,
    dividend: float = 0.0,
    spot: float = 1.0,
    trap: bool = False,
) -> complex | np.ndarray:
    """Funcion caracteristica de ``log S_T`` bajo la medida de riesgo neutral.

    Parameters
    ----------
    u:
        Argumento, posiblemente complejo.
    params:
        Parametros del modelo.
    maturity:
        Plazo en anios.
    rate, dividend, spot:
        Tasa libre de riesgo, tasa de dividendo y subyacente.
    trap:
        Si es ``True`` se usa la formulacion original de Heston, la que cruza el
        corte de rama. Existe para demostrar el fallo, no para usarse.

    Returns
    -------
    complex o numpy.ndarray
    """
    if maturity <= 0.0:
        raise ValueError("el vencimiento debe ser positivo")

    k, th, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0
    iu = 1j * np.asarray(u, dtype=complex)

    # d es la raiz del discriminante; su signo define la rama.
    d = np.sqrt((rho * xi * iu - k) ** 2 + xi**2 * (iu + np.asarray(u, dtype=complex) ** 2))

    if trap:
        # Formulacion original: g1 y exponenciales crecientes.
        g = (k - rho * xi * iu + d) / (k - rho * xi * iu - d)
        exp_d = np.exp(d * maturity)
        c = (k * th / xi**2) * (
            (k - rho * xi * iu + d) * maturity - 2.0 * np.log((1.0 - g * exp_d) / (1.0 - g))
        )
        dd = ((k - rho * xi * iu + d) / xi**2) * ((1.0 - exp_d) / (1.0 - g * exp_d))
    else:
        # Formulacion estable: g2 = 1/g1 y exponenciales decrecientes.
        g = (k - rho * xi * iu - d) / (k - rho * xi * iu + d)
        exp_d = np.exp(-d * maturity)
        c = (k * th / xi**2) * (
            (k - rho * xi * iu - d) * maturity - 2.0 * np.log((1.0 - g * exp_d) / (1.0 - g))
        )
        dd = ((k - rho * xi * iu - d) / xi**2) * ((1.0 - exp_d) / (1.0 - g * exp_d))

    forward_log = np.log(spot) + (rate - dividend) * maturity
    return np.exp(iu * forward_log + c + dd * v0)


def _gil_pelaez_probabilities(
    strike: float,
    params: HestonParams,
    maturity: float,
    rate: float,
    dividend: float,
    spot: float,
    trap: bool,
    limit: int,
) -> tuple[float, float]:
    """Las dos probabilidades de la formula de Heston, por inversion de Gil-Pelaez.

    .. math::

        P_2 = \\frac12 + \\frac1\\pi \\int_0^\\infty
              \\mathrm{Re}\\left[\\frac{e^{-iu\\log K}\\phi(u)}{iu}\\right] du

        P_1 = \\frac12 + \\frac1\\pi \\int_0^\\infty
              \\mathrm{Re}\\left[\\frac{e^{-iu\\log K}\\phi(u-i)}
              {iu\\,\\phi(-i)}\\right] du

    ``P_2`` es la probabilidad de ejercicio bajo la medida de riesgo neutral y
    ``P_1`` la misma bajo la medida del subyacente como numerario.
    """
    log_k = np.log(strike)
    cf = lambda u, sh: characteristic_function(  # noqa: E731
        u, params, maturity, rate, dividend, spot, trap
    ) if sh == 0 else characteristic_function(
        u - 1j, params, maturity, rate, dividend, spot, trap
    )
    forward = spot * np.exp((rate - dividend) * maturity)

    def integrando_p2(u: float) -> float:
        if u == 0.0:
            return 0.0
        return float(np.real(np.exp(-1j * u * log_k) * cf(u, 0) / (1j * u)))

    def integrando_p1(u: float) -> float:
        if u == 0.0:
            return 0.0
        return float(np.real(np.exp(-1j * u * log_k) * cf(u, 1) / (1j * u * forward)))

    # El integrando decae como 1/u^2 pero oscila; se parte el dominio para que
    # la cuadratura adaptativa no se salte oscilaciones en la cola.
    def integra(f) -> float:
        total = 0.0
        for lo, hi in ((1e-10, 1.0), (1.0, 10.0), (10.0, 50.0), (50.0, 200.0)):
            total += quad(f, lo, hi, limit=limit)[0]
        return total

    p2 = 0.5 + integra(integrando_p2) / np.pi
    p1 = 0.5 + integra(integrando_p1) / np.pi
    return float(p1), float(p2)


def price_call(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
    params: HestonParams,
    trap: bool = False,
    limit: int = 200,
) -> float:
    """Precio de un call europeo bajo Heston, por inversion de Fourier.

    .. math:: C = S_0 e^{-q\\tau} P_1 - K e^{-r\\tau} P_2

    Parameters
    ----------
    trap:
        Si es ``True`` usa la formulacion inestable. Solo para demostracion.

    Returns
    -------
    float
    """
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("spot y strike deben ser positivos")
    p1, p2 = _gil_pelaez_probabilities(
        strike, params, maturity, rate, dividend, spot, trap, limit
    )
    return float(spot * np.exp(-dividend * maturity) * p1 - strike * np.exp(-rate * maturity) * p2)


def price_put(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    maturity: float,
    params: HestonParams,
    trap: bool = False,
    limit: int = 200,
) -> float:
    """Precio de un put europeo, por paridad put-call.

    Usar la paridad en lugar de una segunda inversion garantiza que el par
    call-put sea consistente por construccion, sin error numerico relativo.
    """
    call = price_call(spot, strike, rate, dividend, maturity, params, trap, limit)
    return float(
        call - spot * np.exp(-dividend * maturity) + strike * np.exp(-rate * maturity)
    )
