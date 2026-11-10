"""Volatilidad local de Dupire, con regularizacion.

La formula de Dupire extrae la volatilidad local del precio de las opciones:

.. math::

    \\sigma_{loc}^2(K,T) = \\frac{\\partial_T C + (r-q)K\\,\\partial_K C + qC}
                                 {\\tfrac12 K^2\\,\\partial^2_{KK} C}

y **divide por la segunda derivada del precio respecto al strike**. Ese es el
problema: esa segunda derivada es la densidad implicita, que es pequenia en las
alas y se estima a partir de datos ruidosos. El resultado es un operador
mal planteado en el sentido de Hadamard: perturbaciones pequenias en los precios
producen perturbaciones grandes en la volatilidad local. Nombrar a Dupire sin
decir como se regulariza es omitir el unico problema real del metodo.

Aqui se usan dos medidas concretas:

1. **Trabajar sobre la varianza total implicita, no sobre los precios.** La
   formulacion equivalente de Gatheral

   .. math::

       \\sigma_{loc}^2 = \\frac{\\partial_T w}
           {1 - \\frac{k}{w}\\partial_k w
            + \\frac14\\left(-\\frac14 - \\frac1w + \\frac{k^2}{w^2}\\right)
              (\\partial_k w)^2
            + \\frac12 \\partial^2_k w}

   es mucho mejor acondicionada, porque ``w`` es una funcion suave y de magnitud
   controlada mientras los precios varian varios ordenes de magnitud entre el
   dinero y las alas.

2. **Suelo explicito en el denominador.** El denominador es exactamente la
   funcion ``g`` de no-arbitraje. Si la superficie de entrada admite arbitraje,
   ``g`` se anula o cambia de signo y la volatilidad local explota o se vuelve
   imaginaria. En lugar de devolver un ``nan`` silencioso, se recorta con un
   suelo declarado y se informa de cuantos nodos se vieron afectados: una
   volatilidad local recortada es un sintoma de que la superficie implicita de
   partida no es admisible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["LocalVolResult", "local_volatility", "local_volatility_from_svi_slices"]


@dataclass(frozen=True)
class LocalVolResult:
    """Superficie de volatilidad local."""

    log_moneyness: np.ndarray
    maturities: np.ndarray
    local_variance: np.ndarray
    """Matriz ``(n_maturities, n_k)`` de varianza local."""

    n_floored: int
    """Nodos donde hubo que aplicar el suelo del denominador."""

    min_denominator: float
    """Minimo del denominador antes de recortar; cercano a cero senala problemas."""

    @property
    def local_volatility(self) -> np.ndarray:
        """Volatilidad local, raiz de la varianza."""
        return np.sqrt(self.local_variance)

    @property
    def is_well_posed(self) -> bool:
        """``True`` si no hubo que recortar ningun nodo.

        Si es ``False``, la superficie implicita de entrada no es admisible y la
        volatilidad local resultante no es de fiar en los nodos recortados.
        """
        return self.n_floored == 0


def local_volatility(
    w_surface: Callable[[np.ndarray, float], np.ndarray],
    k_grid: np.ndarray,
    maturities: np.ndarray,
    h_k: float = 1e-3,
    h_t: float = 1e-3,
    denominator_floor: float = 1e-6,
) -> LocalVolResult:
    """Varianza local por la formulacion de Gatheral de la ecuacion de Dupire.

    Parameters
    ----------
    w_surface:
        Funcion ``(k, T) -> w(k, T)`` de varianza total implicita.
    k_grid:
        Malla de log-moneyness.
    maturities:
        Vencimientos, estrictamente positivos.
    h_k, h_t:
        Pasos de las diferencias finitas en moneyness y en vencimiento.
    denominator_floor:
        Suelo del denominador. Los nodos recortados se cuentan y se informan.

    Returns
    -------
    LocalVolResult
    """
    k_grid = np.asarray(k_grid, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    if np.any(maturities <= 0.0):
        raise ValueError("los vencimientos deben ser estrictamente positivos")
    if np.any(maturities - h_t <= 0.0):
        raise ValueError(
            f"h_t = {h_t} es demasiado grande para el vencimiento minimo "
            f"{maturities.min()}: la diferencia centrada saldria del dominio"
        )

    variance = np.empty((maturities.size, k_grid.size))
    n_floored = 0
    min_denom = np.inf

    for i, t in enumerate(maturities):
        t = float(t)
        w = np.asarray(w_surface(k_grid, t), dtype=float)
        if np.any(w <= 0.0):
            raise ValueError(f"varianza total no positiva en T = {t}")

        # Derivada en el vencimiento, por diferencia centrada.
        dw_dt = (
            np.asarray(w_surface(k_grid, t + h_t), dtype=float)
            - np.asarray(w_surface(k_grid, t - h_t), dtype=float)
        ) / (2.0 * h_t)

        # Derivadas en moneyness, esquema de cinco puntos por la misma razon que
        # en el modulo de arbitraje: g depende de la segunda derivada.
        w_m2 = np.asarray(w_surface(k_grid - 2.0 * h_k, t), dtype=float)
        w_m1 = np.asarray(w_surface(k_grid - h_k, t), dtype=float)
        w_p1 = np.asarray(w_surface(k_grid + h_k, t), dtype=float)
        w_p2 = np.asarray(w_surface(k_grid + 2.0 * h_k, t), dtype=float)
        dw_dk = (-w_p2 + 8.0 * w_p1 - 8.0 * w_m1 + w_m2) / (12.0 * h_k)
        d2w_dk2 = (-w_p2 + 16.0 * w_p1 - 30.0 * w + 16.0 * w_m1 - w_m2) / (12.0 * h_k**2)

        denominador = (
            1.0
            - k_grid / w * dw_dk
            + 0.25 * (-0.25 - 1.0 / w + k_grid**2 / w**2) * dw_dk**2
            + 0.5 * d2w_dk2
        )
        min_denom = min(min_denom, float(np.min(denominador)))
        recortado = denominador < denominator_floor
        n_floored += int(np.sum(recortado))
        denominador = np.where(recortado, denominator_floor, denominador)

        variance[i] = np.maximum(dw_dt / denominador, 0.0)

    return LocalVolResult(
        log_moneyness=k_grid,
        maturities=maturities,
        local_variance=variance,
        n_floored=n_floored,
        min_denominator=float(min_denom),
    )


def local_volatility_from_svi_slices(
    slices: dict[float, "object"],
    k_grid: np.ndarray,
    maturities: np.ndarray | None = None,
    **kwargs,
) -> LocalVolResult:
    """Volatilidad local a partir de cortes SVI interpolados en el vencimiento.

    La interpolacion de la varianza total en ``T`` es **lineal**, que es la
    eleccion correcta y no la comoda: la interpolacion lineal de la varianza
    total preserva la monotonia en ``T``, es decir no introduce arbitraje de
    calendario. Una interpolacion por splines en ``T`` puede crear tramos
    decrecientes entre nodos y fabricar arbitraje donde los datos no lo tenian.

    Parameters
    ----------
    slices:
        Diccionario ``{vencimiento: SVIParams}``.
    k_grid:
        Malla de log-moneyness.
    maturities:
        Vencimientos donde evaluar; por omision los de ``slices``.

    Returns
    -------
    LocalVolResult
    """
    if len(slices) < 2:
        raise ValueError("se requieren al menos dos cortes para derivar en el vencimiento")
    plazos = np.array(sorted(slices.keys()), dtype=float)
    params = [slices[float(t)] for t in plazos]

    def w_surface(k: np.ndarray, t: float) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        # Las varianzas totales de cada corte, evaluadas en la misma malla.
        valores = np.array([p.total_variance(k) for p in params])
        if t <= plazos[0]:
            return valores[0] * (t / plazos[0])
        if t >= plazos[-1]:
            return valores[-1] * (t / plazos[-1])
        j = int(np.searchsorted(plazos, t))
        t0, t1 = plazos[j - 1], plazos[j]
        peso = (t - t0) / (t1 - t0)
        return (1.0 - peso) * valores[j - 1] + peso * valores[j]

    if maturities is None:
        # Se excluyen los extremos para que la diferencia centrada en T no caiga
        # en la zona de extrapolacion.
        maturities = plazos[1:-1] if plazos.size > 2 else plazos
    return local_volatility(w_surface, k_grid, maturities, **kwargs)
