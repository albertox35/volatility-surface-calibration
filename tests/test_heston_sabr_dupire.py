"""Validacion de Heston, SABR y Dupire contra soluciones cerradas."""

from __future__ import annotations

import math

import numpy as np
import pytest

from volsurface import (
    HestonParams,
    SABRParams,
    SVIParams,
    calibrate_sabr_slice,
    hagan_implied_volatility,
    local_volatility,
    local_volatility_from_svi_slices,
    price_call,
    price_put,
)


def bs_call(spot: float, strike: float, rate: float, div: float, vol: float, t: float) -> float:
    """Black-Scholes cerrado, como referencia independiente."""
    n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol**2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    return spot * math.exp(-div * t) * n(d1) - strike * math.exp(-rate * t) * n(d2)


# ------------------------------------------------------------------- Heston


def test_heston_rechaza_parametros_inadmisibles():
    with pytest.raises(ValueError, match="kappa"):
        HestonParams(v0=0.04, kappa=0.0, theta=0.04, xi=0.3, rho=0.0)
    with pytest.raises(ValueError, match="v0 y theta"):
        HestonParams(v0=-0.01, kappa=1.0, theta=0.04, xi=0.3, rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.3, rho=1.0)


@pytest.mark.parametrize("maturity", [0.25, 1.0, 3.0])
def test_heston_con_volatilidad_de_volatilidad_pequena_coincide_con_black_scholes(maturity):
    """Caso limite con solucion cerrada.

    Con ``v0 = theta`` y ``xi`` muy pequenio la varianza es practicamente
    constante, asi que Heston tiene que reproducir Black-Scholes con
    ``sigma = sqrt(v0)``.
    """
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=0.0)
    heston = price_call(100.0, 100.0, 0.03, 0.01, maturity, params)
    exacto = bs_call(100.0, 100.0, 0.03, 0.01, 0.2, maturity)
    assert heston == pytest.approx(exacto, abs=1e-6)


def test_la_diferencia_con_black_scholes_escala_como_el_cuadrado_de_xi():
    """Validacion del orden de convergencia, no solo del limite.

    La correccion de Heston sobre Black-Scholes es ``O(xi^2)`` cuando
    ``v0 = theta`` y ``rho = 0``. Si la implementacion tuviera un error de
    primer orden, el cociente entre errores consecutivos seria 10 y no 100.
    """
    exacto = bs_call(100.0, 100.0, 0.03, 0.01, 0.2, 1.0)
    errores = []
    for xi in (1e-3, 1e-2):
        params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=xi, rho=0.0)
        errores.append(abs(price_call(100.0, 100.0, 0.03, 0.01, 1.0, params) - exacto))
    razon = errores[1] / errores[0]
    assert 70.0 < razon < 130.0, f"razon observada {razon:.1f}, se esperaba ~100"


def test_la_bandera_de_degeneracion_numerica_avisa_del_limite_inalcanzable():
    """``xi -> 0`` no se alcanza numericamente: la formula divide entre xi^2."""
    assert HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=1e-7, rho=0.0).is_numerically_degenerate
    assert not HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=0.0).is_numerically_degenerate


def test_la_formulacion_con_trampa_falla_a_plazo_largo():
    """El *little Heston trap*, demostrado.

    Las dos formulaciones son algebraicamente identicas. A plazo corto
    coinciden; a plazo largo la original cruza el corte de rama del logaritmo
    complejo y el precio deja de tener sentido o se vuelve ``nan``.
    """
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.7)

    corto_estable = price_call(100.0, 100.0, 0.02, 0.0, 1.0, params, trap=False)
    corto_trampa = price_call(100.0, 100.0, 0.02, 0.0, 1.0, params, trap=True)
    assert corto_estable == pytest.approx(corto_trampa, abs=1e-8)

    largo_estable = price_call(100.0, 100.0, 0.02, 0.0, 5.0, params, trap=False)
    largo_trampa = price_call(100.0, 100.0, 0.02, 0.0, 5.0, params, trap=True)
    assert abs(largo_estable - largo_trampa) > 1.0

    muy_largo_trampa = price_call(100.0, 100.0, 0.02, 0.0, 15.0, params, trap=True)
    assert math.isnan(muy_largo_trampa) or abs(muy_largo_trampa) > 1e3


def test_la_formulacion_estable_es_monotona_en_el_vencimiento():
    """Y la version correcta se comporta: el call vale mas a mayor plazo."""
    params = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.7)
    precios = [
        price_call(100.0, 100.0, 0.02, 0.0, t, params) for t in (1.0, 5.0, 10.0, 20.0, 30.0)
    ]
    assert all(b > a for a, b in zip(precios, precios[1:])), precios
    assert all(math.isfinite(p) for p in precios)


def test_paridad_put_call_en_heston():
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.4, rho=-0.6)
    call = price_call(100.0, 95.0, 0.03, 0.01, 1.5, params)
    put = price_put(100.0, 95.0, 0.03, 0.01, 1.5, params)
    izquierda = call - put
    derecha = 100.0 * math.exp(-0.01 * 1.5) - 95.0 * math.exp(-0.03 * 1.5)
    assert izquierda == pytest.approx(derecha, abs=1e-10)


def test_el_precio_crece_con_la_varianza_inicial():
    base = dict(kappa=1.5, theta=0.04, xi=0.4, rho=-0.5)
    precios = [
        price_call(100.0, 100.0, 0.02, 0.0, 1.0, HestonParams(v0=v, **base))
        for v in (0.01, 0.04, 0.09, 0.16)
    ]
    assert all(b > a for a, b in zip(precios, precios[1:]))


def test_la_condicion_de_feller_se_reporta():
    cumple = HestonParams(v0=0.04, kappa=3.0, theta=0.04, xi=0.3, rho=-0.5)
    incumple = HestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=0.9, rho=-0.5)
    assert cumple.satisfies_feller and cumple.feller_condition > 0.0
    assert not incumple.satisfies_feller and incumple.feller_condition < 0.0


# --------------------------------------------------------------------- SABR


def test_sabr_rechaza_parametros_inadmisibles():
    with pytest.raises(ValueError, match="alpha"):
        SABRParams(alpha=0.0, beta=0.5, rho=0.0, nu=0.3)
    with pytest.raises(ValueError, match="beta"):
        SABRParams(alpha=0.2, beta=1.5, rho=0.0, nu=0.3)
    with pytest.raises(ValueError, match="nu"):
        SABRParams(alpha=0.2, beta=0.5, rho=0.0, nu=-0.1)


def test_sabr_lognormal_sin_volatilidad_de_volatilidad_da_volatilidad_constante():
    """Caso limite exacto: ``beta=1`` y ``nu=0`` reducen SABR a Black.

    Con ``beta = 1`` desaparecen todos los terminos con ``(1-beta)``, y con
    ``nu = 0`` desaparece ``z/x(z)`` y el termino de volatilidad de
    volatilidad. Queda ``sigma = alpha`` para todo strike.
    """
    params = SABRParams(alpha=0.25, beta=1.0, rho=0.0, nu=0.0)
    strikes = np.array([40.0, 70.0, 100.0, 150.0, 250.0])
    vols = params.implied_volatility(strikes, 100.0, 1.0)
    np.testing.assert_allclose(vols, 0.25, rtol=1e-12)


def test_la_formula_en_el_dinero_coincide_con_la_general():
    """El limite ``z -> 0`` tiene que empalmar con la formula general."""
    params = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.3, nu=0.4)
    general = float(params.implied_volatility(np.array([100.0]), 100.0, 1.0)[0])
    atm = params.atm_volatility(100.0, 1.0)
    assert general == pytest.approx(atm, rel=1e-10)


def test_el_smile_es_continuo_al_cruzar_el_dinero():
    """El cociente ``z/x(z)`` es ``0/0`` en el dinero y se resuelve en serie.

    Si el empalme estuviera mal, habria un salto justo en el strike mas
    consultado de todos.
    """
    params = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.4, nu=0.5)
    strikes = 100.0 * np.exp(np.linspace(-1e-6, 1e-6, 2001))
    vols = params.implied_volatility(strikes, 100.0, 1.0)
    assert np.max(np.abs(np.diff(vols))) < 1e-8
    assert np.all(np.isfinite(vols))


def test_rho_negativo_inclina_el_smile_hacia_los_strikes_bajos():
    params = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.6, nu=0.5)
    bajo = float(params.implied_volatility(np.array([80.0]), 100.0, 1.0)[0])
    alto = float(params.implied_volatility(np.array([125.0]), 100.0, 1.0)[0])
    assert bajo > alto


def test_sabr_a_plazo_corto_esta_libre_de_arbitraje():
    params = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.3, nu=0.3)
    libre, min_g = params.is_butterfly_free(100.0, 1.0)
    assert libre
    assert min_g > 0.0


@pytest.mark.parametrize(
    ("maturity", "nu"),
    [(5.0, 0.8), (10.0, 0.8), (20.0, 1.5)],
)
def test_la_expansion_de_hagan_admite_arbitraje_a_plazo_largo(maturity, nu):
    """Limitacion conocida de la formula asintotica, demostrada.

    Hagan es una aproximacion de primer orden en ``T``. Para plazos largos o
    volatilidad de volatilidad alta, el smile resultante tiene densidad
    negativa. No es un defecto de esta implementacion: es la razon por la que en
    produccion se usa SABR con correcciones o se pasa a SSVI.
    """
    params = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.3, nu=nu)
    libre, min_g = params.is_butterfly_free(100.0, maturity)
    assert not libre
    assert min_g < 0.0


def test_la_calibracion_sabr_recupera_los_parametros():
    verdad = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.35, nu=0.45)
    strikes = np.linspace(70.0, 140.0, 15)
    vols = hagan_implied_volatility(strikes, 100.0, 1.0, verdad)
    res = calibrate_sabr_slice(strikes, vols, 100.0, 1.0, beta=0.5)
    assert res.success
    assert res.rmse_volatility < 1e-8
    assert res.params.alpha == pytest.approx(verdad.alpha, rel=1e-5)
    assert res.params.rho == pytest.approx(verdad.rho, abs=1e-5)
    assert res.params.nu == pytest.approx(verdad.nu, rel=1e-5)


def test_la_calibracion_sabr_exige_suficientes_cotizaciones():
    with pytest.raises(ValueError, match="al menos 3"):
        calibrate_sabr_slice(np.array([100.0, 110.0]), np.array([0.2, 0.21]), 100.0, 1.0)


# -------------------------------------------------------------------- Dupire


@pytest.mark.parametrize("sigma", [0.12, 0.27, 0.55])
def test_una_volatilidad_implicita_constante_da_una_volatilidad_local_identica(sigma):
    """Caso con solucion exacta.

    Si ``w(k,T) = sigma^2 T``, entonces ``dw/dT = sigma^2`` y todas las
    derivadas en ``k`` se anulan, asi que el denominador vale 1 y la
    volatilidad local es exactamente ``sigma``. Cualquier error de signo o de
    factor en la formula de Gatheral lo rompe.
    """
    res = local_volatility(
        lambda k, t: np.full_like(np.asarray(k, dtype=float), sigma**2 * t),
        np.linspace(-0.8, 0.8, 17),
        np.array([0.5, 1.0, 2.0]),
    )
    np.testing.assert_allclose(res.local_volatility, sigma, rtol=1e-9)
    assert res.is_well_posed
    assert res.n_floored == 0


def test_la_volatilidad_local_se_calcula_sobre_cortes_svi():
    cortes = {
        0.25: SVIParams(a=0.010, b=0.040, rho=-0.3, m=0.0, sigma=0.25),
        0.50: SVIParams(a=0.020, b=0.055, rho=-0.3, m=0.0, sigma=0.25),
        1.00: SVIParams(a=0.040, b=0.080, rho=-0.3, m=0.0, sigma=0.25),
        2.00: SVIParams(a=0.080, b=0.110, rho=-0.3, m=0.0, sigma=0.25),
    }
    res = local_volatility_from_svi_slices(cortes, np.linspace(-0.5, 0.5, 21))
    assert np.all(res.local_volatility > 0.0)
    assert np.all(np.isfinite(res.local_volatility))
    assert res.is_well_posed


def test_dupire_informa_cuando_recorta_el_denominador():
    """Un denominador cercano a cero es sintoma de que la entrada no es admisible.

    El denominador de la formula de Gatheral **es** la funcion ``g``. Si la
    superficie implicita admite arbitraje, la volatilidad local explota. Se
    informa en lugar de devolver un ``nan`` en silencio.
    """
    malo = SVIParams(a=0.01, b=0.90, rho=-0.95, m=0.05, sigma=0.03)
    res = local_volatility(
        lambda k, t: malo.total_variance(np.asarray(k, dtype=float)) * t,
        np.linspace(-0.5, 0.5, 41),
        np.array([1.0, 2.0]),
    )
    assert not res.is_well_posed
    assert res.n_floored > 0
    assert res.min_denominator < 0.0


def test_dupire_rechaza_un_paso_temporal_que_sale_del_dominio():
    with pytest.raises(ValueError, match="demasiado grande"):
        local_volatility(
            lambda k, t: np.full_like(np.asarray(k, dtype=float), 0.04 * t),
            np.linspace(-0.5, 0.5, 11),
            np.array([0.0005]),
            h_t=1e-3,
        )


def test_dupire_exige_al_menos_dos_cortes():
    with pytest.raises(ValueError, match="al menos dos cortes"):
        local_volatility_from_svi_slices(
            {1.0: SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.3)},
            np.linspace(-0.5, 0.5, 11),
        )
