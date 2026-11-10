"""Validacion de SVI, SSVI y las condiciones de no-arbitraje."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from volsurface import (
    SSVIParams,
    SVIParams,
    calibrate_svi_slice,
    check_butterfly,
    check_calendar,
    g_function,
    implied_density,
    numerical_derivatives,
    svi_derivatives,
)

ADMISIBLE = SVIParams(a=0.04, b=0.10, rho=-0.30, m=0.00, sigma=0.30)
CON_ARBITRAJE = SVIParams(a=0.01, b=0.90, rho=-0.95, m=0.05, sigma=0.03)


# ------------------------------------------------------------------- SVI crudo


def test_svi_rechaza_parametros_inadmisibles():
    with pytest.raises(ValueError, match="b debe ser no negativo"):
        SVIParams(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.2)
    with pytest.raises(ValueError, match="rho debe estar"):
        SVIParams(a=0.04, b=0.1, rho=1.0, m=0.0, sigma=0.2)
    with pytest.raises(ValueError, match="sigma debe ser positivo"):
        SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.0)


def test_las_derivadas_analiticas_coinciden_con_las_numericas():
    """Las derivadas de SVI se hacen a mano, asi que hay que comprobarlas.

    ``g`` depende de ``w''``: un error de signo ahi no rompe el ajuste pero si
    el diagnostico de arbitraje, que es justo lo que no se notaria.
    """
    k = np.linspace(-1.5, 1.5, 51)
    w_a, dw_a, d2w_a = ADMISIBLE.derivatives(k)
    w_n, dw_n, d2w_n = numerical_derivatives(ADMISIBLE.total_variance, k, h=1e-4)
    np.testing.assert_allclose(w_a, w_n, rtol=1e-12)
    np.testing.assert_allclose(dw_a, dw_n, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(d2w_a, d2w_n, rtol=1e-5, atol=1e-8)


def test_el_minimo_de_la_varianza_total_esta_en_el_vertice():
    """``w_min = a + b*sigma*sqrt(1-rho^2)``, alcanzado en el vertice."""
    k = np.linspace(-5.0, 5.0, 20_001)
    assert float(np.min(ADMISIBLE.total_variance(k))) == pytest.approx(
        ADMISIBLE.minimum_variance, rel=1e-6
    )


def test_la_varianza_total_es_positiva_en_un_corte_admisible():
    k = np.linspace(-3.0, 3.0, 1001)
    assert np.all(ADMISIBLE.total_variance(k) > 0.0)


def test_las_pendientes_asintoticas_son_las_esperadas():
    """Lejos del vertice, ``w`` es lineal con pendientes ``b(rho +/- 1)``."""
    b, rho = ADMISIBLE.b, ADMISIBLE.rho
    k = np.array([-500.0, -499.0, 499.0, 500.0])
    w = ADMISIBLE.total_variance(k)
    pendiente_izq = (w[1] - w[0]) / 1.0
    pendiente_der = (w[3] - w[2]) / 1.0
    assert pendiente_izq == pytest.approx(b * (rho - 1.0), rel=1e-6)
    assert pendiente_der == pytest.approx(b * (rho + 1.0), rel=1e-6)


# ----------------------------------------------------- condicion de mariposa


def test_un_corte_admisible_pasa_la_condicion_de_mariposa():
    libre, min_g = ADMISIBLE.is_butterfly_free()
    assert libre
    assert min_g > 0.0


def test_un_corte_con_alas_extremas_viola_la_condicion_de_mariposa():
    """Parametros con ``b`` grande, ``|rho|`` cerca de 1 y ``sigma`` pequenio.

    Pasan todas las condiciones *necesarias* --varianza total positiva en todo
    ``k``-- y aun asi generan densidad negativa. Es la razon por la que las
    condiciones necesarias no bastan.
    """
    cumple, _ = CON_ARBITRAJE.satisfies_necessary_conditions()
    assert cumple, "el caso debe pasar las condiciones necesarias para ser interesante"
    libre, min_g = CON_ARBITRAJE.is_butterfly_free()
    assert not libre
    assert min_g < -0.1


def test_el_signo_de_g_es_el_signo_de_la_densidad():
    """La identidad que justifica usar ``g`` como diagnostico."""
    k = np.linspace(-1.0, 1.0, 201)
    for corte in (ADMISIBLE, CON_ARBITRAJE):
        w, dw, d2w = corte.derivatives(k)
        g = g_function(k, w, dw, d2w)
        p = implied_density(k, w, dw, d2w)
        np.testing.assert_array_equal(np.sign(g), np.sign(p))


def test_la_densidad_implicita_de_un_corte_admisible_integra_a_uno():
    """Comprobacion independiente de que la superficie es una medida valida."""

    def densidad(x: float) -> float:
        k = np.array([x])
        w, dw, d2w = ADMISIBLE.derivatives(k)
        return float(implied_density(k, w, dw, d2w)[0])

    total, _ = quad(densidad, -12.0, 12.0, limit=400)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_la_densidad_de_un_corte_con_arbitraje_se_vuelve_negativa():
    k = np.linspace(-1.0, 1.0, 401)
    w, dw, d2w = CON_ARBITRAJE.derivatives(k)
    assert np.any(implied_density(k, w, dw, d2w) < 0.0)


def test_g_rechaza_varianza_no_positiva():
    with pytest.raises(ValueError, match="estrictamente positiva"):
        g_function(np.array([0.0]), np.array([-0.1]), np.array([0.0]), np.array([0.0]))


# --------------------------------------------------------------------- SSVI


def test_ssvi_rechaza_parametros_inadmisibles():
    with pytest.raises(ValueError, match="rho"):
        SSVIParams(rho=1.5, eta=1.0, gamma=0.5)
    with pytest.raises(ValueError, match="eta"):
        SSVIParams(rho=0.0, eta=-1.0, gamma=0.5)
    with pytest.raises(ValueError, match="gamma"):
        SSVIParams(rho=0.0, eta=1.0, gamma=1.5)


def test_el_corte_svi_equivalente_reproduce_el_ssvi():
    """La identificacion SSVI -> SVI crudo debe ser exacta, no aproximada."""
    ssvi = SSVIParams(rho=-0.4, eta=0.8, gamma=0.4)
    k = np.linspace(-1.2, 1.2, 101)
    for theta in (0.01, 0.04, 0.2):
        directo = ssvi.total_variance(k, theta)
        via_svi = ssvi.to_svi(theta).total_variance(k)
        np.testing.assert_allclose(directo, via_svi, rtol=1e-12, atol=1e-14)


def test_ssvi_en_el_dinero_vale_theta():
    """Por construccion ``w(0, theta) = theta``: es la varianza total ATM."""
    ssvi = SSVIParams(rho=-0.4, eta=0.8, gamma=0.4)
    for theta in (0.01, 0.04, 0.25):
        assert float(ssvi.total_variance(np.array([0.0]), theta)[0]) == pytest.approx(theta)


def test_las_condiciones_de_gatheral_jacquier_son_suficientes_no_necesarias():
    """Distincion que importa al calibrar.

    Un conjunto de parametros puede incumplir la condicion cerrada y aun asi
    tener densidad no negativa. Tratar la condicion suficiente como si fuera
    necesaria descarta regiones del espacio de parametros que si son validas.
    """
    agresivo = SSVIParams(rho=-0.7, eta=5.0, gamma=0.1)
    cumple_cerrada, diag = agresivo.butterfly_conditions(0.1)
    assert not cumple_cerrada
    assert diag["theta_phi2_factor"] > 4.0

    libre_numerica, min_g, _, _ = check_butterfly(
        lambda k: agresivo.total_variance(k, 0.1)
    )
    assert libre_numerica, "la condicion definitoria si se cumple pese a fallar la suficiente"
    assert min_g > 0.0


def test_una_superficie_ssvi_moderada_esta_libre_de_arbitraje():
    ssvi = SSVIParams(rho=-0.4, eta=0.8, gamma=0.4)
    plazos = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    reporte = ssvi.check_surface(0.04 * plazos, plazos)
    assert reporte.arbitrage_free, reporte.summary()
    assert reporte.min_g > 0.0
    assert reporte.min_calendar_slope > 0.0


def test_la_varianza_total_decreciente_se_detecta_como_arbitraje_de_calendario():
    """Un diferencial de calendario con precio negativo es arbitraje directo."""
    libre, pendiente, n = check_calendar(
        lambda k, t: np.full_like(np.asarray(k, dtype=float), 0.1 - 0.02 * t),
        np.array([0.5, 1.0, 2.0]),
    )
    assert not libre
    assert pendiente < 0.0
    assert n > 0


def test_check_calendar_exige_vencimientos_ordenados():
    with pytest.raises(ValueError, match="orden estrictamente creciente"):
        check_calendar(lambda k, t: np.full_like(k, 0.1), np.array([1.0, 0.5]))


# -------------------------------------------------------------- calibracion


def test_la_calibracion_svi_recupera_los_parametros_exactos():
    """Con datos generados por el propio modelo, el ajuste debe ser perfecto."""
    verdad = SVIParams(a=0.025, b=0.12, rho=-0.42, m=0.03, sigma=0.18)
    k = np.linspace(-0.6, 0.6, 21)
    res = calibrate_svi_slice(k, verdad.implied_volatility(k, 1.0), 1.0, forward=100.0)
    ajustado = res.params
    assert res.success
    assert res.rmse_volatility < 1e-8
    for nombre in ("a", "b", "rho", "m", "sigma"):
        assert getattr(ajustado, nombre) == pytest.approx(getattr(verdad, nombre), abs=1e-5)


def test_la_calibracion_informa_del_diagnostico_de_arbitraje():
    """Un ajuste con error pequenio que admite arbitraje es inservible.

    Por eso el resultado trae el diagnostico, en lugar de solo el error.
    """
    verdad = SVIParams(a=0.025, b=0.12, rho=-0.42, m=0.03, sigma=0.18)
    k = np.linspace(-0.6, 0.6, 21)
    res = calibrate_svi_slice(k, verdad.implied_volatility(k, 1.0), 1.0, forward=100.0)
    assert res.arbitrage_free is True
    assert res.min_g is not None and res.min_g > 0.0


def test_la_calibracion_tolera_ruido_de_mercado():
    verdad = SVIParams(a=0.03, b=0.15, rho=-0.35, m=0.02, sigma=0.22)
    k = np.linspace(-0.7, 0.7, 31)
    rng = np.random.default_rng(7)
    vol = verdad.implied_volatility(k, 1.0) * (1.0 + 0.002 * rng.standard_normal(k.size))
    res = calibrate_svi_slice(k, vol, 1.0, forward=100.0)
    assert res.success
    assert res.rmse_volatility < 0.002


def test_la_calibracion_exige_suficientes_cotizaciones():
    with pytest.raises(ValueError, match="al menos 5 cotizaciones"):
        calibrate_svi_slice(np.zeros(4), np.full(4, 0.2), 1.0)


def test_la_ponderacion_por_vega_cambia_el_ajuste():
    """Ponderar o no es una decision economica, no cosmetica."""
    verdad = SVIParams(a=0.03, b=0.15, rho=-0.35, m=0.02, sigma=0.22)
    k = np.linspace(-1.0, 1.0, 25)
    rng = np.random.default_rng(3)
    vol = verdad.implied_volatility(k, 1.0) + 0.01 * rng.standard_normal(k.size)
    con = calibrate_svi_slice(k, vol, 1.0, forward=100.0, vega_weighted=True)
    sin = calibrate_svi_slice(k, vol, 1.0, forward=100.0, vega_weighted=False)
    assert con.params.a != sin.params.a
