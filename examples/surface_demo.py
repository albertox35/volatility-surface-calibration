"""Demostracion de la calibracion y el diagnostico de no-arbitraje.

Ejecutar con:

    python examples/surface_demo.py
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from volsurface import (
    HestonParams,
    SABRParams,
    SSVIParams,
    SVIParams,
    calibrate_sabr_slice,
    calibrate_svi_slice,
    check_butterfly,
    hagan_implied_volatility,
    implied_density,
    local_volatility_from_svi_slices,
    price_call,
)


def seccion(titulo: str) -> None:
    print(f"\n{'=' * 74}\n{titulo}\n{'=' * 74}")


def bs_call(spot, strike, rate, div, vol, t):
    n = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    d1 = (math.log(spot / strike) + (rate - div + 0.5 * vol**2) * t) / (vol * math.sqrt(t))
    return spot * math.exp(-div * t) * n(d1) - strike * math.exp(-rate * t) * n(d1 - vol * math.sqrt(t))


def demo_svi() -> None:
    seccion("SVI: calibracion y diagnostico de no-arbitraje")
    verdad = SVIParams(a=0.025, b=0.12, rho=-0.42, m=0.03, sigma=0.18)
    k = np.linspace(-0.6, 0.6, 21)
    vol = verdad.implied_volatility(k, 1.0)
    res = calibrate_svi_slice(k, vol, 1.0, forward=100.0)
    print(f"parametros verdaderos : a={verdad.a:.5f} b={verdad.b:.5f} rho={verdad.rho:+.5f} "
          f"m={verdad.m:+.5f} sigma={verdad.sigma:.5f}")
    p = res.params
    print(f"parametros ajustados  : a={p.a:.5f} b={p.b:.5f} rho={p.rho:+.5f} "
          f"m={p.m:+.5f} sigma={p.sigma:.5f}")
    print(f"ajuste: {res}")

    print("\nUn corte que pasa las condiciones necesarias y aun asi admite arbitraje:")
    malo = SVIParams(a=0.01, b=0.90, rho=-0.95, m=0.05, sigma=0.03)
    cumple, motivo = malo.satisfies_necessary_conditions()
    libre, min_g = malo.is_butterfly_free()
    print(f"  condiciones necesarias : {cumple} ({motivo})")
    print(f"  condicion de mariposa  : {libre}, min g = {min_g:.5f}")
    kk = np.linspace(-0.5, 0.5, 401)
    w, dw, d2w = malo.derivatives(kk)
    dens = implied_density(kk, w, dw, d2w)
    print(f"  densidad implicita minima: {dens.min():.5f} <- negativa: arbitraje de mariposa")


def demo_ssvi() -> None:
    seccion("SSVI: condiciones cerradas de Gatheral-Jacquier")
    print("Las condiciones son SUFICIENTES, no necesarias. Tratarlas como")
    print("necesarias descarta regiones del espacio de parametros que si son validas.\n")
    print(f"{'parametros':<26} {'theta':>6} {'c1<4':>8} {'c2<=4':>8} {'cerrada':>9} {'g numerica':>12}")
    for nombre, ss in (
        ("rho=-0.4 eta=0.8 g=0.4", SSVIParams(rho=-0.4, eta=0.8, gamma=0.4)),
        ("rho=-0.7 eta=5.0 g=0.1", SSVIParams(rho=-0.7, eta=5.0, gamma=0.1)),
    ):
        for th in (0.02, 0.10):
            ok, d = ss.butterfly_conditions(th)
            libre, g, _, _ = check_butterfly(lambda kk, t=th: ss.total_variance(kk, t))
            print(f"{nombre:<26} {th:>6.2f} {d['theta_phi_factor']:>8.3f} "
                  f"{d['theta_phi2_factor']:>8.3f} {str(ok):>9} "
                  f"{('libre' if libre else 'ARBITRAJE') + f' ({g:.3f})':>12}")

    ss = SSVIParams(rho=-0.4, eta=0.8, gamma=0.4)
    plazos = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    print(f"\nSuperficie completa: {ss.check_surface(0.04 * plazos, plazos).summary()}")


def demo_heston() -> None:
    seccion("Heston: el little Heston trap")
    print("Las dos formulaciones son algebraicamente identicas. La original cruza")
    print("el corte de rama del logaritmo complejo; la estable no.\n")
    p = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.5, rho=-0.7)
    print(f"{'T':>5} {'estable':>12} {'con trampa':>14} {'diferencia':>13}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in (1.0, 2.0, 5.0, 10.0, 15.0, 30.0):
            est = price_call(100.0, 100.0, 0.02, 0.0, t, p, trap=False)
            tra = price_call(100.0, 100.0, 0.02, 0.0, t, p, trap=True)
            dif = "nan" if math.isnan(tra) else f"{abs(est - tra):.4e}"
            tra_s = "nan" if math.isnan(tra) else f"{tra:.6f}"
            print(f"{t:>5.0f} {est:>12.6f} {tra_s:>14} {dif:>13}")
    print("\nA partir de T=10 la formulacion original deja de producir un numero.")

    print("\nValidacion del limite: con xi pequenio, Heston -> Black-Scholes.")
    exacto = bs_call(100.0, 100.0, 0.03, 0.01, 0.2, 1.0)
    print(f"{'xi':>10} {'Heston':>14} {'error vs BS':>14}")
    for xi in (1e-4, 1e-3, 1e-2):
        pp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=xi, rho=0.0)
        h = price_call(100.0, 100.0, 0.03, 0.01, 1.0, pp)
        print(f"{xi:>10.0e} {h:>14.8f} {abs(h - exacto):>14.2e}")
    print("El error escala como xi^2: es la diferencia real del modelo, no error numerico.")


def demo_sabr() -> None:
    seccion("SABR: la expansion de Hagan admite arbitraje a plazo largo")
    print("No es un defecto de implementacion: la formula es una aproximacion de")
    print("primer orden en T, y para plazos largos la densidad se vuelve negativa.\n")
    print(f"{'T':>6} {'nu':>6} {'libre de arbitraje':>20} {'min g':>14}")
    for t in (1.0, 2.0, 5.0, 10.0, 20.0):
        for nu in (0.3, 0.8):
            p = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.3, nu=nu)
            libre, g = p.is_butterfly_free(100.0, t)
            print(f"{t:>6.0f} {nu:>6.1f} {str(libre):>20} {g:>14.5f}")

    print("\nCalibracion sobre datos generados por el propio modelo:")
    verdad = SABRParams(alpha=0.25 * 100.0**0.5, beta=0.5, rho=-0.35, nu=0.45)
    strikes = np.linspace(70.0, 140.0, 15)
    res = calibrate_sabr_slice(
        strikes, hagan_implied_volatility(strikes, 100.0, 1.0, verdad), 100.0, 1.0, beta=0.5
    )
    print(f"  verdad   alpha={verdad.alpha:.6f} rho={verdad.rho:+.6f} nu={verdad.nu:.6f}")
    print(f"  ajustado alpha={res.params.alpha:.6f} rho={res.params.rho:+.6f} nu={res.params.nu:.6f}")
    print(f"  {res}")


def demo_dupire() -> None:
    seccion("Dupire: volatilidad local regularizada")
    print("Caso con solucion exacta: volatilidad implicita constante debe dar")
    print("volatilidad local identica a ella.\n")
    from volsurface import local_volatility

    for sigma in (0.12, 0.27, 0.55):
        res = local_volatility(
            lambda k, t, s=sigma: np.full_like(np.asarray(k, dtype=float), s**2 * t),
            np.linspace(-0.8, 0.8, 17),
            np.array([0.5, 1.0, 2.0]),
        )
        print(f"  sigma implicita {sigma:.2f} -> vol local en [{res.local_volatility.min():.10f}, "
              f"{res.local_volatility.max():.10f}]")

    print("\nSobre cortes SVI con estructura temporal:")
    cortes = {
        0.25: SVIParams(a=0.010, b=0.040, rho=-0.3, m=0.0, sigma=0.25),
        0.50: SVIParams(a=0.020, b=0.055, rho=-0.3, m=0.0, sigma=0.25),
        1.00: SVIParams(a=0.040, b=0.080, rho=-0.3, m=0.0, sigma=0.25),
        2.00: SVIParams(a=0.080, b=0.110, rho=-0.3, m=0.0, sigma=0.25),
    }
    res = local_volatility_from_svi_slices(cortes, np.linspace(-0.4, 0.4, 9))
    print(f"  rango de vol local: [{res.local_volatility.min():.4f}, {res.local_volatility.max():.4f}]")
    print(f"  denominador minimo: {res.min_denominator:.4f}   nodos recortados: {res.n_floored}")
    print(f"  bien planteado: {res.is_well_posed}")


def main() -> None:
    demo_svi()
    demo_ssvi()
    demo_heston()
    demo_sabr()
    demo_dupire()


if __name__ == "__main__":
    main()
