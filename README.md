# volatility-surface-calibration

Calibración de superficies de volatilidad **libres de arbitraje**. SVI y SSVI,
SABR por la expansión de Hagan, Heston por inversión de Fourier y Dupire
regularizado.

Lo que unifica el repositorio es `volsurface.arbitrage`: la función `g` de
Gatheral, cuyo signo **es** el signo de la densidad implícita. Se aplica igual a
cualquier superficie suave, venga del modelo que venga, así que «libre de
arbitraje» deja de ser una afirmación y pasa a ser una medición.

```python
import numpy as np
from volsurface import SVIParams, calibrate_svi_slice, implied_density

k = np.linspace(-0.6, 0.6, 21)
res = calibrate_svi_slice(k, vols_de_mercado, maturity=1.0, forward=100.0)
print(res)
# RMSE 0.0000 puntos de vol, max 0.0000, 8 iteraciones, libre de arbitraje (min g = 2.74e-01)

# Un corte que pasa las condiciones necesarias y aun así admite arbitraje:
malo = SVIParams(a=0.01, b=0.90, rho=-0.95, m=0.05, sigma=0.03)
print(malo.satisfies_necessary_conditions())   # (True, 'cumple las condiciones necesarias')
print(malo.is_butterfly_free())                # (False, -3.63149)
```

---

## El problema está mal nombrado con frecuencia

No es «minimización de restricciones no lineales»: es **mínimos cuadrados no
lineales con restricciones de caja**, y eso determina el algoritmo (región de
confianza reflectiva, que respeta las cotas en cada iterado).

Y «prevención de arbitraje» no es una propiedad que se declare. Son dos
condiciones concretas sobre la varianza total implícita `w(k,T) = σ²(k,T)·T`:

**Mariposa.** La densidad implícita tiene que ser no negativa, lo que equivale a
`g(k) ≥ 0` con

```
g(k) = (1 − k·w'/(2w))² − (w'²/4)(1/w + 1/4) + w''/2
```

La densidad es `p(k) = g(k)·φ(d₂)/√(2πw)`, así que el signo de `g` es el signo de
la densidad. Un `g` negativo es una mariposa con precio negativo: arbitraje puro.

**Calendario.** La varianza total tiene que ser no decreciente en el vencimiento
para cada moneyness fijo.

---

## Resultados de validación

52 pruebas. Reproducible con `python examples/surface_demo.py`.

### SVI: las condiciones necesarias no bastan

| Corte | Condiciones necesarias | Condición de mariposa | Densidad mínima |
|---|---|---|---|
| `a=0.04, b=0.10, ρ=−0.30, σ=0.30` | cumple | **libre**, min g = +0.3147 | positiva |
| `a=0.01, b=0.90, ρ=−0.95, σ=0.03` | **cumple** | **VIOLADA**, min g = −3.6315 | **−4.1677** |

El segundo corte tiene varianza total positiva en todo `k` —pasa todas las
condiciones baratas— y aun así genera densidad negativa. Es exactamente la razón
por la que hace falta evaluar `g`.

La calibración recupera los parámetros de un corte sintético **a 1e-5 en los
cinco parámetros**, con RMSE por debajo de 1e-8 puntos de volatilidad.

### SSVI: las condiciones de Gatheral-Jacquier son suficientes, no necesarias

| Parámetros | θ | θφ(1+\|ρ\|) < 4 | θφ²(1+\|ρ\|) ≤ 4 | Condición cerrada | `g` numérica |
|---|---|---|---|---|---|
| ρ=−0.4, η=0.8, γ=0.4 | 0.02 | 0.106 | 0.400 | cumple | libre (0.331) |
| ρ=−0.4, η=0.8, γ=0.4 | 0.10 | 0.266 | 0.504 | cumple | libre (0.406) |
| ρ=−0.7, η=5.0, γ=0.1 | 0.02 | 0.247 | 1.794 | cumple | libre (0.275) |
| ρ=−0.7, η=5.0, γ=0.1 | 0.10 | 0.982 | **5.674** | **falla** | **libre (0.215)** |

La última fila importa: la condición cerrada falla y la superficie **sí** es
admisible. Tratar una condición suficiente como si fuera necesaria descarta
regiones válidas del espacio de parámetros durante la calibración. El repositorio
ofrece las dos: la cerrada como restricción barata para el optimizador, y `g`
como veredicto definitivo.

### Heston: el *little Heston trap*, demostrado

Las dos formulaciones son **algebraicamente idénticas**. La original cruza el
corte de rama del logaritmo complejo:

| T | Formulación estable | Formulación original | Diferencia |
|---|---|---|---|
| 1 | 8.000349 | 8.000349 | 0 |
| 2 | 11.732476 | 12.004174 | 2.7e-01 |
| 5 | 20.426502 | **8.907706** | **1.2e+01** |
| 10 | 31.316211 | **nan** | — |
| 30 | 58.192358 | **nan** | — |

A T=5 el error es de 11.5 sobre un precio de 20.4. A partir de T=10 la
formulación original deja de producir un número. El síntoma en producción es que
«el optimizador no converge», y se diagnostica mal.

La versión inestable está implementada **a propósito**: existe para que la
prueba pueda demostrar el fallo. Una librería que solo trae la versión correcta
pide que se le crea; esta lo enseña.

**Validación del límite:** con `v0 = θ` y `ξ` pequeño, Heston debe reducirse a
Black-Scholes con `σ = √v₀`.

| ξ | Error contra Black-Scholes |
|---|---|
| 1e-4 | 2.11e-08 |
| 1e-3 | 2.30e-06 |
| 1e-2 | 2.30e-04 |

El error escala como **ξ²** —×10 en ξ da ×100 en error—, que es el orden teórico
de la corrección. Si la implementación tuviera un error de primer orden, el
cociente sería 10 y no 100. Esa es la prueba, no el límite en sí.

El límite `ξ → 0` **no se alcanza numéricamente**: la función característica
divide entre `ξ²`, así que por debajo de `ξ ≈ 1e-5` la cancelación catastrófica
domina (con ξ=1e-7 el error vuelve a subir a 2e-3). La propiedad
`is_numerically_degenerate` lo señala en lugar de dejar que sorprenda.

### SABR: Hagan admite arbitraje a plazo largo

| T | ν=0.3 | ν=0.8 |
|---|---|---|
| 1 | libre (+0.3144) | libre |
| 2 | libre | libre |
| 5 | libre | **arbitraje (−0.2725)** |
| 10 | libre | **arbitraje (−2.5948)** |
| 20 | **arbitraje (−0.1654)** | **arbitraje (−26.0185)** |

No es un defecto de esta implementación: la fórmula de Hagan es una aproximación
asintótica de primer orden en `T`, y para plazos largos o volatilidad de
volatilidad alta la densidad implícita se vuelve negativa. Es la razón por la que
en producción se usa SABR con correcciones, o se pasa a SSVI. El repositorio lo
mide en lugar de ignorarlo.

**Casos límite exactos:** con `β=1` y `ν=0`, SABR da volatilidad constante `α`
para todo strike, a 1e-12. Y la fórmula en el dinero empalma con la general a
1e-10.

El cociente `z/x(z)` es `0/0` en el dinero y se resuelve en serie. El umbral de
conmutación importa: con el valor ingenuo de 1e-10 la fórmula cerrada ya sufre
cancelación catastrófica y el empalme deja un **salto de 2.8e-8 en volatilidad
justo en el strike más consultado**. Con umbral 1e-4 y término cuadrático el
salto baja a **3.8e-11**.

### Dupire: exacto en el caso con solución cerrada

Si la volatilidad implícita es constante, la volatilidad local tiene que ser
idéntica a ella:

| σ implícita | σ local obtenida |
|---|---|
| 0.12 | 0.1200000000 |
| 0.27 | 0.2700000000 |
| 0.55 | 0.5500000000 |

Cualquier error de signo o de factor en la fórmula de Gatheral rompe esta
igualdad.

---

## Qué no hace

- **No ajusta la superficie completa de una vez.** La calibración es por cortes
  (SVI) o global por parámetros (SSVI), no un ajuste simultáneo de todos los
  vencimientos con penalización de suavidad temporal.
- **Sin calibración de Heston a mercado.** El modelo valúa y se puede meter en un
  optimizador, pero no hay una rutina de calibración dedicada: calibrar Heston de
  forma robusta exige multiarranque y regularización, que es un problema propio.
- **Sin modelos de volatilidad rugosa** (rough Bergomi, rough Heston), que son
  hoy un área activa y requieren simulación con núcleos fraccionarios.
- **Sin superficies de tipos de interés ni de divisas.** SABR está implementado
  en la convención de renta variable con forward y strikes; para *swaptions*
  harían falta la convención normal (Bachelier) y el cubo de volatilidad.
- **La interpolación en el vencimiento es lineal en varianza total.** Es la
  elección correcta y no la cómoda: preserva la monotonía en `T`, es decir no
  introduce arbitraje de calendario. Una interpolación por *splines* en `T`
  puede crear tramos decrecientes entre nodos y fabricar arbitraje donde los
  datos no lo tenían. No hay opción de spline por eso.
- **Sin ajuste de dividendos discretos ni de tasas de préstamo de acciones**, que
  en la práctica desplazan el forward y por tanto toda la superficie.

---

## Alcance y decisiones de diseño

**Las derivadas de SVI son analíticas, no numéricas.** `g` depende de `w''`: un
error de diferenciación numérica puede fabricar violaciones de no-arbitraje que
no existen o, peor, ocultar las que sí. Hay una prueba que contrasta las
derivadas analíticas contra las numéricas.

**Donde hay que derivar numéricamente, se usa un esquema de cinco puntos.** Un
esquema de tres puntos es de segundo orden, y su error en `w''` es del mismo
orden que las violaciones de arbitraje que se intentan detectar.

**Se ajusta la varianza total, no la volatilidad.** Es la cantidad en la que SVI
es asintóticamente lineal y en la que se expresan las condiciones de
no-arbitraje. Ajustar volatilidades pondera de forma implícita los vencimientos
cortos.

**Se pondera por vega por omisión.** Un error de un punto de volatilidad vale
dinero muy distinto en el dinero que en el ala. Ajustar sin ponderar equivale a
declarar que todos los strikes importan igual, lo que ningún operador aceptaría.
Se puede desactivar, de forma explícita.

**`beta` de SABR se fija, no se calibra.** `beta` y `rho` controlan ambos la
asimetría del smile y están fuertemente correlacionados: ajustar los cuatro
parámetros a la vez deja el problema prácticamente no identificado.

**Dupire trabaja sobre la varianza total, no sobre precios.** La formulación de
Gatheral está mucho mejor acondicionada: `w` es suave y de magnitud controlada,
mientras que los precios varían varios órdenes de magnitud entre el dinero y las
alas. Y el denominador —que es exactamente `g`— tiene un suelo declarado con
conteo de nodos recortados, porque una volatilidad local recortada es el síntoma
de que la superficie implícita de partida no era admisible.

**El repositorio es autocontenido.** No depende de una librería núcleo compartida
con los demás del conjunto.

---

## Instalación

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python examples/surface_demo.py
```

Requiere Python 3.11 o superior, NumPy y SciPy.

---

## Referencias

- Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces*.
  Quantitative Finance 14(1), 59-71.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley.
  La fórmula de volatilidad local en varianza total es la ecuación (1.10).
- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility
  parameterization*. Global Derivatives & Risk Management, Madrid.
- Hagan, P., Kumar, D., Lesniewski, A. & Woodward, D. (2002). *Managing smile
  risk*. Wilmott Magazine, 84-108.
- Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). *The little
  Heston trap*. Wilmott Magazine, 83-92.
- Heston, S. (1993). *A closed-form solution for options with stochastic
  volatility*. Review of Financial Studies 6(2).
- Dupire, B. (1994). *Pricing with a smile*. Risk Magazine 7(1).
- Roper, M. (2010). *Arbitrage free implied volatility surfaces*. Technical
  report, University of Sydney.
- Lee, R. (2004). *The moment formula for implied volatility at extreme
  strikes*. Mathematical Finance 14(3).

## Licencia

MIT. Ver [LICENSE](LICENSE).
