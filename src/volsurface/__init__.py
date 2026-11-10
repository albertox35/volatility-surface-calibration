"""Calibracion de superficies de volatilidad libres de arbitraje.

Cuatro familias de modelos y, sobre todas ellas, el mismo aparato de
verificacion de no-arbitraje:

- **SVI y SSVI**, el estandar de facto, con las condiciones cerradas de
  Gatheral y Jacquier.
- **SABR** por la expansion asintotica de Hagan, con la advertencia explicita de
  que puede admitir arbitraje para plazos largos.
- **Heston** por inversion de Fourier, con la rama correcta del logaritmo
  complejo.
- **Dupire** regularizado, sobre la varianza total en lugar de sobre precios.

Lo que unifica el repositorio es :mod:`volsurface.arbitrage`: la funcion ``g``
de Gatheral, cuyo signo es el signo de la densidad implicita, se aplica igual a
cualquier superficie suave, venga del modelo que venga.
"""

from __future__ import annotations

from .arbitrage import (
    ArbitrageReport,
    check_butterfly,
    check_calendar,
    g_function,
    implied_density,
    numerical_derivatives,
)
from .calibration import (
    CalibrationResult,
    black_vega,
    calibrate_sabr_slice,
    calibrate_svi_slice,
)
from .dupire import LocalVolResult, local_volatility, local_volatility_from_svi_slices
from .heston import HestonParams, characteristic_function, price_call, price_put
from .sabr import SABRParams, hagan_implied_volatility
from .svi import SSVIParams, SVIParams, svi_derivatives, svi_total_variance

__version__ = "0.1.0"

__all__ = [
    "ArbitrageReport",
    "CalibrationResult",
    "HestonParams",
    "LocalVolResult",
    "SABRParams",
    "SSVIParams",
    "SVIParams",
    "__version__",
    "black_vega",
    "calibrate_sabr_slice",
    "calibrate_svi_slice",
    "characteristic_function",
    "check_butterfly",
    "check_calendar",
    "g_function",
    "hagan_implied_volatility",
    "implied_density",
    "local_volatility",
    "local_volatility_from_svi_slices",
    "numerical_derivatives",
    "price_call",
    "price_put",
    "svi_derivatives",
    "svi_total_variance",
]
