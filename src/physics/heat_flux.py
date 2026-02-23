"""
CycloneNet: Heat flux calculations (latent + sensible) from ERA5 fields.
Based on bulk aerodynamic formulas (COARE 3.0 simplified).
All comments and variable names in English.
"""

import numpy as np

# Physical constants
Lv = 2.5e6          # latent heat of vaporization (J/kg)
Cp = 1005.0         # specific heat of air at constant pressure (J/kg/K)
rho_air = 1.2       # air density (kg/m³) - mean sea level value
epsilon = 0.622     # molecular weight ratio (water vapor/dry air)


def saturation_vapor_pressure(temp_k: np.ndarray) -> np.ndarray:
    """
    Compute saturation vapor pressure (Pa) using Tetens formula.
    temp_k: temperature in Kelvin.
    """
    temp_c = temp_k - 273.15
    e_sat = 611.2 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    return e_sat  # Pa


def specific_humidity_from_dewpoint(dewpoint_k: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    """
    Compute specific humidity (kg/kg) from dew point temperature and pressure.
    """
    e = saturation_vapor_pressure(dewpoint_k)
    q = epsilon * e / (pressure_pa - (1 - epsilon) * e)
    return q


def latent_heat_flux(wind_speed: np.ndarray, sst_k: np.ndarray, q_air: np.ndarray,
                     pressure_pa: np.ndarray, Ce: float = 1.2e-3) -> np.ndarray:
    """
    Latent heat flux (W/m²) = rho_air * Lv * Ce * U * (q_sat(sst) - q_air)
    """
    q_sat = specific_humidity_from_dewpoint(
        sst_k, pressure_pa)   # assuming SST ~ dew point at surface
    return rho_air * Lv * Ce * wind_speed * (q_sat - q_air)


def sensible_heat_flux(wind_speed: np.ndarray, sst_k: np.ndarray, temp_air_k: np.ndarray,
                       Ch: float = 1.2e-3) -> np.ndarray:
    """
    Sensible heat flux (W/m²) = rho_air * Cp * Ch * U * (SST - T_air)
    """
    return rho_air * Cp * Ch * wind_speed * (sst_k - temp_air_k)


def total_heat_flux(lhf: np.ndarray, shf: np.ndarray) -> np.ndarray:
    """Total heat flux (W/m²) = latent + sensible."""
    return lhf + shf


def compute_heat_fluxes(
    sst: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    msl: np.ndarray,
    t2m: np.ndarray = None,        # 2m air temperature (K) – optional
    d2m: np.ndarray = None,        # 2m dew point temperature (K) – optional
    Ce: float = 1.2e-3,
    Ch: float = 1.2e-3
) -> dict:
    """
    Compute latent, sensible, and total heat fluxes from ERA5 fields.
    If t2m or d2m are not provided, approximate values are used (SST - 1K for t2m,
    and t2m - 2K for d2m). Returns a dictionary with arrays of same shape as inputs.
    """
    wind_speed = np.sqrt(u10**2 + v10**2)
    pressure = msl  # already in Pa

    # Fallback approximations if missing
    if t2m is None:
        t2m = sst - 1.0      # air slightly cooler than surface
    if d2m is None:
        d2m = t2m - 2.0      # typical dew point depression

    q_air = specific_humidity_from_dewpoint(d2m, pressure)

    lhf = latent_heat_flux(wind_speed, sst, q_air, pressure, Ce)
    shf = sensible_heat_flux(wind_speed, sst, t2m, Ch)
    thf = total_heat_flux(lhf, shf)

    return {
        'latent_heat_flux': lhf,
        'sensible_heat_flux': shf,
        'total_heat_flux': thf
    }
