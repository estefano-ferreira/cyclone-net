"""
Physics correctness tests for diagnostic channels.

These validate that the finite-difference vorticity/divergence recover the
ANALYTIC values for flows whose vorticity and divergence are known in closed
form. This is the core scientific guarantee behind the 'vorticity'/'divergence'
input channels: if these break, every downstream physical claim is invalid.
"""
import numpy as np
import pytest

from src.physics.diagnostics import (
    divergence,
    estimate_dx_dy_meters,
    vorticity,
    wind_speed,
)

DX = 25_000.0  # meters
DY = 27_000.0
N = 30


def _meter_grid():
    i = np.arange(N)
    j = np.arange(N)
    J, I = np.meshgrid(j, i)  # (N,N); X varies with j (cols), Y with i (rows)
    X = J * DX
    Y = I * DY
    return X.astype(np.float32), Y.astype(np.float32)


def _interior(a):
    return a[2:-2, 2:-2]


def test_solid_body_rotation_has_constant_vorticity_and_zero_divergence():
    omega = 3.0e-5  # s^-1
    X, Y = _meter_grid()
    xc, yc = X.mean(), Y.mean()
    u = (-omega * (Y - yc)).astype(np.float32)
    v = (omega * (X - xc)).astype(np.float32)

    vort = vorticity(u, v, DX, DY)
    div = divergence(u, v, DX, DY)

    # Analytic: vorticity = 2*omega everywhere, divergence = 0.
    assert np.allclose(_interior(vort), 2.0 * omega, rtol=1e-3, atol=1e-9)
    assert np.allclose(_interior(div), 0.0, atol=1e-9)


def test_pure_divergence_field_has_zero_vorticity():
    k = 2.0e-5
    X, Y = _meter_grid()
    xc, yc = X.mean(), Y.mean()
    u = (k * (X - xc)).astype(np.float32)
    v = (k * (Y - yc)).astype(np.float32)

    vort = vorticity(u, v, DX, DY)
    div = divergence(u, v, DX, DY)

    # Analytic: divergence = 2k, vorticity = 0.
    assert np.allclose(_interior(div), 2.0 * k, rtol=1e-3, atol=1e-9)
    assert np.allclose(_interior(vort), 0.0, atol=1e-9)


def test_wind_speed_is_euclidean_norm():
    u = np.full((5, 5), 3.0, dtype=np.float32)
    v = np.full((5, 5), 4.0, dtype=np.float32)
    assert np.allclose(wind_speed(u, v), 5.0)


def test_estimate_dx_dy_matches_haversine_scale_at_latitude():
    # 0.25-degree grid centred near 20N.
    lat1d = 20.0 + 0.25 * np.arange(8)
    lon1d = -60.0 + 0.25 * np.arange(8)
    lons, lats = np.meshgrid(lon1d, lat1d)
    dx, dy = estimate_dx_dy_meters(lats.astype(np.float32), lons.astype(np.float32))

    expected_dy = 0.25 * 111_320.0
    expected_dx = 0.25 * 111_320.0 * np.cos(np.radians(lats.mean()))
    assert dy == pytest.approx(expected_dy, rel=1e-3)
    assert dx == pytest.approx(expected_dx, rel=1e-2)
