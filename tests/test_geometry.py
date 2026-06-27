"""Tests for great-circle distance and soft-argmax localization."""
import numpy as np
import pytest

from src.evaluation.evaluate import soft_argmax_2d, pixel_to_geo
from src.evaluation.spatial_metrics import haversine_distance


def test_haversine_zero_for_same_point():
    assert haversine_distance(12.3, -45.6, 12.3, -45.6) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_at_equator():
    # 1 degree of great circle ~ 111.19 km (R = 6371 km).
    d = haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert d == pytest.approx(111.19, abs=0.5)


def test_haversine_is_symmetric():
    a = haversine_distance(10.0, 20.0, 15.0, 25.0)
    b = haversine_distance(15.0, 25.0, 10.0, 20.0)
    assert a == pytest.approx(b, rel=1e-9)


def test_soft_argmax_recovers_peak_of_one_hot():
    m = np.zeros((20, 20), dtype=np.float32)
    m[7, 13] = 1.0
    y, x = soft_argmax_2d(m, temperature=50.0)
    assert y == pytest.approx(7.0, abs=0.2)
    assert x == pytest.approx(13.0, abs=0.2)


def test_soft_argmax_uniform_map_is_grid_center():
    m = np.ones((11, 11), dtype=np.float32)
    y, x = soft_argmax_2d(m, temperature=10.0)
    assert y == pytest.approx(5.0, abs=1e-6)
    assert x == pytest.approx(5.0, abs=1e-6)


def test_pixel_to_geo_indexes_grid():
    lats = np.array([[10.0, 10.0], [11.0, 11.0]], dtype=np.float32)
    lons = np.array([[-50.0, -49.0], [-50.0, -49.0]], dtype=np.float32)
    lat, lon = pixel_to_geo(1.0, 0.0, lats, lons)
    assert (lat, lon) == (11.0, -50.0)
