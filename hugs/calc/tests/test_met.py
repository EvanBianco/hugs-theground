"""Test the `met` module."""

from hugs.calc import get_wind_dir, get_wind_speed, get_wind_components

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest

def test_speed():
    """Test calculating wind speed."""
    u = np.array([4., 2.,0., 0.])
    v = np.array([0.,2., 4., 0.])

    speed = get_wind_speed(u, v)

    s2 = np.sqrt(2.)
    true_speed = np.array([4., 2 * s2, 4., 0.])

    assert_array_almost_equal(true_speed, speed, 4)


def test_scalar_speed():
    """Test wind speed with scalars."""
    s = get_wind_speed(-3., -4.)
    assert_almost_equal(s, 5., 3)


def test_dir():
    """Test calculating wind direction."""
    u = np.array([4., 2., 0., 0.])
    v = np.array([0., 2., 4., 0.])

    direc = get_wind_dir(u, v)

    true_dir = np.array([270., 225., 180., 270.])

    assert_array_almost_equal(true_dir, direc, 4)

def test_wind_components_scalar():
    """Test calculating wind components using scalar inputs"""
    s = 8.0
    direc = 0
    u, v = get_wind_components(s,direc)
    print(u,v)
    assert_array_almost_equal(u, -0.0, 3)
    assert_array_almost_equal(v, -8.0, 3)

def test_wind_components_array():
    """Test calculating wind components using array inputs"""
    s = np.array([10.0, 10.0, 10.0])
    direc = np.array([0, 45, 90]) 
    u, v = get_wind_components(s,direc)
    print(u)
    true_u = np.array([-0.0, -7.071, -10.0])
    true_v = np.array([-10.0, -7.071, 0.0])
    assert_array_almost_equal(u, true_u, 3)
    assert_array_almost_equal(v, true_v, 3)

def test_warning_direction():
    """Test that warning is raised when wind direction > 360"""
    with pytest.warns(UserWarning):
        get_wind_components(3,480)
