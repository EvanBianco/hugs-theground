"""Test the `geology` module."""

from hugs.calc import snell_angle

from numpy.testing import assert_almost_equal
import pytest

def test_snell():
    """Test the basic wind component calculation."""
    res = snell_angle(12, 2500, 4000)
    assert_almost_equal(res, 19.43022, 4)

def test_snell_zero_velocity_top():
    """Test to ensure that a ValueError is raised 
    on the `snell_angle` function when the top
    layer velocity is zero"""
    with pytest.raises(ValueError):
        snell_angle(30,0,2000) 


