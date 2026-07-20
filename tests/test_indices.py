from opencropphenotyping.indices import compute_ndvi
import numpy as np
import pytest

def test_ndvi_simple():
    red_band = np.array([1, 2])
    nir_band = np.array([3, 4])
    expected_ndvi = (nir_band - red_band) / (nir_band + red_band)
    computed_ndvi = compute_ndvi(red_band, nir_band)
    assert np.allclose(computed_ndvi, expected_ndvi), "NDVI computation failed for simple case."

def test_ndvi_shape_error():
    red_band = np.array([1, 2])
    nir_band = np.array([[3, 4], [5, 6]])
    try:
        compute_ndvi(red_band, nir_band)
        assert False, "Expected ValueError for mismatched shapes."
    except ValueError as e:
        assert str(e) == "Red and NIR bands must have the same shape.", "Unexpected error message."

def test_ndvi_zero_division():
    red_band = np.array([0,0])
    nir_band = np.array([0,0])
    expected_ndvi = np.array([0.0, 0.0])
    computed_ndvi = compute_ndvi(red_band, nir_band)
    assert np.allclose(computed_ndvi, expected_ndvi), "NDVI computation failed for zero division case."

def test_ndvi_large_values():
    red_band = np.array([1000, 2000])
    nir_band = np.array([3000, 4000])
    expected_ndvi = (nir_band - red_band) / (nir_band + red_band)
    computed_ndvi = compute_ndvi(red_band, nir_band)
    assert np.allclose(computed_ndvi, expected_ndvi), "NDVI computation failed for large values."

def test_ndvi_type():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    computed_ndvi = compute_ndvi(red_band, nir_band)
    assert computed_ndvi.dtype == np.float32, "NDVI output type is not float32."