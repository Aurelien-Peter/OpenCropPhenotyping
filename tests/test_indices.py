import numpy as np
import pytest

from opencropphenotyping.indices import _prepare_bands, compute_ndre, compute_ndvi, compute_normalized_difference_index


def test__prepare_bands_shape_error():
    band1 = np.array([[1, 2], [3, 4]])
    band2 = np.array([[5, 6]])
    with pytest.raises(ValueError, match="Input bands must have the same shape."):
        _prepare_bands(band1, band2)


def test_normalized_difference_index_simple():
    band1 = np.array([1, 2])
    band2 = np.array([3, 4])
    expected_ndi = (band1 - band2) / (band1 + band2)
    computed_ndi = compute_normalized_difference_index(band1, band2)
    assert np.allclose(computed_ndi, expected_ndi), "NDI computation failed for simple case."


def test_ndre_simple():
    nir_band = np.array([3, 4])
    red_edge_band = np.array([1, 2])
    expected_ndre = (nir_band - red_edge_band) / (nir_band + red_edge_band)
    computed_ndre = compute_ndre(nir_band, red_edge_band)
    assert np.allclose(computed_ndre, expected_ndre), "NDRE computation failed for simple case."


def test_ndvi_simple():
    red_band = np.array([1, 2])
    nir_band = np.array([3, 4])
    expected_ndvi = (nir_band - red_band) / (nir_band + red_band)
    computed_ndvi = compute_ndvi(red_band, nir_band)
    assert np.allclose(computed_ndvi, expected_ndvi), "NDVI computation failed for simple case."


def test_ndvi_zero_division():
    red_band = np.array([0, 0])
    nir_band = np.array([0, 0])
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
