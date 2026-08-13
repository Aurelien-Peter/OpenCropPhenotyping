import numpy as np
import pytest

from opencropphenotyping.indices import (
    _prepare_bands,
    compute_gndvi,
    compute_indexes,
    compute_ndre,
    compute_ndvi,
    compute_normalized_difference_index,
    compute_savi,
    compute_exg
)


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

def test_ndre_simple():
    nir_band = np.array([3, 4])
    red_edge_band = np.array([1, 2])
    expected_ndre = (nir_band - red_edge_band) / (nir_band + red_edge_band)
    computed_ndre = compute_ndre(nir_band, red_edge_band)
    assert np.allclose(computed_ndre, expected_ndre), "NDRE computation failed for simple case."

def test_ndre_zero_division():
    red_edge_band = np.array([0, 0])
    nir_band = np.array([0, 0])
    expected_ndre = np.array([0.0, 0.0])
    computed_ndre = compute_ndre(nir_band, red_edge_band)
    assert np.allclose(computed_ndre, expected_ndre), "ndre computation failed for zero division case."


def test_ndre_large_values():
    red_edge_band = np.array([1000, 2000])
    nir_band = np.array([3000, 4000])
    expected_ndre = (nir_band - red_edge_band) / (nir_band + red_edge_band)
    computed_ndre = compute_ndre(nir_band, red_edge_band)
    assert np.allclose(computed_ndre, expected_ndre), "ndre computation failed for large values."


def test_ndre_type():
    red_edge_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    computed_ndre = compute_ndre(nir_band, red_edge_band)
    assert computed_ndre.dtype == np.float32, "ndre output type is not float32."

def test_gndvi_simple():
    nir_band = np.array([3, 4])
    green_band = np.array([1, 2])
    expected_gndvi = (nir_band - green_band) / (nir_band + green_band)
    computed_gndvi = compute_gndvi(nir_band, green_band)
    assert np.allclose(computed_gndvi, expected_gndvi), "gndvi computation failed for simple case."

def test_gndvi_zero_division():
    green_band = np.array([0, 0])
    nir_band = np.array([0, 0])
    expected_gndvi = np.array([0.0, 0.0])
    computed_gndvi = compute_gndvi(nir_band, green_band)
    assert np.allclose(computed_gndvi, expected_gndvi), "gndvi computation failed for zero division case."


def test_gndvi_large_values():
    green_band = np.array([1000, 2000])
    nir_band = np.array([3000, 4000])
    expected_gndvi = (nir_band - green_band) / (nir_band + green_band)
    computed_gndvi = compute_gndvi(nir_band, green_band)
    assert np.allclose(computed_gndvi, expected_gndvi), "gndvi computation failed for large values."

def test_gndvi_type():
    green_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    computed_gndvi = compute_gndvi(nir_band, green_band)
    assert computed_gndvi.dtype == np.float32, "gndvi output type is not float32."

def test_savi_simple():
    nir_band = np.array([3, 4])
    red_band = np.array([1, 2])
    L_factor = 0.5
    expected_savi = (
    (nir_band - red_band)
    * (1 + L_factor)
    / (nir_band + red_band + L_factor)
    )
    computed_savi = compute_savi(red_band, nir_band, L_factor = 0.5)
    assert np.allclose(computed_savi, expected_savi), "savi computation failed for simple case."


def test_savi_zero_division():
    red_band = np.array([0, 0])
    nir_band = np.array([0, 0])
    expected_savi = np.array([0.0, 0.0])
    computed_savi = compute_savi(red_band, nir_band)
    assert np.allclose(computed_savi, expected_savi), "savi computation failed for zero division case."


def test_savi_large_values():
    red_band = np.array([1000, 2000])
    nir_band = np.array([3000, 4000])
    L_factor = 0.5
    expected_savi = (
    (nir_band - red_band)
    * (1 + L_factor)
    / (nir_band + red_band + L_factor)
    )
    computed_savi = compute_savi(red_band, nir_band, L_factor = 0.5)
    assert np.allclose(computed_savi, expected_savi), "savi computation failed for large values."


def test_savi_type():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    computed_savi = compute_savi(red_band, nir_band, L_factor = 0.5)
    assert computed_savi.dtype == np.float32, "savi output type is not float32."

def test_savi_negative_L_factor():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    with pytest.raises(ValueError, match="L_factor must be greater than or equal to 0."):
        compute_savi(red_band, nir_band, L_factor = -1)

def test_compute_indexes_all_indexes():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    green_band = np.array([5, 6], dtype=np.int32)
    red_edge_band = np.array([7, 8], dtype=np.int32)

    indices = compute_indexes(
        red_band=red_band,
        nir_band=nir_band,
        green_band=green_band,
        red_edge_band=red_edge_band
    )

    assert "ndvi" in indices
    assert "ndre" in indices
    assert "gndvi" in indices
    assert "savi" in indices
    assert len(indices) == 4

def test_compute_indexes_red_nir():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)

    indices = compute_indexes(
        red_band=red_band,
        nir_band=nir_band,
    )

    assert "ndvi" in indices
    assert "savi" in indices
    assert "ndre" not in indices
    assert "gndvi" not in indices
    assert len(indices) == 2

    assert np.allclose(indices["ndvi"], compute_ndvi(red_band, nir_band))
    assert np.allclose(indices["savi"], compute_savi(red_band, nir_band))

def test_compute_indexes_green_nir():
    green_band = np.array([5, 6], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)

    indices = compute_indexes(
        nir_band=nir_band,
        green_band=green_band, 
    )

    assert "ndvi" not in indices
    assert "ndre" not in indices
    assert "gndvi" in indices
    assert "savi" not in indices
    assert len(indices) == 1

    assert np.allclose(indices["gndvi"], compute_gndvi(nir_band, green_band))


def test_compute_indexes_rededge_nir():
    red_edge_band = np.array([7, 8], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)

    indices = compute_indexes(
        nir_band=nir_band,
        red_edge_band=red_edge_band
    )

    assert "ndvi" not in indices
    assert "ndre" in indices
    assert "gndvi" not in indices
    assert "savi" not in indices
    assert len(indices) == 1

    assert np.allclose(indices["ndre"], compute_ndre(nir_band, red_edge_band))


def test_compute_indexes_no_nir():
    red_band = np.array([1, 2], dtype=np.int32)
    green_band = np.array([5, 6], dtype=np.int32)
    red_edge_band = np.array([7, 8], dtype=np.int32)

    indices = compute_indexes(
        red_band=red_band,
        green_band=green_band, 
        red_edge_band=red_edge_band
    )

    assert indices == {}

def test_compute_indexes_selected_indices():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    green_band = np.array([5, 6], dtype=np.int32)
    red_edge_band = np.array([7, 8], dtype=np.int32)

    indices = compute_indexes(
        red_band=red_band,
        nir_band=nir_band,
        green_band=green_band,
        red_edge_band=red_edge_band,
        indices=["ndvi", "gndvi"]
    )

    assert "ndvi" in indices
    assert "gndvi" in indices
    assert "ndre" not in indices
    assert "savi" not in indices
    assert len(indices) == 2

    assert np.allclose(indices["ndvi"], compute_ndvi(red_band, nir_band))
    assert np.allclose(indices["gndvi"], compute_gndvi(nir_band, green_band))

def test_compute_indexes_empty_selected_indices():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    green_band = np.array([5, 6], dtype=np.int32)
    red_edge_band = np.array([7, 8], dtype=np.int32)

    indices = compute_indexes(
        red_band=red_band,
        nir_band=nir_band,
        green_band=green_band,
        red_edge_band=red_edge_band,
        indices=[]
    )

    assert indices == {}

def test_compute_indexes_none_indices():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)
    green_band = np.array([5, 6], dtype=np.int32)
    red_edge_band = np.array([7, 8], dtype=np.int32)

    indices = compute_indexes(
        red_band=red_band,
        nir_band=nir_band,
        green_band=green_band,
        red_edge_band=red_edge_band,
        indices=None
    )

    assert "ndvi" in indices
    assert "gndvi" in indices
    assert "ndre" in indices
    assert "savi" in indices
    assert len(indices) == 4

def test_compute_indexes_missing_band_for_selected_index():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)

    indices = compute_indexes(
        red_band=red_band,
        nir_band=nir_band,
        indices=["ndvi", "ndre"],
    )

    assert "ndvi" in indices
    assert "ndre" not in indices
    assert len(indices) == 1

def test_compute_indexes_unknown_index():
    red_band = np.array([1, 2], dtype=np.int32)
    nir_band = np.array([3, 4], dtype=np.int32)

    with pytest.raises(ValueError, match="Unknown vegetation index"):
        compute_indexes(
            red_band=red_band,
            nir_band=nir_band,
            indices=["unknown"],
        )

def test_exg_simple():
    rgb_band = np.array(
        [
            [[10, 20, 30], [20, 30, 40]],
            [[30, 40, 50], [40, 50, 60]],
        ],
        dtype=np.uint8,
    )

    expected_exg = np.array(
        [
            [0, 0],
            [0, 0],
        ],
        dtype=np.float32,
    )

    computed_exg = compute_exg(rgb_band)
    assert np.allclose(computed_exg, expected_exg), "exg computation failed for simple case."

def test_exg_large_values():
    rgb_band = np.array(
        [
            [[200, 250, 240], [250, 255, 200]],
            [[180, 240, 220], [220, 250, 230]],
        ],
        dtype=np.uint8,
    )

    red = rgb_band[:, :, 0].astype(np.float32)
    green = rgb_band[:, :, 1].astype(np.float32)
    blue = rgb_band[:, :, 2].astype(np.float32)

    expected_exg = 2 * green - red - blue

    computed_exg = compute_exg(rgb_band)
    assert np.allclose(computed_exg, expected_exg), "exg computation failed for large values."

def test_exg_large_array():
    rgb_band = np.random.randint(
        0,
        256,
        size=(1000, 1000, 3),
        dtype=np.uint8,
    )

    red = rgb_band[:, :, 0].astype(np.float32)
    green = rgb_band[:, :, 1].astype(np.float32)
    blue = rgb_band[:, :, 2].astype(np.float32)

    expected_exg = 2 * green - red - blue

    computed_exg = compute_exg(rgb_band)
    assert np.allclose(computed_exg, expected_exg), "exg computation failed for large values."


def test_exg_type():
    rgb_band = np.random.randint(
        0,
        256,
        size=(10, 10, 3),
        dtype=np.uint8,
    )
    computed_exg = compute_exg(rgb_band)
    assert computed_exg.dtype == np.float32, "exg output type is not float32."