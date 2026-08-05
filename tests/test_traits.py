import numpy as np
import pytest

from opencropphenotyping.traits import compute_crop_cover, create_vegetation_mask


def test_create_vegetation_mask():
    # Create a sample NDVI image with known values
    ndvi_image = np.array([[0.1, 0.4, 0.5],
                           [0.2, 0.3, 0.6],
                           [np.nan, 0.7, 0.8]])

    # Expected vegetation mask with threshold of 0.3
    expected_mask = np.array([[0, 1, 1],
                              [0, 0, 1],
                              [np.nan, 1, 1]])

    # Call the function to create the vegetation mask
    vegetation_mask = create_vegetation_mask(ndvi_image, threshold=0.3)

    # Assert that the generated mask matches the expected mask
    np.testing.assert_array_equal(vegetation_mask, expected_mask)

def test_create_vegetation_mask_threshold():
    ndvi = np.array([
        [0.3, 0.3001]
    ], dtype=np.float32)

    mask = create_vegetation_mask(ndvi, threshold=0.3)

    expected = np.array([
        [0, 1]
    ], dtype=np.float32)

    np.testing.assert_array_equal(mask, expected)

def test_create_vegetation_mask_all_nan():
    ndvi = np.full((2, 2), np.nan, dtype=np.float32)

    mask = create_vegetation_mask(ndvi)

    assert np.all(np.isnan(mask))

def test_compute_crop_cover():
    # Create a sample vegetation mask with known values
    vegetation_mask = np.array([[0, 1, 1],
                                 [0, 0, 1],
                                 [np.nan, 1, 1]])

    # Expected crop cover percentage
    expected_crop_cover = (5 / 8) # 5 vegetation pixels out of 8 valid pixels

    # Call the function to compute crop cover
    crop_cover = compute_crop_cover(vegetation_mask)

    # Assert that the computed crop cover matches the expected value
    assert np.isclose(crop_cover, expected_crop_cover)

def test_compute_crop_cover_no_vegetation():
    mask = np.array([
        [0, 0],
        [0, 0]
    ], dtype=np.float32)
    
    crop_cover = compute_crop_cover(mask)
    assert crop_cover == 0.0

def test_compute_crop_cover_full_vegetation():
    mask = np.array([
        [1, 1],
        [1, 1]
    ], dtype=np.float32)

    crop_cover = compute_crop_cover(mask)

    assert crop_cover == 1.0

def test_compute_crop_cover_all_nan():
    mask = np.full((2, 2), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="The vegetation mask contains only NaN values. Cannot compute crop cover."):
        compute_crop_cover(mask)