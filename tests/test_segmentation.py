import numpy as np
from opencropphenotyping.segmentation import threshold_vegetation_index


def test_threshold_vegetation_index():
    # Create a sample NDVI image with known values
    image = np.array([[0.1, 0.4, 0.5],
                           [0.2, 0.3, 0.6],
                           [np.nan, 0.7, 0.8]])

    # Expected vegetation mask with threshold of 0.3
    expected_mask = np.array([[False, True, True],
                              [False, False, True],
                              [False, True, True]], dtype=bool)

    # Call the function to create the vegetation mask
    vegetation_mask = threshold_vegetation_index(image, threshold=0.3)

    # Assert that the generated mask matches the expected mask
    np.testing.assert_array_equal(vegetation_mask, expected_mask)

def test_threshold_vegetation_index_threshold():
    image = np.array([
        [0.3, 0.3001]
    ], dtype=np.float32)

    mask = threshold_vegetation_index(image, threshold=0.3)

    expected = np.array([
        [False, True]
    ], dtype=bool)

    np.testing.assert_array_equal(mask, expected)

def test_threshold_vegetation_index_all_nan():
    image = np.full((2, 2), np.nan, dtype=np.float32)

    mask = threshold_vegetation_index(image, threshold=0.3)
    expected = np.zeros((2, 2), dtype=bool)
    np.testing.assert_array_equal(mask, expected)