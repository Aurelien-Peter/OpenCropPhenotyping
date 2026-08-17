import numpy as np
from opencropphenotyping.segmentation import *
from scipy.ndimage import rotate
import pytest

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

def test_compute_row_profile():
    mask = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
        [1, 1, 1],
    ], dtype=bool)

    expected = np.array([
        0,
        2,
        0,
        3,
    ])

    profile = compute_row_profile(mask)

    np.testing.assert_array_equal(profile, expected)

def test_compute_row_profile_empty():
    mask = np.zeros((10, 20), dtype=bool)

    profile = compute_row_profile(mask)

    expected = np.zeros(10, dtype=int)

    np.testing.assert_array_equal(profile, expected)   

def test_estimate_row_orientation_horizontal_rows():

    mask = np.zeros((100, 200), dtype=bool)

    mask[20:25, :] = True
    mask[45:50, :] = True
    mask[70:75, :] = True

    angle = estimate_row_orientation(
        mask,
        angles=np.arange(-5, 6, 1),
    )

    assert angle == 0

from scipy.ndimage import rotate


def test_estimate_row_orientation_inclined_rows():

    # Create horizontal synthetic crop rows
    mask = np.zeros((100, 200), dtype=bool)

    mask[20:25, :] = True
    mask[45:50, :] = True
    mask[70:75, :] = True

    # Rotate the rows by a known angle
    known_angle = 10

    rotated_mask = rotate(
        mask,
        angle=known_angle,
        reshape=False,
        order=0,
    )

    # Estimate the orientation
    estimated_angle = estimate_row_orientation(
        rotated_mask,
        angles=np.arange(-20, 21, 1),
    )

    # The estimated rotation should compensate for the
    # known inclination.
    assert estimated_angle == -known_angle

def test_detect_crop_rows():
    profile = np.zeros(500)
    profile[[100, 200, 300, 400]] = 100

    peaks = detect_crop_rows(
        profile,
        window_length=11,
        polyorder=2,
        distance=50,
        prominence=10,
    )

    np.testing.assert_array_equal(
        peaks,
        [100, 200, 300, 400],
    )

def test_detect_crop_rows_different_peak_heights():
    profile = np.array([
        0, 0, 1, 8, 1, 0, 0,
        0, 0, 2, 4, 2, 0, 0,
        0, 0, 1, 12, 1, 0, 0,
    ])

    rows = detect_crop_rows(
        profile,
        window_length=5,
        polyorder=2,
        distance=4,
        prominence=3,
    )

    np.testing.assert_array_equal(
        rows,
        [3, 10, 17],
    )

def test_extract_row_images():
    image = np.zeros((100, 200, 3))

    boundaries = np.array([0, 25, 50, 75, 100])

    row_images = extract_row_images(
        image=image,
        boundaries=boundaries,
    )

    assert len(row_images) == 4

    for row_image in row_images:
        assert row_image.shape == (25, 200, 3)

def test_extract_row_images_content():
    image = np.zeros((10, 5, 3))
    image[:5] = 1
    image[5:] = 2

    boundaries = np.array([0, 5, 10])

    row_images = extract_row_images(
        image=image,
        boundaries=boundaries,
    )

    np.testing.assert_array_equal(
        row_images[0],
        image[:5],
    )

    np.testing.assert_array_equal(
        row_images[1],
        image[5:],
    )

def test_segment_row_images():
    row_image = np.array(
        [
            [[10, 30, 10], [10, 10, 10]],
            [[20, 40, 20], [10, 10, 10]],
        ],
        dtype=np.uint8,
    )

    expected = np.array(
        [
            [True, False],
            [True, False],
        ]
    )

    row_masks = segment_row_images(
        row_images=[row_image],
        threshold=20,
    )

    np.testing.assert_array_equal(
        row_masks[0],
        expected,
    )

def test_estimate_plant_positions():
    row_mask = np.zeros((20, 100), dtype=bool)
    row_mask[:, 10:90] = True

    positions = estimate_plant_positions(
        row_mask=row_mask,
        n_plants=4,
        profile_window_length=11,
        profile_polyorder=2,
        row_profile_threshold=10,
    )

    assert len(positions) == 4
    assert np.all(np.diff(positions) > 0)
    assert np.all(positions >= 0)
    assert np.all(positions < row_mask.shape[1])

def test_estimate_plant_positions_no_vegetation():
    row_mask = np.zeros((20, 100), dtype=bool)

    positions = estimate_plant_positions(
        row_mask=row_mask,
        n_plants=4,
        profile_window_length=11,
        profile_polyorder=2,
        row_profile_threshold=10,
    )

    np.testing.assert_array_equal(
        positions,
        np.array([], dtype=int),
    )

def test_estimate_plant_positions_invalid_n_plants():
    row_mask = np.ones((20, 100), dtype=bool)

    with pytest.raises(
        ValueError,
        match="greater than or equal to 2",
    ):
        estimate_plant_positions(
            row_mask=row_mask,
            n_plants=1,
        )

def test_estimate_plant_positions_number_of_plants():
    row_mask = np.zeros((20, 120), dtype=bool)
    row_mask[:, 10:110] = True

    positions = estimate_plant_positions(
        row_mask=row_mask,
        n_plants=6,
        profile_window_length=11,
        profile_polyorder=2,
        row_profile_threshold=10,
    )

    assert len(positions) == 6

def test_define_plant_search_windows():
    plant_positions = np.array([20, 40, 60, 80])

    windows = define_plant_search_windows(
        plant_positions=plant_positions,
        image_width=100,
    )

    expected = [
        (10, 31),
        (30, 51),
        (50, 71),
        (70, 91),
    ]

    assert windows == expected

def test_define_plant_search_windows_image_boundaries():
    plant_positions = np.array([5, 25, 45])

    windows = define_plant_search_windows(
        plant_positions=plant_positions,
        image_width=50,
    )

    for x_start, x_end in windows:
        assert 0 <= x_start < x_end <= 50

def test_define_plant_search_windows_single_position():
    plant_positions = np.array([50])

    with pytest.raises(
        ValueError,
        match="At least two plant positions",
    ):
        define_plant_search_windows(
            plant_positions=plant_positions,
            image_width=100,
        )

def test_define_plant_search_windows_no_positions():
    plant_positions = np.array([], dtype=int)

    with pytest.raises(
        ValueError,
        match="At least two plant positions",
    ):
        define_plant_search_windows(
            plant_positions=plant_positions,
            image_width=100,
        )

def test_count_vegetation_pixels():
    row_mask = np.array([
        [True,  True, False, False],
        [True,  False, False, True],
    ])

    search_windows = [
        (0, 2),
        (2, 4),
    ]

    counts = count_vegetation_pixels(
        row_mask=row_mask,
        search_windows=search_windows,
    )

    assert counts == [3, 1]

def test_count_vegetation_pixels_empty_window():
    row_mask = np.array([
        [True, False, False, False],
        [True, False, False, False],
    ])

    search_windows = [
        (0, 2),
        (2, 4),
    ]

    counts = count_vegetation_pixels(
        row_mask=row_mask,
        search_windows=search_windows,
    )

    assert counts == [2, 0]

def test_compute_vegetation_centroids():
    row_mask = np.array([
        [True, True, False, False],
        [True, False, False, False],
        [False, True, False, False],
    ])

    search_windows = [
        (0, 2),
        (2, 4),
    ]

    centroids = compute_vegetation_centroids(
        row_mask=row_mask,
        search_windows=search_windows,
        row_y_start=0,
    )

    assert centroids[0] == (0.5, 0.75)
    assert centroids[1] is None    

def test_compute_vegetation_centroids_global_x():
    row_mask = np.array([
        [False, False, True, True],
        [False, False, True, False],
    ])

    search_windows = [(2, 4)]

    centroids = compute_vegetation_centroids(
        row_mask=row_mask,
        search_windows=search_windows,
        row_y_start=0,
    )

    assert np.allclose(
        centroids[0],
        (2.3333333333, 0.3333333333),
    )

def test_compute_vegetation_centroids_global_coordinates():
    row_mask = np.array([
        [False, False, True, True],
        [False, False, True, False],
    ])

    search_windows = [(2, 4)]

    centroids = compute_vegetation_centroids(
        row_mask=row_mask,
        search_windows=search_windows,
        row_y_start=100,
    )

    assert np.allclose(
        centroids[0],
        (2.3333333333, 100.3333333333),
    )

def test_build_plant_dataframe():
    plant_positions = np.array([10, 30])

    vegetation_pixel_counts = [50, 20]

    vegetation_centroids = [
        (12.0, 5.0),
        (27.0, 6.0),
    ]

    plant_df = build_plant_dataframe(
        plant_positions=plant_positions,
        vegetation_pixel_counts=vegetation_pixel_counts,
        vegetation_centroids=vegetation_centroids,
    )

    expected_columns = [
        "plant_position",
        "expected_x",
        "row_profile_pixels",
        "centroid_x",
        "centroid_y",
        "offset",
        "abs_offset",
    ]

    assert list(plant_df.columns) == expected_columns
    assert len(plant_df) == 2

    assert plant_df.loc[0, "plant_position"] == 1
    assert plant_df.loc[0, "expected_x"] == 10
    assert plant_df.loc[0, "row_profile_pixels"] == 50
    assert plant_df.loc[0, "offset"] == 2
    assert plant_df.loc[0, "abs_offset"] == 2

    assert plant_df.loc[1, "offset"] == -3
    assert plant_df.loc[1, "abs_offset"] == 3

def test_build_plant_dataframe_missing_centroid():
    plant_positions = np.array([10])

    vegetation_pixel_counts = [5]

    vegetation_centroids = [None]

    plant_df = build_plant_dataframe(
        plant_positions=plant_positions,
        vegetation_pixel_counts=vegetation_pixel_counts,
        vegetation_centroids=vegetation_centroids,
    )

    assert np.isnan(plant_df.loc[0, "centroid_x"])
    assert np.isnan(plant_df.loc[0, "centroid_y"])
    assert np.isnan(plant_df.loc[0, "offset"])
    assert np.isnan(plant_df.loc[0, "abs_offset"])

def test_identify_missing_plant_candidates_detects_outlier():
    plant_df = pd.DataFrame({
        "plant_position": [1, 2, 3, 4, 5],
        "expected_x": [10, 20, 30, 40, 50],
        "row_profile_pixels": [100, 105, 98, 5, 102],
        "centroid_x": [10, 20, 30, 60, 50],
        "centroid_y": [5, 5, 5, 5, 5],
        "offset": [0, 0, 0, 20, 0],
        "abs_offset": [0, 0, 0, 20, 0],
    })

    result = identify_missing_plant_candidates(
        plant_df
    )

    assert result.loc[3, "missing_candidate"]
    assert not result.loc[0, "missing_candidate"]
    assert not result.loc[1, "missing_candidate"]
    assert not result.loc[2, "missing_candidate"]
    assert not result.loc[4, "missing_candidate"]

def test_identify_missing_plant_candidates_does_not_modify_input():
    plant_df = pd.DataFrame({
        "row_profile_pixels": [100, 100, 5, 100],
        "abs_offset": [0, 0, 20, 0],
    })

    original_columns = plant_df.columns.tolist()

    result = identify_missing_plant_candidates(
        plant_df
    )

    assert plant_df.columns.tolist() == original_columns
    assert "missing_candidate" not in plant_df.columns
    assert "missing_candidate" in result.columns