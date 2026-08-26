from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import rotate
from shapely.geometry import box

from opencropphenotyping.evaluation import transform_original_points_to_rotated
from opencropphenotyping.segmentation import (
    build_plant_dataframe,
    compute_rotation_angle,
    compute_row_boundaries_from_peaks,
    compute_row_boundaries_from_plot,
    compute_row_profile,
    compute_vegetation_centroids,
    compute_vegetation_fraction,
    count_vegetation_pixels,
    define_plant_search_windows,
    define_plant_segments,
    detect_crop_rows,
    detect_plants,
    detect_plants_in_row,
    estimate_plant_positions_from_plot,
    estimate_plant_positions_from_vegetation,
    estimate_row_orientation,
    extract_row_images,
    get_mask_bounds,
    identify_missing_plant_candidates_from_veg_threshold_and_centroid,
    identify_missing_plant_candidates_from_vegetation_fraction,
    load_plot_metadata,
    prepare_rotated_data,
    segment_row_images,
    threshold_vegetation_index,
)


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

def test_compute_row_boundaries_from_peaks():
    row_positions = np.array([100, 200, 300])

    boundaries = compute_row_boundaries_from_peaks(
        row_positions=row_positions,
        image_height=400,
    )

    np.testing.assert_array_equal(
        boundaries,
        [50, 150, 250, 350],
    )  

def test_compute_row_boundaries_from_peaks_single_row():
    row_positions = np.array([200])

    boundaries = compute_row_boundaries_from_peaks(
        row_positions=row_positions,
        image_height=400,
    )

    np.testing.assert_array_equal(
        boundaries,
        [0, 400],
    )  

def test_compute_row_boundaries_from_peaks_no_rows():
    row_positions = np.array([], dtype=int)

    boundaries = compute_row_boundaries_from_peaks(
        row_positions=row_positions,
        image_height=400,
    )

    np.testing.assert_array_equal(
        boundaries,
        np.array([]),
    )

def test_compute_row_boundaries_from_plot():
    boundaries = compute_row_boundaries_from_plot(
        y_start=100,
        y_end=400,
        n_rows=3,
    )

    np.testing.assert_array_equal(
        boundaries,
        [100, 200, 300, 400],
    )

def test_compute_row_boundaries_from_plot_single_row():
    boundaries = compute_row_boundaries_from_plot(
        y_start=100,
        y_end=400,
        n_rows=1,
    )

    np.testing.assert_array_equal(
        boundaries,
        [100, 400],
    )

def test_compute_row_boundaries_from_plot_invalid_n_rows():
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        compute_row_boundaries_from_plot(
            y_start=100,
            y_end=400,
            n_rows=0,
        )

def test_compute_row_boundaries_from_plot_with_bounds():
    boundaries = compute_row_boundaries_from_plot(
        y_start=37,
        y_end=1634,
        n_rows=7,
    )

    assert boundaries[0] == 37
    assert boundaries[-1] == 1634
    assert np.all(np.diff(boundaries) > 0)
        
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

def test_estimate_plant_positions_from_vegetation():
    row_mask = np.zeros((20, 100), dtype=bool)
    row_mask[:, 10:90] = True

    positions = estimate_plant_positions_from_vegetation(
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

def test_estimate_plant_positions_from_vegetation_no_vegetation():
    row_mask = np.zeros((20, 100), dtype=bool)

    positions = estimate_plant_positions_from_vegetation(
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

def test_estimate_plant_positions_from_vegetation_invalid_n_plants():
    row_mask = np.ones((20, 100), dtype=bool)

    with pytest.raises(
        ValueError,
        match="greater than or equal to 2",
    ):
        estimate_plant_positions_from_vegetation(
            row_mask=row_mask,
            n_plants=1,
        )

def test_estimate_plant_positions_from_vegetation_number_of_plants():
    row_mask = np.zeros((20, 120), dtype=bool)
    row_mask[:, 10:110] = True

    positions = estimate_plant_positions_from_vegetation(
        row_mask=row_mask,
        n_plants=6,
        profile_window_length=11,
        profile_polyorder=2,
        row_profile_threshold=10,
    )

    assert len(positions) == 6

def test_estimate_plant_positions_from_plot():
    positions = estimate_plant_positions_from_plot(
        x_start=10,
        x_end=90,
        n_plants=4,
    )

    np.testing.assert_array_equal(
        positions,
        [20, 40, 60, 80],
    )

def test_estimate_plant_positions_from_plot_invalid_n_plants():
    with pytest.raises(
        ValueError,
        match="greater than or equal to 2",
    ):
        estimate_plant_positions_from_plot(
            x_start=10,
            x_end=90,
            n_plants=1,
        )

def test_estimate_plant_positions_from_plot_with_bounds():
    positions = estimate_plant_positions_from_plot(
        x_start=15,
        x_end=1675,
        n_plants=20,
    )

    assert np.all(positions > 15)
    assert np.all(positions < 1675)
    assert np.all(np.diff(positions) > 0)

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

    assert centroids[0] is not None
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

    assert centroids[0] is not None
    assert np.allclose(
        centroids[0],
        (2.3333333333, 100.3333333333),
    )

def test_build_plant_dataframe():
    plant_positions = np.array([10, 30])

    vegetation_pixel_counts = [50, 20]

    vegetation_centroids: list[tuple[float, float] | None] = [
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

    vegetation_centroids: list[tuple[float, float] | None] = [None]

    plant_df = build_plant_dataframe(
        plant_positions=plant_positions,
        vegetation_pixel_counts=vegetation_pixel_counts,
        vegetation_centroids=vegetation_centroids,
    )
    assert plant_df.loc[0] is not None
    assert pd.isna(plant_df.loc[0, "centroid_x"])
    assert pd.isna(plant_df.loc[0, "centroid_y"])
    assert pd.isna(plant_df.loc[0, "offset"])
    assert pd.isna(plant_df.loc[0, "abs_offset"])

def test_identify_missing_plant_candidates_from_veg_threshold_and_centroid_detects_outlier():
    plant_df = pd.DataFrame({
        "plant_position": [1, 2, 3, 4, 5],
        "expected_x": [10, 20, 30, 40, 50],
        "row_profile_pixels": [100, 105, 98, 5, 102],
        "centroid_x": [10, 20, 30, 60, 50],
        "centroid_y": [5, 5, 5, 5, 5],
        "offset": [0, 0, 0, 20, 0],
        "abs_offset": [0, 0, 0, 20, 0],
    })

    result = identify_missing_plant_candidates_from_veg_threshold_and_centroid(
        plant_df
    )

    assert result.loc[3, "missing_candidate"]
    assert not result.loc[0, "missing_candidate"]
    assert not result.loc[1, "missing_candidate"]
    assert not result.loc[2, "missing_candidate"]
    assert not result.loc[4, "missing_candidate"]

def test_identify_missing_plant_candidates_from_veg_threshold_and_centroid_does_not_modify_input():
    plant_df = pd.DataFrame({
        "row_profile_pixels": [100, 100, 5, 100],
        "abs_offset": [0, 0, 20, 0],
    })

    original_columns = plant_df.columns.tolist()

    result = identify_missing_plant_candidates_from_veg_threshold_and_centroid(
        plant_df
    )

    assert plant_df.columns.tolist() == original_columns
    assert "missing_candidate" not in plant_df.columns
    assert "missing_candidate" in result.columns

def test_compute_vegetation_fraction():
    row_mask = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 1],
    ])

    search_windows = [
        (0, 2),
        (2, 4),
    ]

    result = compute_vegetation_fraction(
        row_mask=row_mask,
        search_windows=search_windows,
    )

    expected = [
        3 / 4,
        1 / 4,
    ]

    assert result == expected


def test_compute_vegetation_fraction_full_vegetation():
    row_mask = np.ones(
        (2, 4),
        dtype=np.uint8,
    )

    result = compute_vegetation_fraction(
        row_mask=row_mask,
        search_windows=[(0, 4)],
    )

    assert result == [1.0]


def test_compute_vegetation_fraction_no_vegetation():
    row_mask = np.zeros(
        (2, 4),
        dtype=np.uint8,
    )

    result = compute_vegetation_fraction(
        row_mask=row_mask,
        search_windows=[(0, 4)],
    )

    assert result == [0.0]


def test_compute_vegetation_fraction_empty_window():
    row_mask = np.ones(
        (2, 4),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="is empty",
    ):
        compute_vegetation_fraction(
            row_mask=row_mask,
            search_windows=[(2, 2)],
        )

def test_identify_missing_plant_candidates_from_vegetation_fraction():
    plant_df = pd.DataFrame({
        "vegetation_fraction": [
            0.10,
            0.04,
            0.03,
            0.00,
        ],
    })

    result = identify_missing_plant_candidates_from_vegetation_fraction(
        plant_df
    )

    assert list(result["missing_candidate"]) == [
        False,
        False,
        True,
        True,
    ]

def test_identify_missing_plant_candidates_custom_threshold():
    plant_df = pd.DataFrame({
        "vegetation_fraction": [
            0.10,
            0.05,
            0.02,
        ],
    })

    result = identify_missing_plant_candidates_from_vegetation_fraction(
        plant_df,
        vegetation_fraction_threshold=0.05,
    )

    assert list(result["missing_candidate"]) == [
        False,
        False,
        True,
    ]

def test_identify_missing_plant_candidates_does_not_modify_input():
    plant_df = pd.DataFrame({
        "vegetation_fraction": [0.10, 0.02],
    })

    identify_missing_plant_candidates_from_vegetation_fraction(
        plant_df
    )

    assert "missing_candidate" not in plant_df.columns

def test_define_plant_segments():
    result = define_plant_segments(
        x_start=0,
        x_end=100,
        n_segments=4,
    )

    expected = [
        (0.0, 25.0),
        (25.0, 50.0),
        (50.0, 75.0),
        (75.0, 100.0),
    ]

    assert result == expected


def test_define_plant_segments_single_segment():
    result = define_plant_segments(
        x_start=10,
        x_end=50,
        n_segments=1,
    )

    assert result == [
        (10.0, 50.0),
    ]


@pytest.mark.parametrize(
    "n_segments",
    [0, -1, -5],
)
def test_define_plant_segments_invalid_number(
    n_segments,
):
    with pytest.raises(
        ValueError,
        match="greater than or equal to 1",
    ):
        define_plant_segments(
            x_start=0,
            x_end=100,
            n_segments=n_segments,
        )

@pytest.mark.parametrize(
    ("x_start", "x_end"),
    [
        (10, 10),
        (100, 0),
    ],
)
def test_define_plant_segments_invalid_bounds(
    x_start,
    x_end,
):
    with pytest.raises(
        ValueError,
        match="x_end must be greater than x_start",
    ):
        define_plant_segments(
            x_start=x_start,
            x_end=x_end,
            n_segments=4,
        ) 

def test_transform_points_zero_angle():
    points = np.array([
        [0, 0],
        [50, 25],
        [99, 49],
    ])

    result = transform_original_points_to_rotated(
        points=points,
        angle=0,
        original_shape=(50, 100),
        rotated_shape=(50, 100),
    )

    np.testing.assert_allclose(
        result,
        points,
    )

def test_transform_center_point():
    points = np.array([
        [49.5, 24.5],
    ])

    result = transform_original_points_to_rotated(
        points=points,
        angle=45,
        original_shape=(50, 100),
        rotated_shape=(106, 106),
    )

    expected = np.array([
        [(106 - 1) / 2, (106 - 1) / 2],
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_transform_points_90_degrees():
    points = np.array([
        [3.0, 2.0],
    ])

    result = transform_original_points_to_rotated(
        points=points,
        angle=90,
        original_shape=(5, 5),
        rotated_shape=(5, 5),
    )

    expected = np.array([
        [2.0, 1.0],
    ])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-6,
    )

def test_get_mask_bounds():
    mask = np.zeros((5, 10), dtype=np.uint8)

    mask[1:4, 2:7] = 1

    result = get_mask_bounds(mask)

    assert result == (2, 6, 1, 3)

def test_get_mask_bounds_single_pixel():
    mask = np.zeros((5, 10), dtype=np.uint8)

    mask[3, 7] = 1

    result = get_mask_bounds(mask)

    assert result == (7, 7, 3, 3)

def test_get_mask_bounds_empty_mask():
    mask = np.zeros(
        (5, 10),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="does not contain any positive pixels",
    ):
        get_mask_bounds(mask)

def test_load_plot_metadata(tmp_path):
    geometry = box(
        0,
        0,
        10,
        10,
    )

    gdf = gpd.GeoDataFrame(
        {
            "image_name": ["image_01.png"],
            "n_plants": [10],
            "n_rows": [4],
            "geometry": [geometry],
        },
        crs="EPSG:3857",
    )

    geopackage_path = tmp_path / "plots.gpkg"

    gdf.to_file(
        geopackage_path,
        driver="GPKG",
    )

    result = load_plot_metadata(
        geopackage_path
    )

    image_name, n_plants, n_rows, result_geometry = result

    assert image_name == "image_01.png"
    assert n_plants == 10
    assert n_rows == 4
    assert result_geometry.equals(geometry)

def test_compute_rotation_angle_with_provided_angle(tmp_path):
    image_path = tmp_path / "image.png"

    result = compute_rotation_angle(
        image_path=image_path,
        threshold=25,
        angle=42.5,
    )

    assert result == 42.5

def test_compute_rotation_angle_estimates_angle(
    monkeypatch,
):

    def fake_read_rgb_image(path):
        return np.zeros(
            (10, 10, 3),
            dtype=np.uint8,
        )

    def fake_compute_exg(image):
        return np.zeros(
            (10, 10),
            dtype=float,
        )

    def fake_threshold_vegetation_index(
        exg,
        threshold,
    ):
        return np.zeros(
            (10, 10),
            dtype=bool,
        )

    def fake_estimate_row_orientation(
        vegetation_mask,
        angles,
    ):
        return 17.0

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.read_rgb_image",
        fake_read_rgb_image,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.compute_exg",
        fake_compute_exg,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.threshold_vegetation_index",
        fake_threshold_vegetation_index,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.estimate_row_orientation",
        fake_estimate_row_orientation,
    )

    result = compute_rotation_angle(
        image_path=Path("image.png"),
        threshold=25,
        angle=None,
    )

    assert result == 17.0

def test_detect_plants_in_row(monkeypatch):
    row_mask = np.ones(
        (10, 20),
        dtype=np.uint8,
    )

    plant_positions = np.array([
        5,
        15,
    ])

    search_windows = [
        (2, 8),
        (12, 18),
    ]

    vegetation_pixel_counts = [
        30,
        25,
    ]

    vegetation_fractions = [
        0.50,
        0.03,
    ]

    vegetation_centroids = [
        (5.0, 102.0),
        (15.0, 105.0),
    ]

    def fake_define_plant_search_windows(
        plant_positions,
        image_width,
    ):
        assert image_width == 20

        np.testing.assert_array_equal(
            plant_positions,
            np.array([5, 15]),
        )

        return search_windows

    def fake_count_vegetation_pixels(
        row_mask,
        search_windows,
    ):
        return vegetation_pixel_counts

    def fake_compute_vegetation_fraction(
        row_mask,
        search_windows,
    ):
        return vegetation_fractions

    def fake_compute_vegetation_centroids(
        row_mask,
        search_windows,
        row_y_start,
    ):
        assert row_y_start == 100

        return vegetation_centroids

    def fake_build_plant_dataframe(
        plant_positions,
        vegetation_pixel_counts,
        vegetation_centroids,
    ):
        return pd.DataFrame({
            "plant_position": plant_positions,
            "row_profile_pixels": vegetation_pixel_counts,
            "centroid_x": [
                centroid[0]
                for centroid in vegetation_centroids
            ],
            "centroid_y": [
                centroid[1]
                for centroid in vegetation_centroids
            ],
        })

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.define_plant_search_windows",
        fake_define_plant_search_windows,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.count_vegetation_pixels",
        fake_count_vegetation_pixels,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.compute_vegetation_fraction",
        fake_compute_vegetation_fraction,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.compute_vegetation_centroids",
        fake_compute_vegetation_centroids,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.build_plant_dataframe",
        fake_build_plant_dataframe,
    )

    result = detect_plants_in_row(
        row_mask=row_mask,
        row_y_start=100,
        plant_positions=plant_positions,
        row_number=3,
    )

    assert len(result) == 2

    assert list(result["x_start"]) == [2, 12]

    assert list(result["x_end"]) == [8, 18]

    assert list(result["vegetation_fraction"]) == [
        0.50,
        0.03,
    ]

    assert list(result["row"]) == [
        3,
        3,
    ]

    assert list(result["missing_candidate"]) == [
        False,
        True,
    ]

def test_prepare_rotated_data(toy_dataset):
    rotated_img, rotated_exg, rotated_plot_mask = (
        prepare_rotated_data(
            image_path=toy_dataset["image_path"],
            geotiff_path=toy_dataset["geotiff_path"],
            geometry=toy_dataset["geometry"],
            angle=0,
        )
    )

    assert rotated_img.ndim == 3
    assert rotated_exg.ndim == 2
    assert rotated_plot_mask.ndim == 2
    assert rotated_plot_mask.dtype == np.uint8

    assert rotated_img.shape[:2] == rotated_exg.shape
    assert rotated_exg.shape == rotated_plot_mask.shape

def test_prepare_rotated_data_with_rotation(toy_dataset):
    rotated_img, rotated_exg, rotated_plot_mask = (
        prepare_rotated_data(
            image_path=toy_dataset["image_path"],
            geotiff_path=toy_dataset["geotiff_path"],
            geometry=toy_dataset["geometry"],
            angle=45,
        )
    )

    assert rotated_img.ndim == 3
    assert rotated_exg.ndim == 2
    assert rotated_plot_mask.ndim == 2

    assert rotated_img.shape[:2] == rotated_exg.shape
    assert rotated_exg.shape == rotated_plot_mask.shape

    assert rotated_plot_mask.dtype == np.uint8

def test_detect_plants_in_row_two_plants():
    row_mask = np.zeros(
        (10, 30),
        dtype=np.uint8,
    )

    # Vegetation around expected plant 1
    row_mask[2:5, 3:6] = 1

    # Vegetation around expected plant 2
    row_mask[4:8, 13:16] = 1

    plant_positions = np.array([5, 15])

    result = detect_plants_in_row(
        row_mask=row_mask,
        row_y_start=100,
        plant_positions=plant_positions,
        row_number=2,
    )

    assert isinstance(result, pd.DataFrame)

    # One row per expected plant
    assert len(result) == 2

    # Expected positions
    assert list(result["plant_position"]) == [1, 2]
    assert list(result["expected_x"]) == [5, 15]

    # Search windows
    assert list(result["x_start"]) == [0, 10]
    assert list(result["x_end"]) == [11, 21]

    # Vegetation counts
    assert list(result["row_profile_pixels"]) == [9, 12]

    # Row number
    assert list(result["row"]) == [2, 2]

    # Missing candidates
    assert list(result["missing_candidate"]) == [
        False,
        False,
    ]

def test_detect_plants(toy_dataset, monkeypatch):
    plants_per_row = 3
    n_rows = 2

    plant_df_row = pd.DataFrame({
        "plant_position": [1, 2, 3],
        "missing_candidate": [False, False, False],
        "centroid_x": [10.0, 20.0, 30.0],
        "centroid_y": [100.0, 100.0, 100.0],
    })

    def fake_load_plot_metadata(geopackage_path):
        return (
            "103_DSC01167.png",
            plants_per_row,
            n_rows,
            toy_dataset["geometry"],
        )

    def fake_compute_rotation_angle(
        image_path,
        angle,
        threshold,
    ):
        return 15.0

    def fake_prepare_rotated_data(
        image_path,
        geotiff_path,
        geometry,
        angle,
    ):
        rotated_img = np.zeros(
            (100, 200, 3),
            dtype=np.uint8,
        )

        rotated_exg = np.zeros(
            (100, 200),
            dtype=float,
        )

        rotated_plot_mask = np.ones(
            (100, 200),
            dtype=np.uint8,
        )

        return (
            rotated_img,
            rotated_exg,
            rotated_plot_mask,
        )

    def fake_get_mask_bounds(mask):
        return 0, 199, 0, 99

    def fake_estimate_plant_positions_from_plot(
        x_start,
        x_end,
        n_plants,
    ):
        return np.array([25, 100, 175])

    def fake_compute_row_boundaries_from_plot(
        y_start,
        y_end,
        n_rows,
    ):
        return np.array([0, 50, 100])

    def fake_extract_row_images(
        image,
        boundaries,
    ):
        return [
            np.zeros((50, 200, 3), dtype=np.uint8),
            np.zeros((50, 200, 3), dtype=np.uint8),
        ]

    def fake_segment_row_images(
        row_images,
        threshold,
    ):
        return [
            np.zeros((50, 200), dtype=np.uint8),
            np.zeros((50, 200), dtype=np.uint8),
        ]

    def fake_detect_plants_in_row(
        row_mask,
        row_y_start,
        plant_positions,
        row_number,
        vegetation_fraction_threshold,
    ):
        result = plant_df_row.copy()
        result["row"] = row_number
        return result

    def fake_compute_row_profile(vegetation_mask):
        return np.zeros(100)

    def fake_detect_crop_rows(row_profile):
        return np.array([20, 70])

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.load_plot_metadata",
        fake_load_plot_metadata,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.compute_rotation_angle",
        fake_compute_rotation_angle,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.prepare_rotated_data",
        fake_prepare_rotated_data,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.get_mask_bounds",
        fake_get_mask_bounds,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.estimate_plant_positions_from_plot",
        fake_estimate_plant_positions_from_plot,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.compute_row_boundaries_from_plot",
        fake_compute_row_boundaries_from_plot,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.extract_row_images",
        fake_extract_row_images,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.segment_row_images",
        fake_segment_row_images,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.detect_plants_in_row",
        fake_detect_plants_in_row,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.compute_row_profile",
        fake_compute_row_profile,
    )

    monkeypatch.setattr(
        "opencropphenotyping.segmentation.detect_crop_rows",
        fake_detect_crop_rows,
    )

    result = detect_plants(
        image_path=toy_dataset["image_path"],
        geotiff_path=toy_dataset["geotiff_path"],
        geopackage_path=toy_dataset["geopackage_path"],
        best_angle=None,
        export_all=False,
        threshold=25,
    )

    assert result[:-1] == (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    plants_df = result[-1]

    assert isinstance(plants_df, pd.DataFrame)

    assert len(plants_df) == plants_per_row * n_rows

    assert list(plants_df["row"]) == [
        1, 1, 1,
        2, 2, 2,
    ]

    assert list(plants_df["plant_position"]) == [
        1, 2, 3,
        1, 2, 3,
    ]