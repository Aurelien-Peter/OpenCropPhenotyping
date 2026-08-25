import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import rotate
from scipy.signal import find_peaks, savgol_filter
from shapely.geometry.base import BaseGeometry

from opencropphenotyping.indices import compute_exg
from opencropphenotyping.io import read_rgb_image

def threshold_vegetation_index(
    exg: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Create a vegetation mask from an ExG image."""
    return ~np.isnan(exg) & (exg > threshold)

def compute_row_profile(
    vegetation_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute the vegetation profile along image rows.

    Parameters
    ----------
    vegetation_mask : np.ndarray
        Binary vegetation mask.

    Returns
    -------
    np.ndarray
        Number of vegetation pixels for each image row.
    """
    return vegetation_mask.sum(axis=1)

def estimate_row_orientation(
    vegetation_mask: np.ndarray,
    angles: np.ndarray | None = None,
) -> float:
    """
    Estimate the orientation of crop rows from a vegetation mask.

    Parameters
    ----------
    vegetation_mask : np.ndarray
        Binary vegetation mask.
    angles : np.ndarray, optional
        Candidate rotation angles to evaluate.

    Returns
    -------
    float
        Estimated crop-row orientation in degrees.
    """
    # Refine the decision by rotating the image at different angle
    # and find out the best angle
    # Compute the profile ExG for each rotation angle with a given threshold
    profiles = {}
    
    if angles is not None:
        angle_range = angles
    else: 
        angle_range = np.arange(-90, 90, 1)

    for angle in angle_range:

        rotated = rotate(
            vegetation_mask,
            angle=angle,
            reshape=True,
            order=0,
        )

        profile = compute_row_profile(rotated)
        profiles[angle] = profile

    scores = {}

    for angle, profile in profiles.items():

        mean_profile = profile.mean()

        if mean_profile == 0:
            score = 0.0
        else:
            score = profile.std()

        scores[angle] = score

    best_angle = max(
        scores,
        key=lambda angle: scores[angle],
    )

    return float(best_angle)

def detect_crop_rows(
    row_profile: np.ndarray,
    window_length: int = 51,
    polyorder: int = 2,
    distance: int = 100,
    prominence: float = 10,
) -> np.ndarray:
    """
    Detect crop-row positions from a vegetation profile.

    Parameters
    ----------
    row_profile : np.ndarray
        One-dimensional vegetation profile perpendicular to crop rows.
    window_length : int
        Window length for Savitzky-Golay smoothing.
    polyorder : int
        Polynomial order for Savitzky-Golay smoothing.
    distance : int
        Minimum distance between detected rows in pixels.
    prominence : float
        Minimum peak prominence.

    Returns
    -------
    np.ndarray
        Detected crop-row positions in pixels.
    """
    smoothed_profile = savgol_filter(
        row_profile,
        window_length=window_length,
        polyorder=polyorder,
    )

    peaks, _ = find_peaks(
        smoothed_profile,
        distance=distance,
        prominence=prominence,
    )

    return peaks

def compute_row_boundaries_from_peaks(
    row_positions: np.ndarray,
    image_height: int,
) -> np.ndarray:
    """
    Compute crop-row boundaries from detected row positions.

    Boundaries between neighbouring rows are placed halfway between
    consecutive detected row positions. The outer boundaries are
    estimated using the mean distance between consecutive rows.

    Parameters
    ----------
    row_positions : np.ndarray
        Vertical positions of detected crop rows.
    image_height : int
        Height of the image in pixels.

    Returns
    -------
    np.ndarray
        Vertical boundaries separating crop-row regions.
    """
    if len(row_positions) == 0:
        return np.array([])

    if len(row_positions) == 1:
        return np.array([0, image_height])

    row_spacings = np.diff(row_positions)
    mean_spacing = row_spacings.mean()

    internal_boundaries = (
        row_positions[:-1] + row_positions[1:]
    ) / 2

    top_boundary = max(
        0,
        row_positions[0] - mean_spacing / 2,
    )

    bottom_boundary = min(
        image_height,
        row_positions[-1] + mean_spacing / 2,
    )

    return np.concatenate(
        [
            [top_boundary],
            internal_boundaries,
            [bottom_boundary],
        ]
    )

def compute_row_boundaries_from_plot(
    y_start: int,
    y_end: int,
    n_rows: int,
) -> np.ndarray:
    """
    Compute crop-row boundaries from known plot extent and row count.

    The vertical extent of the experimental plot is divided into
    ``n_rows`` equal regions.

    Parameters
    ----------
    y_start : int
        Starting y-coordinate of the plot in the rotated image.
    y_end : int
        Ending y-coordinate of the plot in the rotated image.
    n_rows : int
        Expected number of crop rows.

    Returns
    -------
    np.ndarray
        Vertical boundaries separating crop-row regions.

    Raises
    ------
    ValueError
        If ``n_rows`` is less than 1.
    """
    if n_rows < 1:
        raise ValueError(
            "The number of crop rows must be greater than or equal to 1."
        )

    boundaries = np.linspace(
        y_start,
        y_end,
        n_rows + 1,
    )

    return np.round(boundaries).astype(int)

def extract_row_images(
    image: np.ndarray,
    boundaries: np.ndarray,
) -> list[np.ndarray]:
    """
    Extract one image region for each detected crop row.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape (height, width, channels).
    boundaries : np.ndarray
        Row-band boundaries along the vertical image axis.

    Returns
    -------
    list[np.ndarray]
        Image region corresponding to each crop row.
    """
    row_images = []

    for i in range(len(boundaries) - 1):
        y_start = int(boundaries[i])
        y_end = int(boundaries[i + 1])

        row_images.append(
            image[y_start:y_end, :, :]
        )

    return row_images

def segment_row_images(
    row_images: list[np.ndarray],
    threshold: float,
) -> list[np.ndarray]:
    """
    Segment vegetation within crop-row image regions.

    Parameters
    ----------
    row_images : list[np.ndarray]
        RGB image regions corresponding to detected crop rows.
        Each image must have shape (height, width, 3).

    threshold : float
        ExG threshold used to separate vegetation from
        soil and background.

    Returns
    -------
    list[np.ndarray]
        Binary vegetation mask for each crop-row region.
    """
    row_masks = []

    for row_image in row_images:
        rgb = (
            row_image[:, :, 0],
            row_image[:, :, 1],
            row_image[:, :, 2],
        )

        exg = compute_exg(rgb)

        mask = threshold_vegetation_index(
            exg,
            threshold=threshold,
        )

        row_masks.append(mask)

    return row_masks

def estimate_plant_positions_from_vegetation(
    row_mask: np.ndarray,
    n_plants: int,
    profile_window_length: int = 31,
    profile_polyorder: int = 2,
    row_profile_threshold: float = 20,
) -> np.ndarray:
    """
    Estimate theoretical plant positions along a crop row from the vegetation signal.

    The binary vegetation mask is projected along the crop-row direction
    to obtain a one-dimensional vegetation profile. The profile is smoothed
    using a Savitzky-Golay filter and thresholded to estimate the effective
    vegetated extent of the row.

    The detected extent is divided into ``n_plants`` equal planting
    intervals. The centre of each interval is used as a theoretical
    expected plant position.

    Parameters
    ----------
    row_mask : np.ndarray
        Binary vegetation mask for one crop-row region. The first dimension
        corresponds to the row width and the second dimension to the
        position along the crop row.
    n_plants : int
        Expected number of plants along the crop row. Must be greater than
        or equal to 2.
    profile_window_length : int, default=31
        Window length used for Savitzky-Golay smoothing of the vegetation
        profile.
    profile_polyorder : int, default=2
        Polynomial order used for Savitzky-Golay smoothing.
    row_profile_threshold : float, default=20
        Minimum vegetation-pixel count required for a position along the
        row to be considered vegetation-supporting.

    Returns
    -------
    np.ndarray
        Theoretical plant positions along the crop row, expressed as
        integer pixel coordinates.

    Raises
    ------
    ValueError
        If ``n_plants`` is less than 2.

    Notes
    -----
    The returned positions are theoretical reference positions. They are
    not assumed to correspond to the actual centres of individual plants.
    """
    if n_plants < 2:
        raise ValueError(
            "The number of plants per row must be greater than or equal to 2."
        )

    # Project the 2D vegetation mask onto the crop-row direction.
    row_profile = row_mask.sum(axis=0)

    # Smooth the profile to reduce local variations in the vegetation signal.
    smoothed_profile = savgol_filter(
        row_profile,
        window_length=profile_window_length,
        polyorder=profile_polyorder,
    )

    # Identify the part of the row with a sufficiently strong vegetation signal.
    profile_pixels = np.where(
        smoothed_profile > row_profile_threshold
    )[0]

    # No vegetation-supporting region means that theoretical positions
    # cannot be estimated.
    if len(profile_pixels) == 0:
        return np.array([], dtype=int)

    # Estimate the effective start and end of the cultivated row.
    x_start = profile_pixels[0]
    x_end = profile_pixels[-1]

    # Divide the effective row extent into equal planting intervals.
    segment_edges = np.linspace(
        x_start,
        x_end,
        n_plants + 1,
    )

    # Use the centre of each interval as the theoretical plant position.
    plant_positions = (
        segment_edges[:-1] + segment_edges[1:]
    ) / 2

    return np.round(plant_positions).astype(int)

def estimate_plant_positions_from_plot(
    x_start: int,
    x_end: int,
    n_plants: int,
) -> np.ndarray:
    """
    Estimate theoretical plant positions from known crop-row boundaries.

    The crop-row extent is defined externally from the experimental
    plot geometry. This extent is divided into ``n_plants`` equal
    planting intervals, and the centre of each interval is used as the
    expected plant position.

    Parameters
    ----------
    x_start : int
        Starting x-coordinate of the crop-row extent.
    x_end : int
        Ending x-coordinate of the crop-row extent.
    n_plants : int
        Expected number of plants along the crop row. Must be greater
        than or equal to 2.

    Returns
    -------
    np.ndarray
        Theoretical plant positions expressed as integer x-coordinates.

    Raises
    ------
    ValueError
        If ``n_plants`` is less than 2.
    """
    if n_plants < 2:
        raise ValueError(
            "The number of plants per row must be greater than or equal to 2."
        )

    segment_edges = np.linspace(
        x_start,
        x_end,
        n_plants + 1,
    )

    plant_positions = (
        segment_edges[:-1] + segment_edges[1:]
    ) / 2

    return np.round(plant_positions).astype(int)

def define_plant_search_windows(
    plant_positions: np.ndarray,
    image_width: int,
) -> list[tuple[int, int]]:
    """
    Define search windows around theoretical plant positions.

    The mean spacing between consecutive theoretical plant positions is
    used to estimate a symmetric search window around each position.
    Window boundaries are clipped to the image limits.

    Parameters
    ----------
    plant_positions : np.ndarray
        Theoretical plant positions along the crop row, expressed as
        pixel coordinates.
    image_width : int
        Width of the crop-row image in pixels.

    Returns
    -------
    list[tuple[int, int]]
        Search windows represented as ``(x_start, x_end)`` pixel positions.
        The end coordinate follows the NumPy slicing convention and is
        therefore exclusive.

    Raises
    ------
    ValueError
        If fewer than two plant positions are provided.
    """
    if len(plant_positions) < 2:
        raise ValueError(
            "At least two plant positions are required to define "
            "the expected spacing."
        )

    # Estimate the expected spacing between neighbouring plants.
    expected_spacing = np.diff(plant_positions).mean()

    # Allow the actual plant centre to deviate from the theoretical
    # position by up to half the expected spacing.
    window_half_width = int(expected_spacing / 2)

    search_windows = []

    for position in plant_positions:
        # Keep each search window inside the image boundaries.
        x_start = max(
            0,
            position - window_half_width,
        )

        x_end = min(
            image_width,
            position + window_half_width + 1,
        )

        search_windows.append(
            (x_start, x_end)
        )

    return search_windows

def count_vegetation_pixels(
    row_mask: np.ndarray,
    search_windows: list[tuple[int, int]],
) -> list[int]:
    """
    Count vegetation pixels within each plant search window.

    Parameters
    ----------
    row_mask : np.ndarray
        Binary vegetation mask for one crop-row region.
    search_windows : list[tuple[int, int]]
        Search windows represented as ``(x_start, x_end)`` pixel
        coordinates. The end coordinate is exclusive.

    Returns
    -------
    list[int]
        Number of vegetation pixels contained in each search window.
    """
    vegetation_pixel_counts = []

    for start, end in search_windows:
        window_mask = row_mask[:, start:end]

        vegetation_pixel_counts.append(
            int(np.sum(window_mask))
        )

    return vegetation_pixel_counts

def compute_vegetation_centroids(
    row_mask: np.ndarray,
    search_windows: list[tuple[int, int]],
    row_y_start: int,
) -> list[tuple[float, float] | None]:
    """
    Compute vegetation centroids within plant search windows.

    Parameters
    ----------
    row_mask : np.ndarray
        Binary vegetation mask for one crop-row region.
    search_windows : list[tuple[int, int]]
        Search windows represented as ``(x_start, x_end)`` pixel
        coordinates. The end coordinate is exclusive.
    row_y_start : int
        Vertical position of the crop-row region in the full rotated image.

    Returns
    -------
    list[tuple[float, float] | None]
        Vegetation centroid ``(x, y)`` for each search window, expressed
        in the coordinate system of the full rotated image.
        ``None`` is returned when no vegetation pixels are present
        in a window.
    """
    vegetation_centroids = []

    for start, end in search_windows:
        window_mask = row_mask[:, start:end]

        y_coords, x_coords = np.where(window_mask)

        if len(x_coords) == 0:
            vegetation_centroids.append(None)
            continue

        # Convert the local x coordinates back to coordinates
        # in the complete row image.
        x_coords_global = x_coords + start

        # Convert local y coordinates to coordinates in the full
        # rotated image.
        y_coords_global = y_coords + row_y_start

        centroid_x = float(x_coords_global.mean())
        centroid_y = float(y_coords_global.mean())

        vegetation_centroids.append(
            (centroid_x, centroid_y)
        )

    return vegetation_centroids

def build_plant_dataframe(
    plant_positions: np.ndarray,
    vegetation_pixel_counts: list[int],
    vegetation_centroids: list[tuple[float, float] | None],
) -> pd.DataFrame:
    """
    Build a table describing vegetation detected around each expected
    plant position.

    Parameters
    ----------
    plant_positions : np.ndarray
        Theoretical plant positions along the crop row.
    vegetation_pixel_counts : list[int]
        Number of vegetation pixels detected in each search window.
    vegetation_centroids : list[tuple[float, float] | None]
        Vegetation centroid for each search window.

    Returns
    -------
    pd.DataFrame
        Table containing the expected position, vegetation amount,
        vegetation centroid, and centroid offset from the expected
        position.
    """
    plant_data = []

    for i, (
        position,
        vegetation_count,
        centroid,
    ) in enumerate(
        zip(
            plant_positions,
            vegetation_pixel_counts,
            vegetation_centroids,
        ),
        start=1,
    ):
        if centroid is None:
            centroid_x = np.nan
            centroid_y = np.nan
            offset = np.nan
            abs_offset = np.nan

        else:
            centroid_x, centroid_y = centroid

            offset = centroid_x - position
            abs_offset = abs(offset)

        plant_data.append(
            {
                "plant_position": i,
                "expected_x": position,
                "row_profile_pixels": vegetation_count,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "offset": offset,
                "abs_offset": abs_offset,
            }
        )

    return pd.DataFrame(plant_data)

def identify_missing_plant_candidates_from_veg_threshold_and_centroid(
    plant_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify potential missing plants using vegetation amount and
    centroid displacement.

    Robust thresholds are estimated from the observed distributions
    using the interquartile range (IQR). A plant is flagged as a
    missing candidate when it simultaneously exhibits unusually low
    vegetation and an unusually large centroid displacement.

    Parameters
    ----------
    plant_df : pd.DataFrame
        Plant characterization table containing ``row_profile_pixels``
        and ``abs_offset`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input table with an additional
        ``missing_candidate`` boolean column.
    """
    plant_df = plant_df.copy()

    vegetation_q1 = plant_df["row_profile_pixels"].quantile(0.25)
    vegetation_q3 = plant_df["row_profile_pixels"].quantile(0.75)
    vegetation_iqr = vegetation_q3 - vegetation_q1

    offset_q1 = plant_df["abs_offset"].quantile(0.25)
    offset_q3 = plant_df["abs_offset"].quantile(0.75)
    offset_iqr = offset_q3 - offset_q1

    low_vegetation_threshold = (
        vegetation_q1 - 1.5 * vegetation_iqr
    )

    high_offset_threshold = (
        offset_q3 + 1.5 * offset_iqr
    )

    plant_df["missing_candidate"] = (
        (
            plant_df["row_profile_pixels"]
            < low_vegetation_threshold
        )
        &
        (
            plant_df["abs_offset"]
            > high_offset_threshold
        )
    )

    return plant_df

def compute_vegetation_fraction(
    row_mask: np.ndarray,
    search_windows: list[tuple[int, int]],
) -> list[float]:
    """
    Compute the fraction of vegetation pixels within each search window.

    Parameters
    ----------
    row_mask : np.ndarray
        Binary vegetation mask for one crop-row region.
    search_windows : list[tuple[int, int]]
        Search windows represented as ``(x_start, x_end)`` pixel
        coordinates. The end coordinate is exclusive.

    Returns
    -------
    list[float]
        Fraction of pixels classified as vegetation in each search window.
    
    Raises
    ------
    ValueError
        If a search window is empty.
    """
    vegetation_fractions = []

    for start, end in search_windows:
        window_mask = row_mask[:, start:end]

        if window_mask.size == 0:
            raise ValueError(
                f"Search window ({start}, {end}) is empty."
            )

        vegetation_fraction = (
            np.sum(window_mask) / window_mask.size
        )

        vegetation_fractions.append(
            vegetation_fraction
        )

    return vegetation_fractions

def identify_missing_plant_candidates_from_vegetation_fraction(
    plant_df: pd.DataFrame,
    vegetation_fraction_threshold: float = 0.04
) -> pd.DataFrame:
    """
    Identify potential missing plants from vegetation fraction.

    A plant is flagged as a missing candidate when the fraction of
    vegetation pixels within its search window is below the specified
    threshold.

    Parameters
    ----------
    plant_df : pd.DataFrame
        Plant characterization table containing a
        ``vegetation_fraction`` column.
    vegetation_fraction_threshold : float, default=0.04
        Minimum vegetation fraction required for a plant not to be
        considered a missing candidate.

    Returns
    -------
    pd.DataFrame
        Copy of the input table with an additional
        ``missing_candidate`` boolean column.

    Raises
    ------
    KeyError
        If ``vegetation_fraction`` is not present in ``plant_df``.
    """
    plant_df = plant_df.copy()

    plant_df["missing_candidate"] = (
        plant_df["vegetation_fraction"]
        < vegetation_fraction_threshold
    )

    return plant_df

def detect_plants(
        image_path: Path,
        geotiff_path: Path,
        geopackage_path: Path,
        best_angle: float|None = None,
        export_all: bool = False,
        threshold: float = 25,
        vegetation_fraction_threshold: float = 0.04,
):
    """
    Detect expected plants within crop rows of an RGB image.

    The detection workflow uses plot metadata from a GeoPackage,
    estimates or uses a predefined crop-row orientation, rotates
    the image and plot geometry, extracts crop rows, and analyses
    expected plant positions using vegetation-based measurements.

    Crop-row detection based on the vegetation profile is currently
    used for diagnostic purposes only. The detected peaks are not
    used to define the crop-row boundaries.

    Parameters
    ----------
    image_path : Path
        Path to the RGB image containing the crop plot.
    geotiff_path : Path
        Path to the GeoTIFF associated with the image and used to
        rasterize the plot geometry.
    geopackage_path : Path
        Path to the GeoPackage containing plot metadata and geometry.
    best_angle : float or None, default=None
        Rotation angle in degrees. If ``None``, the crop-row
        orientation is estimated automatically.
    export_all : bool, default=False
        If ``True``, return intermediate processing results in
        addition to the final plant DataFrame. If ``False``, only
        the final plant DataFrame is returned and the intermediate
        outputs are set to ``None``.
    threshold : float, default=25
        Threshold applied to the ExG vegetation index for vegetation
        segmentation.

    Returns
    -------
    tuple
        If ``export_all`` is ``True``, returns:

        ``(rotated_img, rotated_exg, vegetation_mask, row_profile,
        peaks, boundaries, row_images, row_masks, row_detections,
        plants_df)``.

        If ``export_all`` is ``False``, returns:

        ``(None, None, None, None, None, None, plants_df)``.

        ``plants_df`` contains the plant candidates detected across
        all crop rows.

    Notes
    -----
    The number of crop-row peaks detected from the vegetation
    profile is currently used as an informational diagnostic.
    It may be used in a future version as a warning mechanism or
    as an alternative method for defining crop-row boundaries.
    """
    # Get metadata from geopackage file
    image_name, n_plants, n_rows, geometry=load_plot_metadata(geopackage_path)
    print(f"\nProcessing: {image_name}")

    # Get rotation angle
    best_angle=compute_rotation_angle(
        image_path=image_path,
        angle=best_angle,
        threshold=threshold
    )

    # Get rotated image, exg and vegetation mask
    rotated_img, rotated_exg, rotated_plot_mask = prepare_rotated_data(
        image_path=image_path,
        geotiff_path=geotiff_path,
        geometry=geometry,
        angle=best_angle
    )

    # Get rotated plot coordinates
    x_start, x_end, y_start, y_end = get_mask_bounds(mask=rotated_plot_mask)

    # Compute plant positions from x limits and number of plants
    plant_positions = estimate_plant_positions_from_plot(
        x_start=x_start,
        x_end=x_end,
        n_plants=n_plants,
    )

    # Compute row boundaries
    boundaries = compute_row_boundaries_from_plot(
        y_start=y_start,
        y_end=y_end,
        n_rows=n_rows,
    )

    # Extract and segment crop rows
    row_images = extract_row_images(
        image=rotated_img,
        boundaries=boundaries,
    )

    row_masks = segment_row_images(
        row_images=row_images,
        threshold=threshold,
    )

    row_detections = []

    for i, row_mask in enumerate(row_masks):

        row_detection = detect_plants_in_row(
            row_mask=row_mask,
            row_y_start=int(boundaries[i]),
            plant_positions=plant_positions,
            row_number=i + 1,
            vegetation_fraction_threshold=vegetation_fraction_threshold,
        )

        row_detections.append(row_detection)

    # Combine all rows
    plants_df = pd.concat(
        row_detections,
        ignore_index=True,
    )

    # Keep detected plants
    detected_plants = plants_df[
        ~plants_df["missing_candidate"]
    ].dropna(
        subset=["centroid_x", "centroid_y"]
    )

        # Detect crop rows (informational only)
    vegetation_mask = threshold_vegetation_index(
        rotated_exg,
        threshold,
    )

    row_profile = compute_row_profile(
        vegetation_mask
    )

    peaks = detect_crop_rows(
        row_profile
    )
    
    print(
        f"Rows detected: {len(peaks)}"
        f" | Expected rows: {n_rows}"
        f" | Planting positions: {n_plants * n_rows}"
        f" | Plants detected: {len(detected_plants)}"
    )

    if(export_all):
        return rotated_img, rotated_exg, vegetation_mask, row_profile, peaks, boundaries, row_images, row_masks, row_detections, plants_df
    else:
        return None, None, None, None, None, None, plants_df

def define_plant_segments(
    x_start: int,
    x_end: int,
    n_segments: int,
) -> list[tuple[float, float]]:
    """
    Define equally spaced segments along a crop row.

    Parameters
    ----------
    x_start : int
        Starting x-coordinate of the row.
    x_end : int
        Ending x-coordinate of the row.
    n_segments : int
        Number of segments.

    Returns
    -------
    list[tuple[float, float]]
        Segment boundaries represented as ``(x_start, x_end)``.
        
    Raises
    ------
    ValueError
        If ``n_segments`` is less than 1 or if ``x_end`` is less
        than or equal to ``x_start``.
    """
    if n_segments < 1:
        raise ValueError(
            "The number of segments must be greater than or equal to 1."
        )

    if x_end <= x_start:
        raise ValueError(
            "x_end must be greater than x_start."
        )
    
    edges = np.linspace(
        x_start,
        x_end,
        n_segments + 1,
    )

    return [
        (start, end)
        for start, end in zip(
            edges[:-1],
            edges[1:],
        )
    ]

def transform_original_points_to_rotated(
    points: np.ndarray,
    angle: float,
    original_shape: tuple[int, int],
    rotated_shape: tuple[int, int],
) -> np.ndarray:
    """
    Transform points from the original image coordinates to the
    rotated image coordinates.

    The transformation accounts for the different image centers
    before and after rotation. Coordinates are expected in
    ``(x, y)`` order, while image shapes are given in
    ``(height, width)`` order.

    Parameters
    ----------
    points : np.ndarray
        Array of point coordinates with shape ``(n_points, 2)``.
        Each point is represented as ``(x, y)``.
    angle : float
        Rotation angle in degrees. The rotation convention follows
        the coordinate transformation used for the image rotation.
    original_shape : tuple[int, int]
        Shape of the original image represented as
        ``(height, width)``.
    rotated_shape : tuple[int, int]
        Shape of the rotated image represented as
        ``(height, width)``.

    Returns
    -------
    np.ndarray
        Transformed point coordinates with shape ``(n_points, 2)``.
        Coordinates are represented as ``(x, y)``.
    """
    original_height, original_width = original_shape
    rotated_height, rotated_width = rotated_shape

    original_center = np.array([
        (original_width - 1) / 2,
        (original_height - 1) / 2,
    ])

    rotated_center = np.array([
        (rotated_width - 1) / 2,
        (rotated_height - 1) / 2,
    ])

    shifted_points = points - original_center

    theta = np.deg2rad(angle)

    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])

    rotated_points = (
        shifted_points @ rotation_matrix.T
        + rotated_center
    )

    return rotated_points

def prepare_rotated_data(
    image_path: Path,
    geotiff_path: Path,
    geometry: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare image data and plot mask in the rotated coordinate system.

    The RGB image, ExG image, and plot mask are rotated by the
    specified angle. The plot geometry is first rasterized using
    the GeoTIFF spatial reference before being rotated.

    Parameters
    ----------
    image_path : Path
        Path to the RGB image.
    geotiff_path : Path
        Path to the GeoTIFF used to rasterize the plot geometry.
    geometry : shapely geometry
        Plot geometry to rasterize.
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Rotated RGB image, rotated ExG image, and rotated plot mask.
    """
       
    # Read image and compute ExG
    image_array = np.array(Image.open(image_path))

    exg = compute_exg(
        read_rgb_image(image_path)
    )
    
    # Rotate image and ExG
    rotated_img = rotate(
        image_array,
        angle=angle,
        reshape=True,
        order=1,
    )

    rotated_exg = rotate(
        exg,
        angle=angle,
        reshape=True,
        order=1,
    )

    # Rotate plot geometry
    with rasterio.open(geotiff_path) as src:
        plot_mask = rasterize(
            [(geometry, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype=np.uint8,
        )

    rotated_plot_mask = rotate(
        plot_mask,
        angle=angle,
        reshape=True,
        order=0,
    )
    return rotated_img, rotated_exg, rotated_plot_mask

def get_mask_bounds(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """
    Return the bounding coordinates of non-zero pixels.

    Parameters
    ----------
    mask : np.ndarray
        Binary or integer mask. Pixels greater than zero are
        considered part of the region of interest.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding coordinates represented as
        ``(x_start, x_end, y_start, y_end)``.
        The returned end coordinates correspond to the last
        non-zero pixel and are therefore inclusive.

    Raises
    ------
    ValueError
        If the mask does not contain any positive pixel.
    """
    y_pixels, x_pixels = np.where(mask > 0)

    if len(x_pixels) == 0:
        raise ValueError(
            "Mask does not contain any positive pixels."
        )

    return (
        int(x_pixels.min()),
        int(x_pixels.max()),
        int(y_pixels.min()),
        int(y_pixels.max()),
    )

def detect_plants_in_row(
    row_mask: np.ndarray,
    row_y_start: int,
    plant_positions: np.ndarray,
    row_number: int,
    vegetation_fraction_threshold: float = 0.04,
) -> pd.DataFrame:
    """
    Detect plant candidates within a single crop row.

    The function defines a search window around each expected
    plant position, computes vegetation statistics and centroids,
    builds the corresponding plant DataFrame, and identifies
    potential missing plants.

    Parameters
    ----------
    row_mask : np.ndarray
        Binary vegetation mask for the crop row.
    row_y_start : int
        Y-coordinate of the beginning of the crop row in the
        rotated image.
    plant_positions : np.ndarray
        Expected plant positions along the crop row.
    row_number : int
        Row identifier assigned to the detected plants.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per expected plant, with
        vegetation statistics, search-window coordinates,
        centroid coordinates, missing-plant status, and row number.
    """

    search_windows = define_plant_search_windows(
        plant_positions=plant_positions,
        image_width=row_mask.shape[1],
    )

    vegetation_pixel_counts = count_vegetation_pixels(
        row_mask=row_mask,
        search_windows=search_windows,
    )

    vegetation_fractions = compute_vegetation_fraction(
        row_mask=row_mask,
        search_windows=search_windows,
    )

    vegetation_centroids = compute_vegetation_centroids(
        row_mask=row_mask,
        search_windows=search_windows,
        row_y_start=row_y_start,
    )

    plant_df = build_plant_dataframe(
        plant_positions=plant_positions,
        vegetation_pixel_counts=vegetation_pixel_counts,
        vegetation_centroids=vegetation_centroids,
    )

    # Identify potential missing plants
    plant_df = identify_missing_plant_candidates_from_vegetation_fraction(
        plant_df,
        vegetation_fraction_threshold=vegetation_fraction_threshold,
    )

    plant_df["x_start"] = [
        start for start, _ in search_windows
    ]

    plant_df["x_end"] = [
        end for _, end in search_windows
    ]

    plant_df["vegetation_fraction"] = vegetation_fractions
    plant_df["row"] = row_number

    return plant_df

def load_plot_metadata(
        geopackage_path: Path
    )-> tuple[str, int, int, BaseGeometry]:
    """
    Load metadata for the first plot in a GeoPackage.

    Parameters
    ----------
    geopackage_path : Path
        Path to the GeoPackage containing plot metadata.

    Returns
    -------
    tuple[str, int, int, BaseGeometry]
        Tuple containing the image name, expected number of plants
        per row, expected number of rows, and plot geometry.
    """

    plots = gpd.read_file(
        geopackage_path,
    )

    plot = plots.iloc[0]

    image_name = plot["image_name"]
    n_plants = plot["n_plants"]
    n_rows = plot["n_rows"]
    geometry = plot["geometry"]
    return image_name, n_plants, n_rows, geometry

def compute_rotation_angle(
        image_path: Path,
        threshold: float,
        angle: float|None,
    )-> float:
    """
    Determine the crop-row rotation angle.

    If an angle is provided, it is returned directly. Otherwise,
    the function computes the Excess Green (ExG) vegetation index,
    thresholds it, and estimates the crop-row orientation.

    Parameters
    ----------
    image_path : Path
        Path to the RGB image.
    threshold : float
        Vegetation threshold used to create the binary mask.
    angle : float or None
        Predefined rotation angle. If ``None``, the angle is
        estimated automatically.

    Returns
    -------
    float
        Rotation angle in degrees.
    """
    
    # Estimate crop-row orientation
    if(angle is None):
        exg = compute_exg(
            read_rgb_image(image_path)
        )

        vegetation_mask = threshold_vegetation_index(
            exg,
            threshold=threshold,
        )

        angle = estimate_row_orientation(vegetation_mask=vegetation_mask, 
                                                angles=np.array(np.round(np.arange(-90, 90, 1),1)))
        
    return(angle)