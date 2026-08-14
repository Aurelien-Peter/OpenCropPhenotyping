import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from scipy.signal import find_peaks, savgol_filter

from opencropphenotyping.indices import compute_exg

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

def compute_row_boundaries(
    row_positions: np.ndarray,
    image_height: int,
) -> np.ndarray:
    """
    Compute search-band boundaries from detected crop-row positions.

    Parameters
    ----------
    row_positions : np.ndarray
        Vertical positions of detected crop rows in pixels.
    image_height : int
        Height of the image in pixels.

    Returns
    -------
    np.ndarray
        Vertical boundaries separating crop-row search bands.
    """
    if len(row_positions) == 0:
        return np.array([])

    if len(row_positions) == 1:
        return np.array([0, image_height])

    spacings = np.diff(row_positions)
    mean_spacing = spacings.mean()

    internal_boundaries = (
        row_positions[:-1] + row_positions[1:]
    ) / 2

    top_boundary = row_positions[0] - mean_spacing / 2
    bottom_boundary = row_positions[-1] + mean_spacing / 2

    boundaries = np.concatenate(
        (
            [top_boundary],
            internal_boundaries,
            [bottom_boundary],
        )
    )

    boundaries[0] = max(0, boundaries[0])
    boundaries[-1] = min(image_height, boundaries[-1])

    return boundaries

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

def estimate_plant_positions(
    row_mask: np.ndarray,
    n_plants: int,
    profile_window_length: int = 31,
    profile_polyorder: int = 2,
    row_profile_threshold: float = 20,
) -> np.ndarray:
    """
    Estimate theoretical plant positions along a crop row.

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
    plant_positions: np.ndarray,
    search_windows: list[tuple[int, int]],
) -> list[tuple[float, float] | None]:
    """
    Compute vegetation centroids within plant search windows.

    Parameters
    ----------
    row_mask : np.ndarray
        Binary vegetation mask for one crop-row region.
    plant_positions : np.ndarray
        Theoretical plant positions along the crop row.
        Used to associate each centroid with its expected position.
    search_windows : list[tuple[int, int]]
        Search windows represented as ``(x_start, x_end)`` pixel
        coordinates. The end coordinate is exclusive.

    Returns
    -------
    list[tuple[float, float] | None]
        Vegetation centroid ``(x, y)`` for each search window.
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

        centroid_x = float(x_coords_global.mean())
        centroid_y = float(y_coords.mean())

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

def identify_missing_plant_candidates(
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