import numpy as np


def _prepare_bands(band1: np.ndarray, band2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare the input bands for index computation.

    Parameters
    ----------
    band1 : np.ndarray
        First band.
    band2 : np.ndarray
        Second band.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Prepared bands as float32 arrays.
    """
    # Convert inputs to float32 arrays
    band1 = np.asarray(band1, dtype=np.float32)
    band2 = np.asarray(band2, dtype=np.float32)

    # Ensure that the input bands have the same shape
    if band1.shape != band2.shape:
        raise ValueError("Input bands must have the same shape.")

    return band1, band2


def compute_normalized_difference_index(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Index (NDI) from the given two bands.

    Parameters
    ----------
    band1 : np.ndarray
        First band.

    band2 : np.ndarray
        Second band.

    Returns
    -------
    np.ndarray
        NDI image.
    """
    band1, band2 = _prepare_bands(band1, band2)

    # Compute NDI using the formula: (band1 - band2) / (band1 + band2)
    denominator = band1 + band2
    numerator = band1 - band2
    ndi = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)

    return ndi.astype(np.float32)


def compute_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Vegetation Index (NDVI) from the given red and near-infrared (NIR) bands.

    Parameters
    ----------
    red_band : np.ndarray
        Red band.

    nir_band : np.ndarray
        Near infrared band.

    Returns
    -------
    np.ndarray
        NDVI image.
    """
    return compute_normalized_difference_index(nir_band, red_band)


def compute_ndre(nir_band: np.ndarray, red_edge_band: np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Red Edge Index (NDRE) from the given red and red edge bands.

    Parameters
    ----------
    nir_band : np.ndarray
        Near infrared band.

    red_edge_band : np.ndarray
        Red edge band.

    Returns
    -------
    np.ndarray
        NDRE image.
    """
    return compute_normalized_difference_index(nir_band, red_edge_band)

def compute_gndvi(nir_band: np.ndarray, green_band: np.ndarray) -> np.ndarray:
    """
    Compute the Green Normalized Difference Vegetation Index (GNDVI) from the given nir and green bands.

    Parameters
    ----------
    nir_band : np.ndarray
        Red band.

    green_band : np.ndarray
        Red edge band.

    Returns
    -------
    np.ndarray
        GNDVI image.
    """
    return compute_normalized_difference_index(nir_band, green_band)

def compute_savi(red_band: np.ndarray, nir_band: np.ndarray, L_factor : float = 0.5) -> np.ndarray:
    """
    Compute the Soil Adjusted Vegetation Index (SAVI) from the given red and near-infrared (NIR) bands.
    The L factor is typically set to 0.5 for moderate vegetation cover.

    Parameters
    ----------
    red_band : np.ndarray
        Red band.

    nir_band : np.ndarray
        Near infrared band.

    Returns
    -------
    np.ndarray
        SAVI image.
    """
    # Raise an error if L < 0
    if L_factor < 0:
        raise ValueError("L_factor must be greater than or equal to 0.")

    prepared_red_band, prepared_nir_band = _prepare_bands(red_band, nir_band)

    # Compute SAVI index
    denominator = prepared_nir_band + prepared_red_band + L_factor
    numerator = (prepared_nir_band - prepared_red_band) * (1 + L_factor)
    savi = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)

    return savi.astype(np.float32)

VEGETATION_INDICES = {
    "ndvi": {
        "function": compute_ndvi,
        "bands": ["red_band", "nir_band"],
    },
    "savi": {
        "function": compute_savi,
        "bands": ["red_band", "nir_band"],
    },
    "ndre": {
        "function": compute_ndre,
        "bands": ["nir_band", "red_edge_band"],
    },
    "gndvi": {
        "function": compute_gndvi,
        "bands": ["nir_band", "green_band"],
    },
}

def compute_indexes(
        red_band: np.ndarray | None = None,
        nir_band: np.ndarray | None = None,
        green_band: np.ndarray | None = None,
        red_edge_band: np.ndarray | None = None,
        indices: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
    """Compute vegetation indices from available spectral bands.

    Parameters
    ----------
    red_band : np.ndarray or None, optional
        Red spectral band.
    nir_band : np.ndarray or None, optional
        Near-infrared (NIR) spectral band.
    green_band : np.ndarray or None, optional
        Green spectral band.
    red_edge_band : np.ndarray or None, optional
        Red-edge spectral band.
    indices : list[str] or None, optional
        Names of the vegetation indices to compute. If None, all
        available vegetation indices are requested.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping each successfully computed vegetation index
        to its resulting array. An index is skipped when one or more
        of its required spectral bands are unavailable.

    Raises
    ------
    ValueError
        If an unknown vegetation index is requested.
    """
    results = {}

    if indices is None:
        indices = ["ndvi", "savi", "ndre", "gndvi"]

    available_bands = {
        "red_band": red_band,
        "nir_band": nir_band,
        "green_band": green_band,
        "red_edge_band": red_edge_band,
    }

    for index in indices:
        if index not in VEGETATION_INDICES:
            raise ValueError(f"Unknown vegetation index: {index}")
        
        index_config = VEGETATION_INDICES[index]
        required_bands = index_config["bands"]

        if all(
            available_bands[band] is not None
            for band in required_bands
        ):
            compute_function = index_config["function"]
            band_arrays = [
                available_bands[band]
                for band in required_bands
            ]

            results[index] = compute_function(*band_arrays)
            
    return results
    
def compute_exg(rgb: np.ndarray) -> np.ndarray:
    """Compute the Excess Green Index (ExG).

    Parameters
    ----------
    rgb
        RGB image with shape (height, width, 3).

    Returns
    -------
    np.ndarray
        ExG image.
    """
    rgb = rgb.astype(np.float32)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    return 2 * green - red - blue