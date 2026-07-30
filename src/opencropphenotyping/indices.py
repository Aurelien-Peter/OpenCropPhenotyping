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

def compute_indexes(
        red_band: np.ndarray | None = None,
        nir_band: np.ndarray | None = None,
        green_band: np.ndarray | None = None,
        red_edge_band: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
    results = {}
    if(nir_band is not None):
        if(red_band is not None):
            results["ndvi"] = compute_ndvi(red_band, nir_band)
            results["savi"] = compute_savi(red_band, nir_band)
        if(red_edge_band is not None):
            results["ndre"] = compute_ndre(nir_band, red_edge_band)
        if(green_band is not None):
            results["gndvi"] = compute_gndvi(nir_band, green_band)
    return results
    
    