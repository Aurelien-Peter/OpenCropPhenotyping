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
    ndi = np.divide(numerator, 
                    denominator, 
                    out=np.zeros_like(denominator), 
                    where=denominator!=0)

    return ndi.astype(np.float32)

def compute_ndvi(red_band : np.ndarray, nir_band : np.ndarray) -> np.ndarray:
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

def compute_ndre(nir_band : np.ndarray, red_edge_band : np.ndarray) -> np.ndarray:
    """
    Compute the Normalized Difference Red Edge Index (NDRE) from the given red and red edge bands.

    Parameters
    ----------
    red_band : np.ndarray
        Red band.

    red_edge_band : np.ndarray
        Red edge band.

    Returns
    -------
    np.ndarray
        NDRE image.
    """    
    return compute_normalized_difference_index(nir_band, red_edge_band)