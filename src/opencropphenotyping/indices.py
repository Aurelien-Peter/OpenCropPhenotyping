import numpy as np

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
    # Convert inputs to float32 arrays
    red_band = np.asarray(red_band, dtype=np.float32)
    nir_band = np.asarray(nir_band, dtype=np.float32)

    # Ensure that the input bands have the same shape
    if red_band.shape != nir_band.shape:
        raise ValueError("Red and NIR bands must have the same shape.")

    # Compute NDVI using the formula: (NIR - Red) / (NIR + Red)
    denominator = nir_band + red_band
    numerator = nir_band - red_band
    ndvi = np.divide(numerator, 
                     denominator, 
                     out=np.zeros_like(denominator), 
                     where=denominator!=0)

    return ndvi.astype(np.float32)