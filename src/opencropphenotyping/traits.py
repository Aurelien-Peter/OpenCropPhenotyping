import numpy as np


def create_vegetation_mask(ndvi_image: np.ndarray, 
                           threshold: float = 0.3
                           ) -> np.ndarray:
    """
    Create a binary vegetation mask based on NDVI values.

    Parameters
    ----------
    ndvi_image : np.ndarray
        NDVI image.
    threshold : float, optional
        NDVI threshold above which pixels are classified as vegetation.
        Default is 0.3.
    Returns
    -------
    np.ndarray
        Binary mask where vegetation pixels are marked as 1 and non-vegetation as 0.
        Nan indicates invalid pixels.
    """
    # Create a binary mask where NDVI values above the threshold are considered vegetation
    vegetation_mask = np.where(
        np.isnan(ndvi_image),
        np.nan,
        np.where(ndvi_image > threshold, 1, 0)
    )
    
    return vegetation_mask

def compute_crop_cover(vegetation_mask: np.ndarray) -> float:
    """
    Compute the crop cover percentage based on the vegetation mask.

    Parameters
    ----------
    vegetation_mask : np.ndarray
        Binary mask where vegetation pixels are marked as 1 and non-vegetation as 0.
        Nan indicates invalid pixels.

    Returns
    -------
    float
        Crop cover percentage.
    """
    # Count valid pixels, excluding NaN values
    valid_pixels = ~np.isnan(vegetation_mask)
    total__valid_pixels = np.sum(valid_pixels)

    # Raise an error if the vegetation mask contains only NaN values
    if total__valid_pixels == 0:
        raise ValueError(
            "The vegetation mask contains only NaN values. "
            "Cannot compute crop cover."
        )

    vegetation_pixels = np.nansum(vegetation_mask)
    crop_cover = (vegetation_pixels / total__valid_pixels) * 100
    return crop_cover
