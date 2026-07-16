from pathlib import Path
import numpy as np
import rasterio

def read_band(filepath: Path) -> tuple[np.ndarray, dict]:
    """
    Read a single band from a raster file.

    Parameters
    ----------
    filepath : Path
        Path to the raster file.

    Returns
    -------
    image : numpy.ndarray
        Raster values.

    profile : dict
        Raster metadata.
    """
    with rasterio.open(filepath) as src:
        image = src.read(1)
        profile = src.profile
    return image, profile