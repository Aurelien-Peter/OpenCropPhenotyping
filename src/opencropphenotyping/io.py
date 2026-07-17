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

def find_granule(safe_path: Path) -> Path:
    """
    Find the granule directory inside a Sentinel-2 SAFE product.
    """
    return next((safe_path / "GRANULE").iterdir())

def find_band(safe_path: Path, band: str) -> Path:
    """
    Find the path to a specific band in a Sentinel-2 SAFE directory.

    Parameters
    ----------
    safe_path : Path
        Path to the Sentinel-2 SAFE directory.
    band : str
        Band identifier (e.g., 'B04', 'B08').

    Returns
    -------
    Path
        Path to the band file.
    """
    granule_dir = find_granule(safe_path)
    r10m_dir = granule_dir / "IMG_DATA" / "R10m"
    band_file = next(r10m_dir.glob(f"*_{band}_10m.jp2"))
    return band_file