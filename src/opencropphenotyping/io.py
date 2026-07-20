from pathlib import Path
import profile
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

def write_raster(image: np.ndarray, 
                 profile: dict, 
                 output_path: Path) -> None:
    """
    Write a raster image to a file.

    Parameters
    ----------
    image : numpy.ndarray
        Raster values to write.
    profile : dict
        Raster metadata.
    output_path : Path
        Path to the output raster file.
    """

    # Update the profile for the output raster, as its dtype and count may differ from the input
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        driver="GTiff",
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(image, 1) # Select first band for writing