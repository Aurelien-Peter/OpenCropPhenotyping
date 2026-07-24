from pathlib import Path

import matplotlib.pyplot as plt
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
    ## Ensure the file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with rasterio.open(filepath) as src:
        image = src.read(1)
        profile = src.profile
    return image, profile


def find_granule(safe_path: Path) -> Path:
    """
    Find the granule directory inside a Sentinel-2 SAFE product.
    """
    ## Ensure the file exists
    if not safe_path.exists():
        raise FileNotFoundError(f"SAFE path not found: {safe_path}")

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

    ## Raise an error if the R10m directory does not exist
    if not r10m_dir.exists():
        raise FileNotFoundError(f"R10m directory not found in granule: {granule_dir}")

    ## Raise an error if the band file does not exist
    if len(list(r10m_dir.glob(f"*_{band}_10m.jp2"))) == 0:
        raise FileNotFoundError(f"Band file not found: {band}")

    band_file = next(r10m_dir.glob(f"*_{band}_10m.jp2"))
    return band_file


def write_raster(image: np.ndarray, profile: dict, output_path: Path) -> None:
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
    # Raise an error if the output directory does not exist
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")

    # Update the profile for the output raster, as its dtype and count may differ from the input
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        driver="GTiff",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(image, 1)  # Select first band for writing


def write_png(image: np.ndarray, output_path: Path, cmap="gray") -> None:
    """
    Save a raster image as a PNG file.

    Parameters
    ----------
    image : numpy.ndarray
        Raster values to save.
    output_path : Path
        Path to the output PNG file.
    """
    # Raise an error if the output directory does not exist
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")

    # Raise an error if the input image is not 2D
    if image.ndim != 2:
        raise ValueError("Raster image must be a 2D array.")

    plt.imsave(output_path, image, cmap=cmap)
