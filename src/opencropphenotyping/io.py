from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


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


def find_band(safe_path: Path, band: str, resolution: int | None) -> Path:
    """
    Find the path to a specific band in a Sentinel-2 SAFE directory.

    Parameters
    ----------
    safe_path : Path
        Path to the Sentinel-2 SAFE directory.
    band : str
        Band identifier (e.g., 'B04', 'B08').
    resolution : int | None
        Target resolution for the band file.

    Returns
    -------
    Path
        Path to the band file.
    """

    granule_dir = find_granule(safe_path)
    img_data_dir = granule_dir / "IMG_DATA"

    if(resolution is not None):
        img_data_dir = img_data_dir / f"R{resolution}m"

    ## Raise an error if the img_data_dir directory does not exist
    if not img_data_dir.exists():
        raise FileNotFoundError(f"IMG_DATA directory not found: {img_data_dir}")

    # Search band file in all subdirectories of the granule directory
    band_files = list(img_data_dir.rglob(f"*_{band}_*.jp2"))

    ## Raise an error if the band file does not exist
    if len(band_files) == 0:
        raise FileNotFoundError(f"Band file not found: {band} in {img_data_dir}")

    ## Raise an error if there are multiple band files
    if len(band_files) > 1:
        raise FileExistsError(f"Multiple band files found for {band}: {band_files}")

    band_file = next(img_data_dir.rglob(f"*_{band}_*.jp2"))
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


def resample_raster(image: np.ndarray, 
                  profile: dict, 
                  target_resolution: float | None = None,
                  target_profile: dict | None = None,
                  resampling: Resampling = Resampling.bilinear,
                  ) -> tuple[np.ndarray, dict]:
    """
    Resample a raster image to a target resolution.

    Parameters
    ----------
    image : numpy.ndarray
        Input raster values.
    profile : dict
        Raster metadata.
    target_resolution : float | None
        Target resolution for the resampled image.
    target_profile : dict | None
        Target raster metadata for the resampled image.

    Returns
    -------
    numpy.ndarray
        Resampled raster values.
    resampled_profile : dict
        Updated raster metadata for the resampled image.
    """
    # Check that the image is 2D
    if image.ndim != 2:
        raise ValueError("Raster image must be a 2D array.")

    # Check that either target_resolution or target_profile is provided
    if(target_resolution is None and target_profile is None):
        raise ValueError("Either target_resolution or target_profile must be provided.")

    # Check that target resolution > 0
    if(target_resolution is not None and target_resolution <= 0):
        raise ValueError("target_resolution must be greater than 0.")

    # Get target shape based on either target_resolution or target_profile
    if(target_resolution is not None):
        # If target_resolution is provided, calculate the target shape based on the desired resolution
        target_height = round(image.shape[0] * abs(profile['transform'].e) / target_resolution)
        target_width = round(image.shape[1] * abs(profile['transform'].a) / target_resolution)
        target_shape = (target_height, target_width)
    else:
        # If target_profile is provided, use its height and width for the target shape
        assert target_profile is not None
        target_height = target_profile['height']
        target_width = target_profile['width']
        target_shape = (target_height, target_width)
        # Calculate the target resolution based on the target profile's transform
        target_resolution = abs(target_profile['transform'].a)

    # Get the updated profile for the resampled image
    resampled_profile = profile.copy()
    resampled_profile.update(
        height=target_height,
        width=target_width,
        transform=rasterio.Affine(
            target_resolution, profile['transform'].b, profile['transform'].c,
            profile['transform'].d, -target_resolution, profile['transform'].f
        )
    )

    # Create the destination array for the resampled image
    resampled_image = np.empty(target_shape, dtype=np.float32)

    # Resample the image to the target shape using bilinear interpolation
    if target_shape != image.shape:
        resampled_image = reproject(
            source=image,
            destination=resampled_image,
            src_transform=profile['transform'],
            src_crs=profile['crs'],
            dst_crs=profile['crs'],
            dst_transform=resampled_profile['transform'],
            dst_nodata=profile.get('nodata'),
            resampling=resampling
        )[0]

    return resampled_image, resampled_profile