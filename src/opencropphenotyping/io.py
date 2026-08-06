from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def build_band_catalog(
        input_dir: Path,
        bands: list[str],
        ) -> dict[str, dict[int, Path]]:
    """
    Build a catalog of available Sentinel-2 bands at different resolutions.

    Parameters
    ----------
    input_dir : Path
        Path to a Sentinel-2 directory.
    bands : list[str]
        List of band identifiers to include in the catalog.

    Returns
    -------
    dict[str, dict[int, Path]]
        Dictionary mapping each band identifier to its available
        resolutions and corresponding file paths.
    """
    catalog = {}
    for band in bands:
        catalog[band] = {}
        for resolution in [10, 20, 60]:
            try:
                band_path = find_band(
                    input_dir, 
                    band, 
                    resolution=resolution)
                catalog[band][resolution] = band_path
            except FileNotFoundError:
                continue  # Skip if the band is not found
    return catalog

def select_bands(
        catalog: dict[str, dict[int, Path]],
        resolution: int = 10,
        output_dir: Path | None = None
        ) -> dict[str, Path]:
    """
    Select Sentinel-2 bands at the requested spatial resolution.

    Bands already available at the target resolution are used directly.
    Missing resolutions are obtained by resampling the closest available
    resolution.

    Parameters
    ----------
    catalog : dict[str, dict[int, Path]]
        Catalog of available bands and their resolutions.
    resolution : int, optional
        Target spatial resolution in meters. Default is 10.
    output_dir : Path, optional
        Directory where resampled bands are written.

    Returns
    -------
    dict[str, Path]
        Selected bands at the requested resolution.
    """
    selected_bands = {}

    for band_name, resolutions in catalog.items():

        if not resolutions:
            raise FileNotFoundError(
                f"No available resolution found for band {band_name}."
            )

        # Band already available at target resolution
        if resolution in resolutions:
            selected_bands[band_name] = resolutions[resolution]
            continue

        # Find the closest available resolution
        closest_resolution = min(
            resolutions,
            key=lambda r: abs(r - resolution)
        )

        image, profile = read_band(resolutions[closest_resolution])

        resampled_image, resampled_profile = resample_raster(
            image,
            profile,
            target_resolution=resolution
        )

        if output_dir is None:
            raise ValueError(
                f"Output directory required to resample band {band_name}."
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        resampled_band_path = (
            output_dir /
            f"Resampled_{band_name}_from_R{closest_resolution}m_to_R{resolution}m.tif"
        )

        write_raster(
            resampled_image,
            resampled_profile,
            resampled_band_path
        )

        selected_bands[band_name] = resampled_band_path

    return selected_bands
    
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


def find_band(
    input_dir: Path,
    band: str,
    resolution: int | None,
) -> Path:
    """
    Find the path to a specific Sentinel-2 band or raster band file.

    The function supports both standard Sentinel-2 SAFE products and
    simplified datasets containing band files directly in the input directory.

    Parameters
    ----------
    input_dir : Path
        Path to the Sentinel-2 SAFE directory or simplified dataset directory.
    band : str
        Band identifier (e.g., 'B04', 'B08').
    resolution : int | None
        Target resolution for the band file. Used for Sentinel-2 SAFE products.
        Ignored for simplified datasets.

    Returns
    -------
    Path
        Path to the band file.

    Raises
    ------
    FileNotFoundError
        If the expected directory or band file cannot be found.
    FileExistsError
        If multiple matching band files are found.
    """

    # Check whether the input directory follows the Sentinel-2 SAFE structure
    granule_dirs = list(input_dir.glob("GRANULE/*"))

    if granule_dirs:
        granule_dir = find_granule(input_dir)
        img_data_dir = granule_dir / "IMG_DATA"

        if resolution is not None:
            img_data_dir = img_data_dir / f"R{resolution}m"

        if not img_data_dir.exists():
            raise FileNotFoundError(
                f"IMG_DATA directory not found: {img_data_dir}"
            )

        band_files = list(
            img_data_dir.rglob(f"*_{band}_*.jp2")
        )

    else:
        # Simplified dataset: search directly in the input directory
        band_files = list(
            input_dir.glob(f"*{band}*.tif")
        )

    if len(band_files) == 0:
        raise FileNotFoundError(
            f"Band file not found: {band} in {input_dir}"
        )

    if len(band_files) > 1:
        raise FileExistsError(
            f"Multiple band files found for {band}: {band_files}"
        )

    return band_files[0]


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