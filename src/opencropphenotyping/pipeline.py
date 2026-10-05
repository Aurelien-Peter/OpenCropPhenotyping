import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from opencropphenotyping.indices import compute_indexes
from opencropphenotyping.io import build_band_catalog, read_band, select_bands, write_grayscale_png, write_png, write_raster
from opencropphenotyping.segmentation import detect_plants
from opencropphenotyping.statistics import compute_statistics
from opencropphenotyping.traits import compute_crop_cover, create_vegetation_mask_ndvi


@dataclass
class ProcessingResult:
    product_name: str
    indices: dict[str, np.ndarray]
    profile: dict
    statistics: dict[str, dict[str, float]]
    vegetation_mask: np.ndarray | None
    crop_cover: float | None

@dataclass
class UAVSegmentationResult:
    """Results produced by the UAV plant-detection pipeline."""

    plants: pd.DataFrame
    best_angle: float
    row_profile: np.ndarray
    row_positions: np.ndarray
    row_boundaries: np.ndarray
    vegetation_mask: np.ndarray
    rotated_image: np.ndarray
    rotated_exg: np.ndarray

def process_sentinel2(
    input_dir: Path,
    output_dir: Path | None = None,
    indices: list[str] | None = None,
    ndvi_threshold: float = 0.3,
    resolution: int = 10,
) -> ProcessingResult:
    """ 
    Process Sentinel-2 imagery and compute vegetation-related outputs. 
    The pipeline discovers the required Sentinel-2 bands, selects or 
    resamples them to the requested resolution, computes the requested 
    vegetation indices, and calculates statistics for each computed index. 

    If NDVI is successfully computed, a vegetation mask and crop cover 
    percentage are also calculated. If NDVI cannot be computed because 
    required bands are unavailable, the pipeline continues without 
    creating the vegetation mask or crop cover. 

    Parameters 
    ---------- 
    input_dir : Path 
        Directory containing the input Sentinel-2 data. 
    output_dir : Path | None, optional
        Directory where the resampled rasters will be exported.
        If ``None``, results are not exported.
    indices : list[str] | None, optional 
        Vegetation indices to compute. Available indices are ``"ndvi"``, 
        ``"savi"``, ``"ndre"``, and ``"gndvi"``. 
        If ``None``, all available indices are requested. 
        Indices whose required bands are unavailable are not computed. 
    ndvi_threshold : float, optional 
        NDVI threshold used to create the vegetation mask. 
        Default is 0.3. 
    resolution : int, optional 
        Target spatial resolution in metres for the input bands. 
        Available Sentinel-2 resolutions are 10, 20, and 60 metres. 
        Bands are resampled when the requested resolution is not 
        directly available. 
        Default is 10. 
        
    Returns 
    ------- 
    ProcessingResult 
        Processing results containing: 
        
        - ``indices``: computed vegetation indices as NumPy arrays. 
        - ``profile``: raster profile used for exporting the results. 
        - ``statistics``: statistics computed for each vegetation index. 
        - ``vegetation_mask``: binary vegetation mask based on NDVI, 
            or ``None`` if NDVI could not be computed. 
        - ``crop_cover``: proportion of pixels classified as vegetation, 
            or ``None`` if NDVI could not be computed. 
    """

    # 0. Extract product name from input directory
    product_name = input_dir.name

    # 1. Find available bands
    catalog = build_band_catalog(
        input_dir,
        bands=["B03", "B04", "B05", "B08"],
    )

    # 2. Select / resample bands
    bands = select_bands(
        catalog,
        resolution=resolution,
        output_dir=output_dir
    )

    # 3. Read selected bands
    band_images = {}
    band_profiles = {}

    for band_name, band_path in bands.items():
        image, profile = read_band(band_path)
        band_images[band_name] = image
        band_profiles[band_name] = profile

    profile = next(iter(band_profiles.values()))

    # 4. Compute requested vegetation indices
    if indices is None:
        indices = ["ndvi", "savi", "ndre", "gndvi"]

    red_band = band_images.get("B04")
    nir_band = band_images.get("B08")
    green_band = band_images.get("B03")
    red_edge_band = band_images.get("B05")

    required_bands = set()

    if "ndvi" in indices or "savi" in indices:
        required_bands.update(["B04", "B08"])

    if "ndre" in indices:
        required_bands.update(["B05", "B08"])

    if "gndvi" in indices:
        required_bands.update(["B03", "B08"])

    for band in required_bands:
        if band not in band_images:
            warnings.warn(
                f"{band} is not available. "
                "Some requested vegetation indices may not be computed.",
                UserWarning,
            )

    computed_indices = compute_indexes(
        red_band=red_band,
        nir_band=nir_band,
        green_band=green_band,
        red_edge_band=red_edge_band,
        indices=indices
    )
    
    # 5. Compute statistics
    statistics = {}

    for index_name, index_raster in computed_indices.items():
        statistics[index_name] = compute_statistics(index_raster)
        
    # 6. Create vegetation mask and compute crop cover
    vegetation_mask = None
    crop_cover = None

    if "ndvi" in computed_indices:
        vegetation_mask = create_vegetation_mask_ndvi(
            computed_indices["ndvi"],
            threshold=ndvi_threshold,
        )

        # 7. Compute crop cover
        crop_cover = compute_crop_cover(vegetation_mask)

    # 8. Return all results

    return ProcessingResult(
        product_name=product_name,
        indices=computed_indices,
        profile=profile,
        statistics=statistics,
        vegetation_mask=vegetation_mask,
        crop_cover=crop_cover,
    )

def export_results(
    result: ProcessingResult,
    product_output_dir: Path,
) -> None:
    """
    Export processing results to the specified output directory.

    The exported results include the computed vegetation indices,
    statistics, and, when available, the vegetation mask and crop cover.

    The following directory structure is created::

        product_output_dir/
        ├── indices/
        │   ├── ndvi.tif
        │   ├── savi.tif
        │   ├── ndre.tif
        │   └── gndvi.tif
        ├── vegetation_mask.tif
        ├── statistics.csv
        └── crop_cover.txt

    The vegetation mask and crop cover files are only created when the
    corresponding values are available in ``result``.

    Parameters
    ----------
    result : ProcessingResult
        Processing results containing vegetation indices, raster profile,
        statistics, vegetation mask, and crop cover.
    product_output_dir : Path
        Directory where the processing results for the specific product will be exported.

    Returns
    -------
    None
    """
    product_output_dir.mkdir(parents=True, exist_ok=True)
    indices_dir = product_output_dir / "indices"
    indices_dir.mkdir(exist_ok=True)

    # Export indices
    for index_name, index_raster in result.indices.items():
        index_path = indices_dir / f"{index_name}.tif"
        write_raster(index_raster, result.profile, index_path)

    # Export statistics
    stats_path = product_output_dir / "statistics.csv"
    statistics_df = pd.DataFrame.from_dict(
        result.statistics,
        orient="index",
    )
    statistics_df.index.name = "index"
    statistics_df.to_csv(stats_path, index=True)

    # Export vegetation mask
    if result.vegetation_mask is not None:
        mask_path = product_output_dir / "vegetation_mask.tif"
        write_raster(
            result.vegetation_mask,
            result.profile,
            mask_path,
        )

    # Export crop cover
    if result.crop_cover is not None:
        cover_path = product_output_dir / "crop_cover.txt"
        with open(cover_path, "w", encoding="utf-8") as f:
            f.write(f"Crop Cover: {result.crop_cover:.2f}\n")

def process_uav_segmentation(
    image_path: Path,
    geotiff_path: Path,
    geopackage_path: Path,
    best_angle: float | None = None,
    threshold: float = 25.0,
    vegetation_fraction_threshold: float = 0.04,
    export_intermediate: bool = False,
) -> UAVSegmentationResult:
    """
    Process a UAV RGB image and detect crop plants.

    Parameters
    ----------
    image_path : Path
        Path to the UAV RGB image.
    geotiff_path : Path
        Path to the georeferenced GeoTIFF.
    geopackage_path : Path
        Path to the experimental plot GeoPackage.
    best_angle : float | None, optional
        Crop-row orientation angle. If omitted, it is estimated automatically.
        Default is None.
    threshold : float, optional
        ExG threshold used for vegetation segmentation. Default is 25.0.
    vegetation_fraction_threshold : float, optional
        Vegetation fraction threshold used to classify planting positions.
        Default is 0.04.
    export_intermediate : bool, optional
        If True, all intermediate results (rotated image, vegetation mask, etc.) will be exported.
        Default is True.

    Returns
    -------
    UAVSegmentationResult
        Results produced by the UAV plant-detection pipeline, including detected plants,
        best angle, row profile, row positions, row boundaries, vegetation mask,
        rotated image, and rotated ExG.
    """
    (
        best_angle,
        rotated_img,
        rotated_exg,
        vegetation_mask,
        row_profile,
        peaks,
        boundaries,
        row_images,
        row_masks,
        row_detections,
        plants_df,
    ) = detect_plants(
        image_path=image_path,
        geotiff_path=geotiff_path,
        geopackage_path=geopackage_path,
        best_angle=best_angle,
        threshold=threshold,
        vegetation_fraction_threshold=vegetation_fraction_threshold,
        export_all=export_intermediate
    )

    return UAVSegmentationResult(
        plants=plants_df,
        best_angle=best_angle,
        row_profile=row_profile,
        row_positions=peaks,
        row_boundaries=boundaries,
        vegetation_mask=vegetation_mask,
        rotated_image=rotated_img,
        rotated_exg=rotated_exg
    )

def export_uav_results(
    result: UAVSegmentationResult,
    output_dir: Path,
    export_intermediate: bool = False,
) -> None:
    """
    Export UAV plant-detection results.

    Parameters
    ----------
    result : UAVSegmentationResult
        Results produced by ``detect_plants``.
    output_dir : Path
        Directory where results will be saved.
    export_intermediate : bool, optional
        If True, all intermediate results will be exported.
        Default is False.
    """ 
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.plants.to_csv(
        output_dir / "plants.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "row_position": result.row_positions,
        }
    ).to_csv(
        output_dir / "row_positions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "boundary": result.row_boundaries,
        }
    ).to_csv(
        output_dir / "row_boundaries.csv",
        index=False,
    ) 

    if(export_intermediate):
        print(
            "rotated_image:",
            result.rotated_image.shape,
            result.rotated_image.dtype,
        )

        print(
            "vegetation_mask:",
            result.vegetation_mask.shape,
            result.vegetation_mask.dtype,
        )
        write_png(
            result.rotated_image,
            output_dir / "rotated_image.png",
        )

        write_grayscale_png(
            result.vegetation_mask,
            output_dir / "vegetation_mask.png",
        )
    print(f"Export complete. Results saved to: {output_dir}")