from dataclasses import dataclass
import warnings

import numpy as np
from pathlib import Path
from opencropphenotyping.io import build_band_catalog, select_bands, read_band
from opencropphenotyping.indices import compute_indexes
from opencropphenotyping.statistics import compute_statistics
from opencropphenotyping.traits import create_vegetation_mask, compute_crop_cover

@dataclass
class ProcessingResult:
    indices: dict[str, np.ndarray]
    statistics: dict[str, dict[str, float]]
    vegetation_mask: np.ndarray | None
    crop_cover: float | None


def process_sentinel2(
    input_dir: Path,
    indices: list[str] | None = None,
    ndvi_threshold: float = 0.3,
    resolution: int = 10,
) -> ProcessingResult:

    # 1. Find available bands
    catalog = build_band_catalog(
        input_dir,
        bands=["B03", "B04", "B05", "B08"],
    )

    # 2. Select / resample bands
    bands = select_bands(
        catalog,
        resolution=resolution,
    )

    # 3. Read selected bands
    band_images = {}
    band_profiles = {}

    for band_name, band_path in bands.items():
        image, profile = read_band(band_path)
        band_images[band_name] = image
        band_profiles[band_name] = profile

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
        vegetation_mask = create_vegetation_mask(
            computed_indices["ndvi"],
            threshold=ndvi_threshold,
        )

        # 7. Compute crop cover
        crop_cover = compute_crop_cover(vegetation_mask)

    # 8. Return all results

    return ProcessingResult(
        indices=computed_indices,
        statistics=statistics,
        vegetation_mask=vegetation_mask,
        crop_cover=crop_cover,
    )