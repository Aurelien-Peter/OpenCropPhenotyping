from pathlib import Path

import geopandas as gpd
import pytest


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[1]

@pytest.fixture
def toy_dataset(project_root):
    dataset_dir = (
        project_root
        / "data"
        / "raw"
        / "toy_datasets"
        / "toy_segmentation"
    )

    image_path = dataset_dir / "103_DSC01167.png"
    geotiff_path = (
        dataset_dir
        / "georeferenced"
        / "103_DSC01167.tif"
    )
    geopackage_path = (
        dataset_dir
        / "geopackages"
        / "103_DSC01167.gpkg"
    )

    plots = gpd.read_file(geopackage_path)
    plot = plots.iloc[0]

    return {
        "dataset_dir": dataset_dir,
        "image_path": image_path,
        "geotiff_path": geotiff_path,
        "geopackage_path": geopackage_path,
        "geometry": plot["geometry"],
        "image_name": plot["image_name"],
        "n_plants": plot["n_plants"],
        "n_rows": plot["n_rows"],
    }