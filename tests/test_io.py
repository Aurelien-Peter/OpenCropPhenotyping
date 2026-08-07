from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from opencropphenotyping.io import (
    build_band_catalog,
    find_band,
    find_granule,
    read_band,
    resample_raster,
    select_bands,
    write_png,
    write_raster,
)


## Create paths
@pytest.fixture
def safe_dir(project_root):
    return project_root / "data" / "raw" / "sentinel_2" / "S2A_MSIL2A_20250804T104701_N0511_R051_T31TCJ_20250804T161517.SAFE"

## Create variables
@pytest.fixture
def raster_profile():
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": 100,
        "height": 100,
        "crs": "EPSG:4326",
        "transform": rasterio.Affine(10, 0, 0, 0, -10, 0),
    }

## Perform tests
def test_build_band_catalog(monkeypatch, tmp_path):

    def mock_find_band(safe_path, band, resolution):
        return safe_path / f"{band}_{resolution}m.jp2"

    monkeypatch.setattr(
        "opencropphenotyping.io.find_band",
        mock_find_band
    )

    catalog = build_band_catalog(
        input_dir=tmp_path,
        bands=["B03", "B04"]
    )

    expected = {
        "B03": {
            10: tmp_path / "B03_10m.jp2",
            20: tmp_path / "B03_20m.jp2",
            60: tmp_path / "B03_60m.jp2",
        },
        "B04": {
            10: tmp_path / "B04_10m.jp2",
            20: tmp_path / "B04_20m.jp2",
            60: tmp_path / "B04_60m.jp2",
        },
    }

    assert catalog == expected

def test_build_band_catalog_missing_band(monkeypatch, tmp_path):
    def mock_find_band(safe_path, band, resolution):
        raise FileNotFoundError

    monkeypatch.setattr(
        "opencropphenotyping.io.find_band",
        mock_find_band
    )

    catalog = build_band_catalog(
        input_dir=tmp_path,
        bands=["B05"]
    )

    assert catalog == {"B05": {}}

def test_build_band_catalog_partial_resolutions(monkeypatch, tmp_path):

    def mock_find_band(safe_path, band, resolution):
        if band == "B03" and resolution in [10, 20]:
            return safe_path / f"{band}_{resolution}m.jp2"

        raise FileNotFoundError

    monkeypatch.setattr(
        "opencropphenotyping.io.find_band",
        mock_find_band
    )

    catalog = build_band_catalog(
        input_dir=tmp_path,
        bands=["B03"]
    )

    expected = {
        "B03": {
            10: tmp_path / "B03_10m.jp2",
            20: tmp_path / "B03_20m.jp2",
        }
    }

    assert catalog == expected

def test_select_bands_existing_resolution(tmp_path):

    catalog = {
        "B03": {
            10: tmp_path / "B03_10m.jp2",
            20: tmp_path / "B03_20m.jp2",
            60: tmp_path / "B03_60m.jp2",
        },
        "B04": {
            10: tmp_path / "B04_10m.jp2",
            20: tmp_path / "B04_20m.jp2",
            60: tmp_path / "B04_60m.jp2",
        },
    }

    selected_bands = select_bands(catalog, resolution=10)

    expected = {
        "B03": tmp_path / "B03_10m.jp2",
        "B04": tmp_path / "B04_10m.jp2",
    }

    assert selected_bands == expected

def test_select_bands_resample(monkeypatch, tmp_path):

    catalog = {
        "B05": {
            10: tmp_path / "B05_10m.jp2",
            60: tmp_path / "B05_60m.jp2",
        }
    }

    # Mock read_band
    def mock_read_band(path):
        assert path == tmp_path / "B05_10m.jp2"
        return "image", "profile"

    monkeypatch.setattr(
        "opencropphenotyping.io.read_band",
        mock_read_band
    )

    # Mock resample_raster
    def mock_resample_raster(image, profile, target_resolution):
        assert image == "image"
        assert profile == "profile"
        assert target_resolution == 20

        return "resampled_image", "resampled_profile"

    monkeypatch.setattr(
        "opencropphenotyping.io.resample_raster",
        mock_resample_raster
    )

    # Mock write_raster
    def mock_write_raster(image, profile, output_path):
        assert image == "resampled_image"
        assert profile == "resampled_profile"
        output_path.touch()

    monkeypatch.setattr(
        "opencropphenotyping.io.write_raster",
        mock_write_raster
    )

    selected_bands = select_bands(
        catalog,
        resolution=20,
        output_dir=tmp_path
    )

    expected_path = (
        tmp_path /
        "Resampled_B05_from_R10m_to_R20m.tif"
    )

    assert selected_bands == {
        "B05": expected_path
    }

def test_select_bands_empty_band():

    catalog = {
        "B06": {}
    }

    with pytest.raises(
        FileNotFoundError,
        match="No available resolution found for band B06"
    ):
        select_bands(catalog, resolution=10)

def test_find_band_success(safe_dir):
    # Test find_band function
    band_path = find_band(safe_dir, "B04", resolution = 10)
    assert band_path.exists(), "Band path does not exist."
    assert band_path.name.endswith("_B04_10m.jp2"), "Band path does not point to the correct file."

def test_find_band_wrong_directory():
    # Test find_band function with a wrong directory
    with pytest.raises(FileNotFoundError):
        find_band(Path("C:/wrong/path"), "B04", resolution = 10)  # Assuming this directory does not exist

def test_find_band_unknown_band(safe_dir):
    # Test find_band function with an unknown band
    with pytest.raises(FileNotFoundError):
        find_band(safe_dir, "B99", resolution = 10)  # Assuming B99 does not exist in the structure

def test_find_band_multiple_bands(safe_dir):
    # Test find_band function with multiple bands found
    with pytest.raises(FileExistsError):
        find_band(safe_dir, "B04", resolution = None)

def test_read_band_success(safe_dir):
    # Test read_band function
    band_path = find_band(safe_dir, "B04", resolution = 10)
    image, profile = read_band(band_path)
    assert isinstance(image, np.ndarray), "Image is not a numpy array."
    assert image.ndim == 2, "Image is not 2D."
    assert image.dtype == np.uint16, "Image dtype is not uint16."
    assert profile["driver"] == "JP2OpenJPEG", "Profile driver is not JP2OpenJPEG."
    assert profile["count"] == 1, "Profile count is not 1."
    assert profile["dtype"] == "uint16", "Profile dtype is not uint16."
    assert profile["width"] == image.shape[1], "Profile width does not match image width."
    assert profile["height"] == image.shape[0], "Profile height does not match image height."
    assert profile["crs"] is not None, "Profile CRS is None."


def test_read_band_success2(tmp_path):

    raster_path = tmp_path / "test.tif"

    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 1, 1),
    }

    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(data, 1)

    image, metadata = read_band(raster_path)

    assert isinstance(image, np.ndarray)
    assert image.shape == (2, 2)
    assert image.dtype == np.uint16

    assert metadata["driver"] == "GTiff"
    assert metadata["count"] == 1
    assert metadata["dtype"] == "uint16"
    assert metadata["crs"] is not None


def test_read_band_wrong_directory():
    # Test read_band function with a wrong directory
    with pytest.raises(FileNotFoundError):
        read_band(Path("C:/wrong/path/B04_10m.jp2"))  # Assuming this directory does not exist


def test_find_granule_wrong_directory():
    # Test find_granule function with a wrong directory
    with pytest.raises(FileNotFoundError):
        find_granule(Path("C:/wrong/path"))  # Assuming this directory does not exist


def test_write_raster(tmp_path):
    # Test write_raster function
    image = np.random.rand(100, 100).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": 100,
        "height": 100,
        "crs": "EPSG:4326",
        "transform": None,
    }
    output_path = tmp_path / "output.tif"
    write_raster(image, profile, output_path)
    image2, profile2 = read_band(output_path)
    assert np.allclose(image, image2), "Written and read images do not match."
    assert image2.dtype == "float32", "Read image dtype is not float32."
    assert profile2["driver"] == "GTiff", "Output raster file does not have the correct driver."
    assert profile2["dtype"] == "float32", "Output raster file does not have float32 dtype."
    assert profile2["width"] == 100 and profile2["height"] == 100, "Output raster file does not have the correct shape."
    assert profile2["count"] == 1, "Output raster file does not have the correct count."
    assert profile2["crs"] == "EPSG:4326", "Output raster file does not have the correct CRS."


def test_write_raster_nonexistent_directory(tmp_path):
    # Test write_raster function with a non-existent directory
    image = np.random.rand(100, 100).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": 100,
        "height": 100,
        "crs": "EPSG:4326",
        "transform": None,
    }
    output_path = tmp_path / "nonexistent_dir" / "output.tif"
    with pytest.raises(FileNotFoundError):
        write_raster(image, profile, output_path)


def test_write_png(tmp_path):
    # Test write_png function
    image = np.random.rand(100, 100).astype(np.float32)
    output_path = tmp_path / "output.png"
    write_png(image, output_path)
    assert output_path.exists(), "PNG file was not created."


def test_write_png_nonexistent_directory(tmp_path):
    # Test write_png function with a non-existent directory
    image = np.random.rand(100, 100).astype(np.float32)
    output_path = tmp_path / "nonexistent_dir" / "output.png"
    with pytest.raises(FileNotFoundError):
        write_png(image, output_path)


def test_write_png_wrong_dimension(tmp_path):
    # Test write_png function with a non-2D array
    image = np.random.rand(100, 100, 3).astype(np.float32)  # 3D array
    output_path = tmp_path / "output.png"
    with pytest.raises(ValueError, match="Raster image must be a 2D array."):
        write_png(image, output_path)

def test_resample_raster_3d(raster_profile):
    # Test resample_master function with non-2D arrays
    image_3D = np.random.rand(100, 100, 3).astype(np.float32)  # 3D array
    with pytest.raises(ValueError, match="Raster image must be a 2D array."):
        resample_raster(image_3D, 
                        profile=raster_profile, 
                        target_resolution = 10)


def test_resample_raster_1d(raster_profile):
    # Test resample_master function with non-2D arrays
    image_1D = np.random.rand(100).astype(np.float32)  # 1D array
    with pytest.raises(ValueError, match="Raster image must be a 2D array."):
        resample_raster(image_1D, 
                        profile=raster_profile, 
                        target_resolution = 10)

def test_resample_raster_missing_target(raster_profile):
    # Test resample_master function without targets
    image = np.random.rand(100, 100).astype(np.float32)
    with pytest.raises(ValueError, match="Either target_resolution or target_profile must be provided."):
        resample_raster(image, 
                        profile=raster_profile, 
                        target_resolution = None,
                        target_profile=None)

def test_resample_raster_invalid_resolution(raster_profile):
    # Test resample_master with negative resolution
    image = np.random.rand(100, 100).astype(np.float32)
    with pytest.raises(ValueError, match="target_resolution must be greater than 0."):
        resample_raster(image, 
                        profile=raster_profile, 
                        target_resolution=-10)

def test_resample_raster_target_resolution(raster_profile):
    # Test resample_master specifiying target resolution
    image = np.random.rand(100, 100).astype(np.float32)
    resampled_image, resampled_profile = resample_raster(image, 
                    profile=raster_profile, 
                    target_resolution=5)
    assert resampled_image.shape == (200, 200)
    assert resampled_profile["height"] == 200
    assert resampled_profile["width"] == 200
    assert abs(resampled_profile["transform"].a) == 5
    assert resampled_image.dtype == np.float32

def test_resample_raster_target_profile(raster_profile):
    # Test resample_master specifiying target profile
    image = np.random.rand(100, 100).astype(np.float32)
    target_profile = raster_profile.copy()
    target_profile.update(
        width=300,
        height=150,
    )
    resampled_image, resampled_profile = resample_raster(image, 
                    profile=raster_profile, 
                    target_profile=target_profile)
    assert resampled_image.shape == (150, 300)
    assert resampled_profile["height"] == 150
    assert resampled_profile["width"] == 300
    assert resampled_image.dtype == np.float32

