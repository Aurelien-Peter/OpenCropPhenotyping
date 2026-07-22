from opencropphenotyping.io import read_band, find_band, find_granule, write_raster
import numpy as np
import pytest
import rasterio
from pathlib import Path
from rasterio.transform import from_origin

## Create paths 
@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[1]

@pytest.fixture
def safe_dir(project_root):
    return (
        project_root
        / "data"
        / "raw"
        / "sentinel_2"
        / "S2A_MSIL2A_20250804T104701_N0511_R051_T31TCJ_20250804T161517.SAFE"
    )

## Perform tests
def test_find_band_success(safe_dir):
    # Test find_band function
    band_path = find_band(safe_dir, "B04")
    assert band_path.exists(), "Band path does not exist."
    assert band_path.name.endswith("_B04_10m.jp2"), "Band path does not point to the correct file."

def test_find_band_wrong_directory():
    # Test find_band function with a wrong directory
    with pytest.raises(FileNotFoundError):
        find_band(Path("C:/wrong/path"), "B04")  # Assuming this directory does not exist

def test_find_band_unknown_band(safe_dir):
    # Test find_band function with an unknown band
    with pytest.raises(FileNotFoundError):
        find_band(safe_dir, "B99")  # Assuming B99 does not exist in the structure

def test_read_band_success(safe_dir):
    # Test read_band function
    band_path = find_band(safe_dir, "B04")
    image, profile = read_band(band_path)
    assert isinstance(image, np.ndarray), "Image is not a numpy array."
    assert image.ndim == 2, "Image is not 2D."
    assert image.dtype == np.uint16, "Image dtype is not uint16."
    assert profile['driver'] == 'JP2OpenJPEG', "Profile driver is not JP2OpenJPEG."
    assert profile['count'] == 1, "Profile count is not 1."
    assert profile['dtype'] == 'uint16', "Profile dtype is not uint16."
    assert profile['width'] == image.shape[1], "Profile width does not match image width."
    assert profile['height'] == image.shape[0], "Profile height does not match image height."
    assert profile['crs'] is not None, "Profile CRS is None."

def test_read_band_success2(tmp_path):

    raster_path = tmp_path / "test.tif"

    data = np.array(
        [
            [1, 2],
            [3, 4]
        ],
        dtype=np.uint16
    )

    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "uint16",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 0, 1, 1)
    }

    with rasterio.open(
        raster_path,
        "w",
        **profile
    ) as dst:
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

def test_write_raster(tmp_path):
    # Test write_raster function
    image = np.random.rand(100, 100).astype(np.float32)
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'width': 100,
        'height': 100,
        'crs': "EPSG:4326",
        'transform': None
    }
    output_path = tmp_path / "output.tif"
    write_raster(image, profile, output_path)
    image2, profile2 = read_band(output_path)
    assert np.allclose(image, image2), "Written and read images do not match."
    assert image2.dtype == 'float32', "Read image dtype is not float32."
    assert profile2['driver'] == 'GTiff', "Output raster file does not have the correct driver."
    assert profile2['dtype'] == 'float32', "Output raster file does not have float32 dtype."
    assert profile2['width'] == 100 and profile2['height'] == 100, "Output raster file does not have the correct shape."
    assert profile2['count'] == 1, "Output raster file does not have the correct count."
    assert profile2['crs'] == "EPSG:4326", "Output raster file does not have the correct CRS."