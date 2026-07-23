from pathlib import Path

import pytest

from opencropphenotyping.io import read_band

## Create paths 
@pytest.fixture
def project_root():
    return Path(__file__).resolve().parents[1]

def test_toy_dataset(project_root):
    toy_dir = project_root / "data" / "toy dataset"

    b04 = toy_dir / "toy_image_b04.tif"
    b08 = toy_dir / "toy_image_b08.tif"

    assert b04.exists()
    assert b08.exists()

    red, profile_red = read_band(b04)
    nir, profile_nir = read_band(b08)

    assert red.shape == nir.shape
    assert red.dtype == nir.dtype

    assert profile_red["crs"] == profile_nir["crs"]
    assert profile_red["transform"] == profile_nir["transform"]