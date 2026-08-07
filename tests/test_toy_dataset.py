from opencropphenotyping.io import read_band


def test_toy_dataset(project_root):
    toy_dir = project_root / "data" / "raw"  / "toy_datasets"  / "toy_dataset_1"

    b03 = toy_dir / "toy_image_b03_10m.tif"
    b04 = toy_dir / "toy_image_b04_10m.tif"
    b05 = toy_dir / "toy_image_b05_20m.tif"
    b08 = toy_dir / "toy_image_b08_10m.tif"

    assert b03.exists()
    assert b04.exists()
    assert b05.exists()
    assert b08.exists()

    green, profile_green = read_band(b03)
    red, profile_red = read_band(b04)
    red_edge, profile_red_edge = read_band(b05)
    nir, profile_nir = read_band(b08)

    assert red.shape == nir.shape
    assert red.dtype == nir.dtype
    assert green.shape == nir.shape
    assert red_edge.shape != nir.shape

    assert profile_red["crs"] == profile_nir["crs"]
    assert profile_red["transform"] == profile_nir["transform"]
    assert profile_green["crs"] == profile_nir["crs"]
    assert profile_green["transform"] == profile_nir["transform"]
    assert profile_red_edge["crs"] == profile_nir["crs"]
    assert profile_red_edge["transform"] != profile_nir["transform"]
