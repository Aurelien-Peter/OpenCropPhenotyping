import numpy as np
import pytest

from opencropphenotyping.pipeline import process_sentinel2


def test_process_sentinel2_with_ndvi(monkeypatch, tmp_path):

    mat_one = np.ones((10, 10))

    # Mock build_band_catalog
    def mock_build_band_catalog(input_dir, bands):
        return {
            "B03": {10: tmp_path / "B03_10m.jp2"},
            "B04": {10: tmp_path / "B04_10m.jp2"},
            "B05": {20: tmp_path / "B05_20m.jp2"},
            "B08": {10: tmp_path / "B08_10m.jp2"},
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.build_band_catalog",
        mock_build_band_catalog,
    )

    # Mock select_bands
    def mock_select_bands(catalog, resolution):
        return {
            "B03": tmp_path / "B03_10m.jp2",
            "B04": tmp_path / "B04_10m.jp2",
            "B05": tmp_path / "B05_10m.jp2",
            "B08": tmp_path / "B08_10m.jp2",
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.select_bands",
        mock_select_bands,
    )

    # Mock read_band
    def mock_read_band(path):
        return mat_one, {}

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.read_band",
        mock_read_band,
    )

    # Mock compute_indexes
    def mock_compute_indexes(
        red_band,
        nir_band,
        green_band,
        red_edge_band,
        indices
    ):
        assert indices == ['ndvi', 'savi', 'ndre', 'gndvi']  # Default indices should be used
        return {
            "ndvi": mat_one,
            "savi": mat_one,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_indexes",
        mock_compute_indexes,
    )

    # Mock compute_statistics
    def mock_compute_statistics(image):
        return {
            "mean": 1.0,
            "std": 0.0,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_statistics",
        mock_compute_statistics,
    )

    # Mock create_vegetation_mask
    vegetation_mask = mat_one

    def mock_create_vegetation_mask(image, threshold):
        assert image is mat_one
        assert threshold == 0.3
        return vegetation_mask

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.create_vegetation_mask",
        mock_create_vegetation_mask,
    )

    # Mock compute_crop_cover
    def mock_compute_crop_cover(mask):
        assert mask is vegetation_mask
        return 0.75

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_crop_cover",
        mock_compute_crop_cover,
    )

    # Run pipeline
    result = process_sentinel2(
        input_dir=tmp_path,
        ndvi_threshold=0.3,
        resolution=10,
    )

    # Check result
    assert np.allclose(result.indices["ndvi"], mat_one)
    assert np.allclose(result.indices["savi"], mat_one)

    assert result.statistics["ndvi"]["mean"] == 1.0
    assert result.statistics["ndvi"]["std"] == 0.0
    assert result.statistics["savi"]["mean"] == 1.0
    assert result.statistics["savi"]["std"] == 0.0

    assert result.vegetation_mask is vegetation_mask
    assert result.crop_cover == 0.75

def test_process_sentinel2_without_ndvi(monkeypatch, tmp_path):

    mat_one = np.ones((10, 10))

    # Mock build_band_catalog
    def mock_build_band_catalog(input_dir, bands):
        return {
            "B03": {10: tmp_path / "B03_10m.jp2"},
            "B05": {20: tmp_path / "B05_20m.jp2"},
            "B08": {10: tmp_path / "B08_10m.jp2"},
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.build_band_catalog",
        mock_build_band_catalog,
    )

    # Mock select_bands
    def mock_select_bands(catalog, resolution):
        return {
            "B03": tmp_path / "B03_10m.jp2",
            "B05": tmp_path / "B05_20m.jp2",
            "B08": tmp_path / "B08_10m.jp2",
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.select_bands",
        mock_select_bands,
    )

    # Mock read_band
    def mock_read_band(path):
        return mat_one, {}

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.read_band",
        mock_read_band,
    )

    # Mock compute_indexes without NDVI
    def mock_compute_indexes(
        red_band,
        nir_band,
        green_band,
        red_edge_band,
        indices
    ):
        assert red_band is None
        assert nir_band is mat_one
        assert green_band is mat_one
        assert red_edge_band is mat_one
        assert indices == ['ndvi', 'savi', 'ndre', 'gndvi']  # Default indices should be used

        return {
            "ndre": mat_one,
            "gndvi": mat_one,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_indexes",
        mock_compute_indexes,
    )

    # Mock compute_statistics
    def mock_compute_statistics(image):
        return {
            "mean": 1.0,
            "std": 0.0,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_statistics",
        mock_compute_statistics,
    )

    # These functions must not be called
    def mock_create_vegetation_mask(image, threshold):
        raise AssertionError(
            "create_vegetation_mask should not be called without NDVI"
        )

    def mock_compute_crop_cover(mask):
        raise AssertionError(
            "compute_crop_cover should not be called without NDVI"
        )

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.create_vegetation_mask",
        mock_create_vegetation_mask,
    )

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_crop_cover",
        mock_compute_crop_cover,
    )

    # Run pipeline
    result = process_sentinel2(
        input_dir=tmp_path,
        resolution=10,
    )

    # Check result
    assert "ndre" in result.indices
    assert "gndvi" in result.indices
    assert "ndvi" not in result.indices

    assert result.vegetation_mask is None
    assert result.crop_cover is None

def test_process_sentinel2_selected_indices(monkeypatch, tmp_path):

    mat_one = np.ones((10, 10))

    # Mock build_band_catalog
    def mock_build_band_catalog(input_dir, bands):
        return {
            "B03": {10: tmp_path / "B03_10m.jp2"},
            "B04": {10: tmp_path / "B04_10m.jp2"},
            "B05": {20: tmp_path / "B05_20m.jp2"},
            "B08": {10: tmp_path / "B08_10m.jp2"},
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.build_band_catalog",
        mock_build_band_catalog,
    )

    # Mock select_bands
    def mock_select_bands(catalog, resolution):
        return {
            "B03": tmp_path / "B03_10m.jp2",
            "B04": tmp_path / "B04_10m.jp2",
            "B05": tmp_path / "B05_10m.jp2",
            "B08": tmp_path / "B08_10m.jp2",
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.select_bands",
        mock_select_bands,
    )

    # Mock read_band
    def mock_read_band(path):
        return mat_one, {}

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.read_band",
        mock_read_band,
    )

    # Mock compute_indexes
    def mock_compute_indexes(
        red_band,
        nir_band,
        green_band,
        red_edge_band,
        indices,
    ):
        assert indices == ["ndvi"]

        return {
            "ndvi": mat_one,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_indexes",
        mock_compute_indexes,
    )

    # Mock compute_statistics
    def mock_compute_statistics(image):
        return {
            "mean": 1.0,
            "std": 0.0,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_statistics",
        mock_compute_statistics,
    )

    # Mock create_vegetation_mask
    vegetation_mask = mat_one

    def mock_create_vegetation_mask(image, threshold):
        assert image is mat_one
        return vegetation_mask

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.create_vegetation_mask",
        mock_create_vegetation_mask,
    )

    # Mock compute_crop_cover
    def mock_compute_crop_cover(mask):
        return 0.75

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_crop_cover",
        mock_compute_crop_cover,
    )

    # Run pipeline
    result = process_sentinel2(
        input_dir=tmp_path,
        indices=["ndvi"],
        resolution=10,
    )

    # Check result
    assert result.indices == {"ndvi": mat_one}

    assert result.statistics == {
        "ndvi": {
            "mean": 1.0,
            "std": 0.0,
        }
    }

    assert result.vegetation_mask is vegetation_mask
    assert result.crop_cover == 0.75

def test_process_sentinel2_missing_required_band(monkeypatch, tmp_path):

    mat_one = np.ones((10, 10))

    # Mock build_band_catalog
    def mock_build_band_catalog(input_dir, bands):
        return {
            "B03": {10: tmp_path / "B03_10m.jp2"},
            "B08": {10: tmp_path / "B08_10m.jp2"},
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.build_band_catalog",
        mock_build_band_catalog,
    )

    # Mock select_bands
    def mock_select_bands(catalog, resolution):
        return {
            "B03": tmp_path / "B03_10m.jp2",
            "B08": tmp_path / "B08_10m.jp2",
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.select_bands",
        mock_select_bands,
    )

    # Mock read_band
    def mock_read_band(path):
        return mat_one, {}

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.read_band",
        mock_read_band,
    )

    # Mock compute_indexes
    def mock_compute_indexes(
        red_band,
        nir_band,
        green_band,
        red_edge_band,
        indices,
    ):
        assert indices == ["ndvi"]
        assert red_band is None
        assert nir_band is mat_one

        return {}

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.compute_indexes",
        mock_compute_indexes,
    )

    # Run pipeline and check warning
    with pytest.warns(
        UserWarning,
        match="B04 is not available",
    ):
        result = process_sentinel2(
            input_dir=tmp_path,
            indices=["ndvi"],
            resolution=10,
        )

    # NDVI cannot be computed
    assert result.indices == {}