import numpy as np
import pytest

from opencropphenotyping.pipeline import ProcessingResult, process_sentinel2, export_results


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

def test_export_results(monkeypatch, tmp_path):

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
    }

    ndvi = np.ones((10, 10))
    savi = np.ones((10, 10)) * 2
    vegetation_mask = np.ones((10, 10), dtype=bool)

    result = ProcessingResult(
        indices={
            "ndvi": ndvi,
            "savi": savi,
        },
        profile=profile,
        statistics={
            "ndvi": {
                "mean": 1.0,
                "std": 0.0,
            },
            "savi": {
                "mean": 2.0,
                "std": 0.0,
            },
        },
        vegetation_mask=vegetation_mask,
        crop_cover=0.75,
    )

    written_rasters = {}

    def mock_write_raster(image, profile, output_path):
        written_rasters[output_path] = {
            "image": image,
            "profile": profile,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.write_raster",
        mock_write_raster,
    )

    export_results(
        result,
        tmp_path,
    )

    # Check indices
    assert tmp_path / "indices" / "ndvi.tif" in written_rasters
    assert tmp_path / "indices" / "savi.tif" in written_rasters

    assert written_rasters[
        tmp_path / "indices" / "ndvi.tif"
    ]["image"] is ndvi

    assert written_rasters[
        tmp_path / "indices" / "ndvi.tif"
    ]["profile"] == profile

    # Check vegetation mask
    mask_path = tmp_path / "vegetation_mask.tif"

    assert mask_path in written_rasters
    assert written_rasters[mask_path]["image"] is vegetation_mask

    # Check statistics
    statistics_path = tmp_path / "statistics.csv"

    assert statistics_path.exists()

    content = statistics_path.read_text()

    assert "index,mean,std" in content
    assert "ndvi,1.0,0.0" in content
    assert "savi,2.0,0.0" in content

    # Check crop cover
    crop_cover_path = tmp_path / "crop_cover.txt"

    assert crop_cover_path.exists()

    content = crop_cover_path.read_text()

    assert content == "Crop Cover: 0.75\n"

def test_export_results_no_vegetation_mask_crop_cover(monkeypatch, tmp_path):

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
    }

    ndvi = np.ones((10, 10))
    savi = np.ones((10, 10)) * 2
    vegetation_mask = None

    result = ProcessingResult(
        indices={
            "ndvi": ndvi,
            "savi": savi,
        },
        profile=profile,
        statistics={
            "ndvi": {
                "mean": 1.0,
                "std": 0.0,
            },
            "savi": {
                "mean": 2.0,
                "std": 0.0,
            },
        },
        vegetation_mask=vegetation_mask,
        crop_cover=None,
    )

    written_rasters = {}

    def mock_write_raster(image, profile, output_path):
        written_rasters[output_path] = {
            "image": image,
            "profile": profile,
        }

    monkeypatch.setattr(
        "opencropphenotyping.pipeline.write_raster",
        mock_write_raster,
    )

    export_results(
        result,
        tmp_path,
    )

    # Check vegetation mask
    mask_path = tmp_path / "vegetation_mask.tif"

    assert mask_path not in written_rasters

    # Check crop cover
    crop_cover_path = tmp_path / "crop_cover.txt"

    assert not crop_cover_path.exists()