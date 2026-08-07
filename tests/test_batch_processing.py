from opencropphenotyping.batch_processing import process_batch


def test_process_batch(monkeypatch, tmp_path):
    # Create a temporary input directory with dummy product directories
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    product_1 = input_dir / "product_1"
    product_1.mkdir()
    product_2 = input_dir / "product_2"
    product_2.mkdir()

    # Create a temporary output and processed directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    processed_products = []

    def mock_process_sentinel2(
        input_dir,
        output_dir,
        indices,
        ndvi_threshold,
        resolution,
    ):
        processed_products.append({
            "input_dir": input_dir,
            "output_dir": output_dir,
            "indices": indices,
            "ndvi_threshold": ndvi_threshold,
            "resolution": resolution,
        })

        return f"result_{input_dir.name}"

    monkeypatch.setattr(
        "opencropphenotyping.batch_processing.process_sentinel2",
        mock_process_sentinel2,
    )

    exported_results = []

    def mock_export_results(result, output_dir):
        exported_results.append({
            "result": result,
            "output_dir": output_dir,
        })

    monkeypatch.setattr(
        "opencropphenotyping.batch_processing.export_results",
        mock_export_results,
    )

    # Call the process_batch function
    process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
        indices=["NDVI"],
        ndvi_threshold=0.3,
        resolution=10
    )

    # Check that both products were processed correctly
    assert len(processed_products) == 2
    assert processed_products[0]["input_dir"] == product_1
    assert processed_products[1]["input_dir"] == product_2
    assert processed_products[0]["output_dir"] == (
        processed_dir / "product_1"
    )

    assert processed_products[1]["output_dir"] == (
        processed_dir / "product_2"
    )
    assert processed_products[0]["indices"] == ["NDVI"]
    assert processed_products[0]["ndvi_threshold"] == 0.3
    assert processed_products[0]["resolution"] == 10

    # Check that the results were exported correctly
    assert len(exported_results) == 2
    assert exported_results[0]["result"] == "result_product_1"
    assert exported_results[1]["result"] == "result_product_2"
    assert exported_results[0]["output_dir"] == output_dir/ "product_1"
    assert exported_results[1]["output_dir"] == output_dir/ "product_2"

def test_process_batch_empty_input(monkeypatch, tmp_path):

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"

    input_dir.mkdir()

    process_calls = []
    export_calls = []

    def mock_process_sentinel2(*args, **kwargs):
        process_calls.append((args, kwargs))

    def mock_export_results(*args, **kwargs):
        export_calls.append((args, kwargs))

    monkeypatch.setattr(
        "opencropphenotyping.batch_processing.process_sentinel2",
        mock_process_sentinel2,
    )

    monkeypatch.setattr(
        "opencropphenotyping.batch_processing.export_results",
        mock_export_results,
    )

    process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
    )

    assert process_calls == []
    assert export_calls == []

def test_process_batch_with_toy_datasets(tmp_path, project_root):

    input_dir = project_root / "data" / "raw" / "toy_datasets"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
        indices=["ndvi", "gndvi"],
        ndvi_threshold=0.3,
        resolution=10
    )

    # Check that both products were processed correctly
    assert (processed_dir / "toy_dataset_1").exists()
    assert (
        processed_dir
        / "toy_dataset_1"
        / "Resampled_B05_from_R20m_to_R10m.tif"
    ).exists()
    assert (processed_dir / "toy_dataset_2").exists()

    assert (output_dir / "toy_dataset_1").exists()
    assert (output_dir / "toy_dataset_1" / "indices").exists()
    assert (output_dir / "toy_dataset_1" / "crop_cover.txt").exists()
    assert (output_dir / "toy_dataset_1" / "statistics.csv").exists()
    assert (output_dir / "toy_dataset_1" / "vegetation_mask.tif").exists()
    assert (output_dir / "toy_dataset_1" / "indices" / "ndvi.tif").exists()
    assert (output_dir / "toy_dataset_1" / "indices" / "gndvi.tif").exists()
    assert not (output_dir / "toy_dataset_1" / "indices" / "ndre.tif").exists()
    assert not (output_dir / "toy_dataset_1" / "indices" / "savi.tif").exists()

    assert (output_dir / "toy_dataset_2").exists()
    assert (output_dir / "toy_dataset_2" / "indices").exists()
    assert (output_dir / "toy_dataset_2" / "crop_cover.txt").exists()
    assert (output_dir / "toy_dataset_2" / "statistics.csv").exists()
    assert (output_dir / "toy_dataset_2" / "vegetation_mask.tif").exists()
    assert (output_dir / "toy_dataset_2" / "indices" / "ndvi.tif").exists()
    assert (output_dir / "toy_dataset_2" / "indices" / "gndvi.tif").exists()
    assert not (output_dir / "toy_dataset_2" / "indices" / "ndre.tif").exists()
    assert not (output_dir / "toy_dataset_2" / "indices" / "savi.tif").exists()

def test_process_batch_continues_after_product_error(
    tmp_path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"

    product_1 = input_dir / "toy_dataset_1"
    product_2 = input_dir / "toy_dataset_2"

    product_1.mkdir(parents=True)
    product_2.mkdir(parents=True)

    def mock_process_sentinel2(
        input_dir,
        output_dir,
        indices,
        ndvi_threshold,
        resolution,
    ):
        if input_dir.name == "toy_dataset_1":
            raise FileNotFoundError("B05 not found")

        return "fake_result"

    def mock_export_results(result, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "opencropphenotyping.batch_processing.process_sentinel2",
        mock_process_sentinel2,
    )

    monkeypatch.setattr(
        "opencropphenotyping.batch_processing.export_results",
        mock_export_results,
    )

    process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
        indices=["ndvi"],
    )

    assert not (output_dir / "toy_dataset_1").exists()
    assert (output_dir / "toy_dataset_2").exists()