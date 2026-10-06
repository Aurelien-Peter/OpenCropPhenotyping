import pytest
from typer.testing import CliRunner

from opencropphenotyping.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "process" in result.output
    assert "batch" in result.output
    assert "segment-uav" in result.output
    assert "batch-uav" in result.output

def test_process_cli(tmp_path, project_root):
    input_dir = project_root / "data" / "raw" / "toy_datasets" / "toy_dataset_1"
    output_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"

    result = runner.invoke(
        app,
        [
            "process",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--processed-dir",
            str(processed_dir),
            "--indices",
            "ndvi",
            "--ndvi-threshold",
            "0.3",
            "--resolution",
            "10",
        ],
    )

    print("exit code:", result.exit_code)
    print("exception:", repr(result.exception))
    print("output:", repr(result.output))

    assert result.exit_code == 0, (
        f"CLI failed with exception: {result.exception!r}\n"
        f"Output:\n{result.output}"
    )
    assert "Processing completed successfully." in result.output
    assert (output_dir / "indices").exists()
    assert (output_dir / "indices" / "ndvi.tif").exists()
    assert (output_dir / "crop_cover.txt").exists()
    assert (output_dir / "statistics.csv").exists()
    assert (output_dir / "vegetation_mask.tif").exists()

def test_batch_cli(tmp_path, project_root):
    input_dir = (
        project_root
        / "data"
        / "raw"
        / "toy_datasets"
    )

    output_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--processed-dir",
            str(processed_dir),
            "--indices",
            "ndvi",
            "--ndvi-threshold",
            "0.3",
            "--resolution",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "toy_dataset_1").exists()
    assert (output_dir / "toy_dataset_2").exists()
    assert (
        output_dir
        / "toy_dataset_1"
        / "indices"
        / "ndvi.tif"
    ).exists()

    assert (
        output_dir
        / "toy_dataset_2"
        / "indices"
        / "ndvi.tif"
    ).exists()

def test_process_cli_invalid_input_dir(tmp_path):
    output_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"
    input_dir = tmp_path / "does_not_exist"

    result = runner.invoke(
        app,
        [
            "process",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--processed-dir",
            str(processed_dir),
            "--indices",
            "ndvi",
            "--ndvi-threshold",
            "0.3",
            "--resolution",
            "10",
        ],
    )

    assert result.exit_code == 1
    assert "Error during processing" in result.output

def test_batch_cli_invalid_input_dir(tmp_path):
    output_dir = tmp_path / "results"
    processed_dir = tmp_path / "processed"
    input_dir = tmp_path / "does_not_exist"

    result = runner.invoke(
        app,
        [
            "batch",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--processed-dir",
            str(processed_dir),
            "--indices",
            "ndvi",
            "--ndvi-threshold",
            "0.3",
            "--resolution",
            "10",
        ],
    )

    assert result.exit_code == 1
    assert "Error during processing" in result.output

@pytest.mark.slow
def test_segment_uav_cli(tmp_path, project_root):
    dataset_dir = (
        project_root
        / "data"
        / "raw"
        / "toy_datasets"
        / "toy_segmentation"
        / "batch"
        / "plot_103_DSC01167"
    )

    image_path = next(
        path
        for path in dataset_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    geotiff_path = next(
        path
        for path in dataset_dir.iterdir()
        if path.suffix.lower() in {".tif", ".tiff"}
    )

    geopackage_path = next(
        path
        for path in dataset_dir.iterdir()
        if path.suffix.lower() == ".gpkg"
    )

    output_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "segment-uav",
            "--image",
            str(image_path),
            "--geotiff",
            str(geotiff_path),
            "--geopackage",
            str(geopackage_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "UAV segmentation completed successfully." in result.output

    assert (output_dir / "plants.csv").exists()
    assert (output_dir / "row_positions.csv").exists()
    assert (output_dir / "row_boundaries.csv").exists()

@pytest.mark.slow
def test_segment_uav_cli_with_options(tmp_path, project_root):
    dataset_dir = (
        project_root
        / "data"
        / "raw"
        / "toy_datasets"
        / "toy_segmentation"
        / "batch"
        / "plot_103_DSC01167"
    )

    image_path = next(
        path
        for path in dataset_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    geotiff_path = next(
        path
        for path in dataset_dir.iterdir()
        if path.suffix.lower() in {".tif", ".tiff"}
    )

    geopackage_path = next(
        path
        for path in dataset_dir.iterdir()
        if path.suffix.lower() == ".gpkg"
    )

    output_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "segment-uav",
            "--image",
            str(image_path),
            "--geotiff",
            str(geotiff_path),
            "--geopackage",
            str(geopackage_path),
            "--output-dir",
            str(output_dir),
            "--best-angle",
            "-11",
            "--threshold",
            "25",
            "--vegetation-fraction-threshold",
            "0.04",
            "--export-intermediate",
        ],
    )

    assert result.exit_code == 0
    assert "UAV segmentation completed successfully." in result.output
    assert "Best angle: -11.0°" in result.output

    assert (output_dir / "plants.csv").exists()
    assert (output_dir / "row_positions.csv").exists()
    assert (output_dir / "row_boundaries.csv").exists()
    assert (output_dir / "rotated_image.png").exists()
    assert (output_dir / "vegetation_mask.png").exists()

@pytest.mark.slow
def test_batch_uav_cli(tmp_path, project_root):
    input_dir = (
        project_root
        / "data"
        / "raw"
        / "toy_datasets"
        / "toy_segmentation"
        / "batch_two"
    )

    output_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "batch-uav",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--threshold",
            "25",
            "--vegetation-fraction-threshold",
            "0.04",
        ],
    )

    assert result.exit_code == 0
    assert "UAV batch processing completed successfully." in result.output

    assert (output_dir / "plot_103_DSC01167").exists()
    assert (output_dir / "plot_104_DSC09694").exists()

    assert (
        output_dir
        / "plot_103_DSC01167"
        / "plants.csv"
    ).exists()

    assert (
        output_dir
        / "plot_104_DSC09694"
        / "plants.csv"
    ).exists()

def test_segment_uav_cli_invalid_image(tmp_path):
    output_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "segment-uav",
            "--image",
            str(tmp_path / "does_not_exist.png"),
            "--geotiff",
            str(tmp_path / "does_not_exist.tif"),
            "--geopackage",
            str(tmp_path / "does_not_exist.gpkg"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Error during UAV segmentation" in result.output

def test_batch_uav_cli_invalid_input_dir(tmp_path):
    output_dir = tmp_path / "results"
    input_dir = tmp_path / "does_not_exist"

    result = runner.invoke(
        app,
        [
            "batch-uav",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1
    assert "Error during UAV batch processing" in result.output