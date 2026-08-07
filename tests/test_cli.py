from opencropphenotyping.cli import app
from typer.testing import CliRunner

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "process" in result.output
    assert "batch" in result.output

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

    assert result.exit_code == 0
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