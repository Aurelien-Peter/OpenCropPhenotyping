from pathlib import Path

import typer

from opencropphenotyping.batch_processing import process_batch
from opencropphenotyping.pipeline import export_results, process_sentinel2

app = typer.Typer()


@app.command()
def process(
    input_dir: Path = typer.Option(..., help="Path to the Sentinel-2 product."),
    output_dir: Path = typer.Option(..., help="Directory where results will be saved."),
    processed_dir: Path = typer.Option(..., help="Directory for processed files."),
    indices: list[str] = typer.Option(
        ["ndvi"],
        help="Vegetation indices to compute.",
    ),
    ndvi_threshold: float = typer.Option(
        0.3,
        help="NDVI threshold used for vegetation masking.",
    ),
    resolution: int = typer.Option(
        10,
        help="Target spatial resolution in meters.",
    ),
):
    """Process a single Sentinel-2 product."""

    try:
        result = process_sentinel2(
            input_dir=input_dir,
            output_dir=processed_dir,
            indices=indices,
            ndvi_threshold=ndvi_threshold,
            resolution=resolution,
        )

        export_results(
            result,
            output_dir,
        )
    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Processing completed successfully.")

@app.command()
def batch(
    input_dir: Path = typer.Option(..., help="Path to the directory containing Sentinel-2 products."),
    output_dir: Path = typer.Option(..., help="Directory where results will be saved."),
    processed_dir: Path = typer.Option(..., help="Directory for processed files."),
    indices: list[str] = typer.Option(
        ["ndvi"],
        help="Vegetation indices to compute.",
    ),
    ndvi_threshold: float = typer.Option(
        0.3,
        help="NDVI threshold used for vegetation masking.",
    ),
    resolution: int = typer.Option(
        10,
        help="Target spatial resolution in meters.",
    ),
):
    """Process multiple Sentinel-2 products from a directory."""
    
    try:
        result = process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
        indices=indices,
        ndvi_threshold=ndvi_threshold,
        resolution=resolution,
    )
    except Exception as e:
        typer.echo(f"Error during processing: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Batch processing completed successfully.")

if __name__ == "__main__":
    app()