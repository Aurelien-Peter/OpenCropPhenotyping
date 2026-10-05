from pathlib import Path

import typer

from opencropphenotyping.batch_processing import process_batch, process_uav_batch
from opencropphenotyping.pipeline import export_results, process_sentinel2, process_uav_segmentation, export_uav_results

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
        process_batch(
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

@app.command()
def segment_uav(
    image: Path = typer.Option(
        ...,
        help="Path to the UAV RGB image.",
    ),
    geotiff: Path = typer.Option(
        ...,
        help="Path to the georeferenced GeoTIFF.",
    ),
    geopackage: Path = typer.Option(
        ...,
        help="Path to the experimental plot GeoPackage.",
    ),
    output_dir: Path = typer.Option(
        ...,
        help="Directory where UAV segmentation results will be saved.",
    ),
    best_angle: float | None = typer.Option(
        None,
        help="Crop-row orientation angle. If omitted, it is estimated automatically.",
    ),
    threshold: float = typer.Option(
        25.0,
        help="ExG threshold used for vegetation segmentation.",
    ),
    vegetation_fraction_threshold: float = typer.Option(
        0.04,
        help="Vegetation fraction threshold used to classify planting positions.",
    ),
    export_intermediate: bool = typer.Option(
        False,
        help="If True, all intermediate results will be exported.",
    )
):
    """Process a UAV RGB image and detect crop plants."""

    try:
        result = process_uav_segmentation(
            image_path=image,
            geotiff_path=geotiff,
            geopackage_path=geopackage,
            best_angle=best_angle,
            threshold=threshold,
            vegetation_fraction_threshold=vegetation_fraction_threshold,
            export_intermediate=export_intermediate,
        )

        export_uav_results(
            result,
            output_dir,
            export_intermediate=export_intermediate
        )

    except Exception as e:
        typer.echo(
            f"Error during UAV segmentation: {e}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        "UAV segmentation completed successfully."
    )

    if(export_intermediate):
        typer.echo(
            f"Best angle: {result.best_angle:.1f}°"
        )

    typer.echo(
        f"Detected rows: {len(result.row_positions)}"
    )

    typer.echo(
        f"Planting positions: {len(result.plants)}"
    )

    typer.echo(
        "Occupied positions: "
        f"{(~result.plants['missing_candidate']).sum()}"
    )

    typer.echo(
        "Potentially missing positions: "
        f"{result.plants['missing_candidate'].sum()}"
    )

@app.command()
def batch_uav(
    input_dir: Path = typer.Option(
        ...,
        help="Parent directory containing UAV datasets.",
    ),
    output_dir: Path = typer.Option(
        ...,
        help="Directory where UAV results will be saved.",
    ),
    threshold: float = typer.Option(
        25,
        help="ExG threshold used for vegetation segmentation.",
    ),
    vegetation_fraction_threshold: float = typer.Option(
        0.04,
        help="Threshold used to classify missing plants.",
    ),
    export_intermediate: bool = typer.Option(
        False,
        help="If True, all intermediate results will be exported.",
    ),
):
    """Process multiple UAV datasets from a parent directory."""
    try:
        process_uav_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            threshold=threshold,
            vegetation_fraction_threshold=vegetation_fraction_threshold,
            export_intermediate=export_intermediate,
        )
    except Exception as error:
        typer.echo(
            f"Error during UAV batch processing: {error}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("UAV batch processing completed successfully.")

if __name__ == "__main__":
    app()