from pathlib import Path

from opencropphenotyping.pipeline import export_results, process_sentinel2, process_uav_segmentation, export_uav_results


def process_batch(
    input_dir: Path,
    output_dir: Path,
    processed_dir: Path,
    indices: list[str] | None = None,
    ndvi_threshold: float = 0.3,
    resolution: int = 10,
) -> None:
    product_dirs = [
        path for path in input_dir.iterdir()
        if path.is_dir()
    ]

    for product_dir in product_dirs:
        product_output_dir = output_dir / product_dir.name
        product_processed_dir = processed_dir / product_dir.name

        print(f"Processing product: {product_dir.name}")

        try:
            result_product = process_sentinel2(
                input_dir=product_dir,
                output_dir=product_processed_dir,
                indices=indices,
                ndvi_threshold=ndvi_threshold,
                resolution=resolution,
            )

            export_results(
                result_product,
                product_output_dir,
            )

            print(f"Successfully processed: {product_dir.name}")

        except Exception as error:
            print(
                f"Error processing product '{product_dir.name}': {error}"
            )

def process_uav_batch(
    input_dir: Path,
    output_dir: Path,
    threshold: float = 25,
    vegetation_fraction_threshold: float = 0.04,
    export_intermediate: bool = False,
) -> None:
    """Process multiple UAV image datasets from a parent directory."""

    dataset_dirs = [
        path for path in input_dir.iterdir()
        if path.is_dir()
    ]

    for dataset_dir in dataset_dirs:
        dataset_output_dir = output_dir / dataset_dir.name

        image_files = [
            path for path in dataset_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and "georef" not in path.stem.lower()
        ]

        geotiff_files = [
            path for path in dataset_dir.iterdir()
            if path.suffix.lower() in {".tif", ".tiff"}
        ]

        geopackage_files = [
            path for path in dataset_dir.iterdir()
            if path.suffix.lower() == ".gpkg"
        ]

        if len(image_files) != 1:
            print(
                f"Error processing '{dataset_dir.name}': "
                f"expected 1 RGB image, found {len(image_files)}."
            )
            continue

        if len(geotiff_files) != 1:
            print(
                f"Error processing '{dataset_dir.name}': "
                f"expected 1 GeoTIFF, found {len(geotiff_files)}."
            )
            continue

        if len(geopackage_files) != 1:
            print(
                f"Error processing '{dataset_dir.name}': "
                f"expected 1 GeoPackage, found {len(geopackage_files)}."
            )
            continue

        image_path = image_files[0]
        geotiff_path = geotiff_files[0]
        geopackage_path = geopackage_files[0]

        print(f"Processing UAV dataset: {dataset_dir.name}")

        try:
            result = process_uav_segmentation(
                image_path=image_path,
                geotiff_path=geotiff_path,
                geopackage_path=geopackage_path,
                threshold=threshold,
                vegetation_fraction_threshold=vegetation_fraction_threshold,
                export_intermediate=export_intermediate,
            )

            export_uav_results(
                result,
                dataset_output_dir,
                export_intermediate=export_intermediate
            )

            print(f"Successfully processed: {dataset_dir.name}")

        except Exception as error:
            print(
                f"Error processing UAV dataset "
                f"'{dataset_dir.name}': {error}"
            )