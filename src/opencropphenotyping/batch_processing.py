from pathlib import Path

from opencropphenotyping.pipeline import export_results, process_sentinel2


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