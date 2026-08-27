import json
from pathlib import Path
import numpy as np
import pandas as pd

def load_coco_annotations(
    json_paths: list[Path],
) -> dict[str, dict]:
    """
    Load COCO annotation files from JSON files.

    Parameters
    ----------
    json_paths : list[Path]
        Paths to COCO annotation JSON files.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping each JSON filename to its decoded
        annotation data.
    """
    annotations = {}

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            annotations[path.name] = json.load(f)

    return annotations

def build_image_annotation_index(
    coco_datasets: dict[str, dict],
) -> dict[str, dict]:
    """
    Build an index linking each image to its COCO annotations.

    Parameters
    ----------
    coco_datasets : dict[str, dict]
        COCO datasets loaded from JSON files.

    Returns
    -------
    dict[str, dict]
        Dictionary indexed by image filename. Each entry contains
        the image metadata under ``"image"`` and the corresponding
        annotations under ``"annotations"``.
    """
    image_annotations = {}

    for data in coco_datasets.values():

        for image in data["images"]:

            image_annotations[image["file_name"]] = {
                "image": image,
                "annotations": [
                    annotation
                    for annotation in data["annotations"]
                    if annotation["image_id"] == image["id"]
                ],
            }

    return image_annotations

def build_plant_annotations_dataframe(
    image_annotations: dict[str, dict],
) -> pd.DataFrame:
    """
    Convert indexed COCO plant annotations into a DataFrame.

    Parameters
    ----------
    image_annotations : dict[str, dict]
        Image annotation index produced by
        :func:`build_image_annotation_index`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per annotated plant.
        Bounding-box coordinates and plant centre coordinates are
        expressed in the original image coordinate system.
    """
    plant_annotations = []

    columns = [
        "image_id",
        "image_name",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "center_x",
        "center_y",
    ]

    for image_name, info in image_annotations.items():

        image_id = info["image"]["id"]

        for annotation in info["annotations"]:

            x, y, width, height = annotation["bbox"]

            plant_annotations.append(
                {
                    "image_id": image_id,
                    "image_name": image_name,
                    "category_id": annotation["category_id"],
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_width": width,
                    "bbox_height": height,
                    "center_x": x + width / 2,
                    "center_y": y + height / 2,
                }
            )

    return pd.DataFrame(plant_annotations, columns=columns)

def get_image_annotations(
    plant_annotations: pd.DataFrame,
    image_path: Path,
) -> pd.DataFrame:
    """
    Return plant annotations associated with a given image.

    Parameters
    ----------
    plant_annotations : pd.DataFrame
        DataFrame containing plant annotations for all images.
        Must contain an ``image_name`` column.
    image_path : Path
        Path to the image for which annotations are requested.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the annotations associated with
        ``image_path``.
    """
    return plant_annotations[
        plant_annotations["image_name"] == image_path.name
    ].copy()

def get_annotation_centers(
    plant_annotations: pd.DataFrame,
) -> np.ndarray:
    """
    Extract annotated plant centres as an array of x/y coordinates.

    Parameters
    ----------
    plant_annotations : pd.DataFrame
        Plant annotations containing ``center_x`` and ``center_y``.

    Returns
    -------
    np.ndarray
        Array of plant centres with shape ``(n_annotations, 2)``.
        Columns correspond to x and y coordinates.
    """
    if len(plant_annotations) == 0:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    return plant_annotations[
        ["center_x", "center_y"]
    ].to_numpy(dtype=float)