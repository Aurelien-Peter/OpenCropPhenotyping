import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from opencropphenotyping.coco import (
    build_image_annotation_index,
    build_plant_annotations_dataframe,
    get_annotation_centers,
    load_coco_annotations,
    get_image_annotations,
)


def test_load_coco_annotations(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first_data = {
        "images": [],
        "annotations": [],
    }

    second_data = {
        "images": [],
        "annotations": [],
    }

    with open(first_path, "w", encoding="utf-8") as file:
        json.dump(first_data, file)

    with open(second_path, "w", encoding="utf-8") as file:
        json.dump(second_data, file)

    result = load_coco_annotations(
        [first_path, second_path]
    )

    assert len(result) == 2

    assert result["first.json"] == first_data
    assert result["second.json"] == second_data

def test_build_image_annotation_index():
    coco_datasets = {
        "dataset.json": {
            "images": [
                {
                    "id": 1,
                    "file_name": "image_1.png",
                },
                {
                    "id": 2,
                    "file_name": "image_2.png",
                },
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [50, 60, 20, 10],
                },
                {
                    "id": 3,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                },
            ],
        }
    }

    result = build_image_annotation_index(
        coco_datasets
    )

    assert set(result.keys()) == {
        "image_1.png",
        "image_2.png",
    }

    assert result["image_1.png"]["image"]["id"] == 1
    assert result["image_2.png"]["image"]["id"] == 2

    assert len(
        result["image_1.png"]["annotations"]
    ) == 2

    assert len(
        result["image_2.png"]["annotations"]
    ) == 1

    assert result["image_1.png"]["annotations"][0]["id"] == 1
    assert result["image_1.png"]["annotations"][1]["id"] == 2
    assert result["image_2.png"]["annotations"][0]["id"] == 3

def test_build_image_annotation_index_with_no_annotations():
    coco_datasets = {
        "dataset.json": {
            "images": [
                {
                    "id": 1,
                    "file_name": "image_1.png",
                },
            ],
            "annotations": [],
        }
    }

    result = build_image_annotation_index(
        coco_datasets
    )

    assert "image_1.png" in result

    assert result["image_1.png"]["annotations"] == []

def test_build_plant_annotations_dataframe():
    image_annotations = {
        "image.png": {
            "image": {
                "id": 1,
            },
            "annotations": [
                {
                    "category_id": 1,
                    "bbox": [10, 20, 20, 10],
                },
                {
                    "category_id": 1,
                    "bbox": [100, 50, 40, 30],
                },
            ],
        }
    }

    result = build_plant_annotations_dataframe(
        image_annotations
    )

    assert len(result) == 2

    assert list(result["image_name"]) == [
        "image.png",
        "image.png",
    ]

    assert list(result["image_id"]) == [
        1,
        1,
    ]

    assert result.loc[0, "bbox_x"] == 10
    assert result.loc[0, "bbox_y"] == 20
    assert result.loc[0, "bbox_width"] == 20
    assert result.loc[0, "bbox_height"] == 10

    assert result.loc[0, "center_x"] == pytest.approx(
        20.0
    )

    assert result.loc[0, "center_y"] == pytest.approx(
        25.0
    )

    assert result.loc[1, "center_x"] == pytest.approx(
        120.0
    )

    assert result.loc[1, "center_y"] == pytest.approx(
        65.0
    )

def test_build_plant_annotations_dataframe_empty():
    image_annotations = {
        "image.png": {
            "image": {
                "id": 1,
            },
            "annotations": [],
        }
    }

    result = build_plant_annotations_dataframe(
        image_annotations
    )

    assert result.empty

    assert list(result.columns) == [
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

def test_get_image_annotations():
    plant_annotations = pd.DataFrame({
        "image_id": [1, 1, 2],
        "image_name": [
            "image_1.png",
            "image_1.png",
            "image_2.png",
        ],
        "category_id": [1, 1, 1],
        "center_x": [10.0, 20.0, 30.0],
        "center_y": [15.0, 25.0, 35.0],
    })

    image_path = Path("image_1.png")

    result = get_image_annotations(
        plant_annotations=plant_annotations,
        image_path=image_path,
    )

    assert len(result) == 2
    assert list(result["image_name"]) == [
        "image_1.png",
        "image_1.png",
    ]
    assert list(result["center_x"]) == [
        10.0,
        20.0,
    ]

def test_get_image_annotations_empty():
    plant_annotations = pd.DataFrame({
        "image_id": [1],
        "image_name": ["image_1.png"],
        "category_id": [1],
        "center_x": [10.0],
        "center_y": [15.0],
    })

    result = get_image_annotations(
        plant_annotations=plant_annotations,
        image_path=Path("image_2.png"),
    )

    assert result.empty
    
def test_get_annotation_centers():
    annotations = pd.DataFrame(
        {
            "center_x": [10.0, 30.5],
            "center_y": [20.0, 40.5],
        }
    )

    result = get_annotation_centers(
        annotations
    )

    expected = np.array([
        [10.0, 20.0],
        [30.5, 40.5],
    ])

    np.testing.assert_array_equal(
        result,
        expected,
    )

def test_get_annotation_centers_empty():
    annotations = pd.DataFrame(
        columns=["center_x", "center_y"]
    )

    result = get_annotation_centers(
        annotations
    )

    assert result.shape == (0, 2)
    assert result.dtype == float