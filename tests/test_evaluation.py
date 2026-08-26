import numpy as np
import pandas as pd
import pytest

from opencropphenotyping.evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix,
    evaluate_plant_detection,
    match_annotations_to_plant_positions,
    transform_original_points_to_rotated,
)


def test_compute_confusion_matrix():
    predicted_occupied = pd.Series([
        True,
        True,
        False,
        False,
    ])

    actual_occupied = pd.Series([
        True,
        False,
        True,
        False,
    ])

    result = compute_confusion_matrix(
        predicted_occupied=predicted_occupied,
        actual_occupied=actual_occupied,
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1

def test_compute_confusion_matrix_perfect_prediction():
    predicted_occupied = pd.Series([
        True,
        False,
        True,
        False,
    ])

    actual_occupied = pd.Series([
        True,
        False,
        True,
        False,
    ])

    result = compute_confusion_matrix(
        predicted_occupied=predicted_occupied,
        actual_occupied=actual_occupied,
    )

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0

def test_compute_classification_metrics():
    result = compute_classification_metrics(
        tp=1,
        fp=1,
        fn=1,
    )

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)

def test_compute_classification_metrics_no_positive_prediction():
    result = compute_classification_metrics(
        tp=0,
        fp=0,
        fn=2,
    )

    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["f1"] == 0

def test_compute_classification_metrics_no_actual_positive():
    result = compute_classification_metrics(
        tp=0,
        fp=2,
        fn=0,
    )

    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0

def test_evaluate_plant_detection(monkeypatch):

    plants_df = pd.DataFrame({
        "row": [1, 1, 1, 1],
        "x_start": [0, 10, 20, 30],
        "x_end": [10, 20, 30, 40],
        "missing_candidate": [
            False,
            True,
            False,
            True,
        ],
    })

    annotation_points = np.array([
        [5.0, 5.0],
        [25.0, 5.0],
    ])

    boundaries = np.array([
        0,
        10,
        20,
        30,
        40,
    ])

    def fake_match_annotations_to_plant_positions(
        plants_df,
        annotation_points,
        boundaries,
    ):
        result = plants_df.copy()

        result["annotation_count"] = [
            1,
            0,
            1,
            0,
        ]

        result["annotation_present"] = [
            True,
            False,
            True,
            False,
        ]

        return result

    monkeypatch.setattr(
        "opencropphenotyping.evaluation.match_annotations_to_plant_positions",
        fake_match_annotations_to_plant_positions,
    )

    result = evaluate_plant_detection(
        plants_df=plants_df,
        annotation_points=annotation_points,
        boundaries=boundaries,
    )

    assert result[1]["tp"] == 2
    assert result[1]["fp"] == 0
    assert result[1]["fn"] == 0

    assert result[1]["precision"] == pytest.approx(1.0)
    assert result[1]["recall"] == pytest.approx(1.0)
    assert result[1]["f1"] == pytest.approx(1.0)

def test_transform_original_points_to_rotated_zero_angle():
    points = np.array([
        [10.0, 20.0],
        [50.0, 40.0],
    ])

    result = transform_original_points_to_rotated(
        points=points,
        angle=0.0,
        original_shape=(100, 100),
        rotated_shape=(100, 100),
    )

    np.testing.assert_allclose(
        result,
        points,
    )

def test_transform_original_points_to_rotated_center():
    points = np.array([
        [49.5, 49.5],
    ])

    result = transform_original_points_to_rotated(
        points=points,
        angle=45.0,
        original_shape=(100, 100),
        rotated_shape=(100, 100),
    )

    np.testing.assert_allclose(
        result,
        points,
    )

def test_transform_original_points_to_rotated_90_degrees():
    points = np.array([
        [50.0, 40.0],
    ])

    result = transform_original_points_to_rotated(
        points=points,
        angle=90.0,
        original_shape=(100, 100),
        rotated_shape=(100, 100),
    )

    expected = np.array([
        [40.0, 49.0],
    ])

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-6,
    )

def test_match_annotations_to_plant_positions():
    plants_df = pd.DataFrame({
        "row": [1, 1, 1],
        "x_start": [0, 10, 20],
        "x_end": [10, 20, 30],
    })

    annotation_points = np.array([
        [5.0, 5.0],
        [15.0, 5.0],
        [25.0, 5.0],
    ])

    boundaries = np.array([
        0,
        10,
    ])

    result = match_annotations_to_plant_positions(
        plants_df=plants_df,
        annotation_points=annotation_points,
        boundaries=boundaries,
    )

    assert list(result["annotation_count"]) == [
        1,
        1,
        1,
    ]

    assert list(result["annotation_present"]) == [
        True,
        True,
        True,
    ]

def test_match_annotations_to_plant_positions_missing():
    plants_df = pd.DataFrame({
        "row": [1, 1],
        "x_start": [0, 10],
        "x_end": [10, 20],
    })

    annotation_points = np.array([
        [5.0, 5.0],
    ])

    boundaries = np.array([
        0,
        10,
    ])

    result = match_annotations_to_plant_positions(
        plants_df=plants_df,
        annotation_points=annotation_points,
        boundaries=boundaries,
    )

    assert list(result["annotation_count"]) == [
        1,
        0,
    ]

    assert list(result["annotation_present"]) == [
        True,
        False,
    ]

def test_match_annotations_multiple_annotations():
    plants_df = pd.DataFrame({
        "row": [1],
        "x_start": [0],
        "x_end": [10],
    })

    annotation_points = np.array([
        [2.0, 5.0],
        [5.0, 5.0],
        [8.0, 5.0],
    ])

    boundaries = np.array([
        0,
        10,
    ])

    result = match_annotations_to_plant_positions(
        plants_df=plants_df,
        annotation_points=annotation_points,
        boundaries=boundaries,
    )

    assert result["annotation_count"].iloc[0] == 3
    assert result["annotation_present"].iloc[0]

