import numpy as np
import pandas as pd


def transform_original_points_to_rotated(
    points: np.ndarray,
    angle: float,
    original_shape: tuple[int, int],
    rotated_shape: tuple[int, int],
) -> np.ndarray:
    """
    Transform points from the original image coordinates to the
    rotated image coordinates.

    The transformation accounts for the different image centers
    before and after rotation. Coordinates are expected in
    ``(x, y)`` order, while image shapes are given in
    ``(height, width)`` order.

    Parameters
    ----------
    points : np.ndarray
        Array of point coordinates with shape ``(n_points, 2)``.
        Each point is represented as ``(x, y)``.
    angle : float
        Rotation angle in degrees. The rotation convention follows
        the coordinate transformation used for the image rotation.
    original_shape : tuple[int, int]
        Shape of the original image represented as
        ``(height, width)``.
    rotated_shape : tuple[int, int]
        Shape of the rotated image represented as
        ``(height, width)``.

    Returns
    -------
    np.ndarray
        Transformed point coordinates with shape ``(n_points, 2)``.
        Coordinates are represented as ``(x, y)``.
    """
    original_height, original_width = original_shape
    rotated_height, rotated_width = rotated_shape

    original_center = np.array([
        (original_width - 1) / 2,
        (original_height - 1) / 2,
    ])

    rotated_center = np.array([
        (rotated_width - 1) / 2,
        (rotated_height - 1) / 2,
    ])

    shifted_points = points - original_center

    theta = np.deg2rad(angle)

    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])

    rotated_points = (
        shifted_points @ rotation_matrix.T
        + rotated_center
    )

    return rotated_points

def match_annotations_to_plant_positions(
    plants_df: pd.DataFrame,
    annotation_points: np.ndarray,
    boundaries: np.ndarray,
) -> pd.DataFrame:
    """
    Match annotated plant centres to theoretical planting positions.

    Parameters
    ----------
    plants_df : pd.DataFrame
        Plant detection table containing ``row``, ``x_start`` and
        ``x_end`` columns.
    annotation_points : np.ndarray
        Annotated plant centres in rotated image coordinates.
        Expected shape is ``(n_annotations, 2)`` with columns
        ``x`` and ``y``.
    boundaries : np.ndarray
        Vertical boundaries defining the crop rows.

    Returns
    -------
    pd.DataFrame
        Copy of ``plants_df`` with the additional columns
        ``annotation_count`` and ``annotation_present``.
    """
    plants_df = plants_df.copy()

    annotation_count = []
    annotation_present = []

    for _, segment in plants_df.iterrows():

        row_index = int(segment["row"])

        x_start = segment["x_start"]
        x_end = segment["x_end"]

        y_start = boundaries[row_index - 1]
        y_end = boundaries[row_index]

        in_segment = (
            (annotation_points[:, 0] >= x_start)
            & (annotation_points[:, 0] < x_end)
            & (annotation_points[:, 1] >= y_start)
            & (annotation_points[:, 1] < y_end)
        )

        count = int(np.sum(in_segment))

        annotation_count.append(count)
        annotation_present.append(count > 0)

    plants_df["annotation_count"] = annotation_count
    plants_df["annotation_present"] = annotation_present

    return plants_df

def compute_confusion_matrix(
    predicted_occupied: pd.Series,
    actual_occupied: pd.Series,
) -> dict[str, int]:
    """
    Compute binary classification confusion matrix.

    Parameters
    ----------
    predicted_occupied : pd.Series
        Boolean predictions indicating whether a planting position
        is considered occupied.
    actual_occupied : pd.Series
        Boolean reference labels indicating whether an annotated plant
        is present.

    Returns
    -------
    dict[str, int]
        Counts of true positives (TP), false positives (FP),
        false negatives (FN), and true negatives (TN).
    """
    tp = int(
        (predicted_occupied & actual_occupied).sum()
    )

    fp = int(
        (predicted_occupied & ~actual_occupied).sum()
    )

    fn = int(
        (~predicted_occupied & actual_occupied).sum()
    )

    tn = int(
        (~predicted_occupied & ~actual_occupied).sum()
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

def compute_classification_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    """
    Compute precision, recall, and F1 score.

    Parameters
    ----------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.

    Returns
    -------
    dict[str, float]
        Precision, recall, and F1 score.
    """
    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def evaluate_plant_detection(
    plants_df: pd.DataFrame,
    annotation_points: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate plant detection against annotated plant positions.

    Parameters
    ----------
    plants_df : pd.DataFrame
        Plant detection results.
    annotation_points : np.ndarray
        Annotated plant centres in rotated image coordinates.
    boundaries : np.ndarray
        Vertical boundaries defining the crop rows.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Enriched plant detection table and evaluation metrics.
    """
    plants_df = match_annotations_to_plant_positions(
        plants_df=plants_df,
        annotation_points=annotation_points,
        boundaries=boundaries,
    )

    predicted_occupied = ~plants_df["missing_candidate"]
    actual_occupied = plants_df["annotation_present"]

    confusion = compute_confusion_matrix(
        predicted_occupied=predicted_occupied,
        actual_occupied=actual_occupied,
    )

    metrics = compute_classification_metrics(
        tp=confusion["tp"],
        fp=confusion["fp"],
        fn=confusion["fn"],
    )

    evaluation = {
        **confusion,
        **metrics,
    }

    return plants_df, evaluation

