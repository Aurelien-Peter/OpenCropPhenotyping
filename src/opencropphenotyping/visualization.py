import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def display_raster(
    image: np.ndarray,
    cmap: str = "RdYlGn",
    title: str | None = None,
) -> None:
    """
    Display a raster image using matplotlib.

    Parameters
    ----------
    image : np.ndarray
        Raster values to display.
    cmap : str, optional
        Colormap to use for displaying the image. Default is "RdYlGn".
    title : str, optional
        Title for the plot. Default is None.
    """
    image = np.asarray(image)

    # Raise an error if the input image is not 2D
    if image.ndim != 2:
        raise ValueError("Raster image must be a 2D array.")

    # Plot the image
    img = plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.colorbar(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_statistics(
        image : np.ndarray,
        statistics : dict[str, float],
        index_name: str = ""
    ) -> list[Figure]:
    """
    Create plots showing the distribution of pixel values in an image.

    Parameters
    ----------
    image : np.ndarray
        Raster image containing the pixel values to visualize.
    statistics : dict[str, float]
        Statistics computed from the image, such as mean and median.
    index_name : str, optional
        Name of the vegetation index or variable represented by the image.

    Returns
    -------
    list[plt.Figure]
        List containing the generated Matplotlib figures.
    """
    figures = []

    # Remove NaN values before plotting
    values = image[~np.isnan(image)]

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(values, bins="auto")

    ax.set_title(f"{index_name} - Distribution of values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Number of pixels")

    if "mean" in statistics:
        ax.axvline(
            statistics["mean"],
            linestyle="solid",
            label="Mean"
        )

    if "median" in statistics:
        ax.axvline(
            statistics["median"],
            linestyle="--",
            label="Median"
        )

    ax.legend()
    figures.append(fig)

    # Boxplot
    fig, ax = plt.subplots()
    ax.boxplot(values)

    ax.set_title(f"{index_name} - Boxplot of values")
    ax.set_ylabel("Value")

    figures.append(fig)

    return figures

def plot_crop_row_profiles(
    row_profile: np.ndarray,
    peaks: np.ndarray | None = None,
    boundaries: np.ndarray | None = None,
    title: str = "Crop-row vegetation profile",
) -> None:
    """
    Display the vegetation profile used for crop-row detection.

    Parameters
    ----------
    row_profile : np.ndarray
        One-dimensional vegetation profile along the vertical image axis.
    peaks : np.ndarray, optional
        Detected crop-row positions.
    boundaries : np.ndarray, optional
        Crop-row boundaries.
    title : str, default="Crop-row vegetation profile"
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(row_profile)

    if peaks is not None:
        ax.scatter(
            peaks,
            row_profile[peaks],
            s=40,
            label="Detected rows",
        )

    if boundaries is not None:
        for boundary in boundaries:
            ax.axvline(
                boundary,
                linestyle="--",
                linewidth=1,
            )

    ax.set_xlabel("Y position (pixels)")
    ax.set_ylabel("Vegetation pixels")
    ax.set_title(title)
    ax.grid()
    ax.legend()

    plt.show()

def plot_row_masks(
    row_masks: list[np.ndarray],
    title: str = "Vegetation masks by crop row",
) -> None:
    """
    Display vegetation masks for all crop-row regions.

    Parameters
    ----------
    row_masks : list[np.ndarray]
        Binary vegetation masks, one per crop row.
    title : str, default="Vegetation masks by crop row"
        Figure title.
    """
    fig, axes = plt.subplots(
        len(row_masks),
        1,
        figsize=(16, 3 * len(row_masks)),
    )

    if len(row_masks) == 1:
        axes = [axes]

    for i, (ax, row_mask) in enumerate(
        zip(axes, row_masks),
        start=1,
    ):
        ax.imshow(
            row_mask,
            cmap="gray",
            aspect="auto",
        )
        ax.set_title(f"Row {i}")
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_rgb_with_detected_plants(
    image: np.ndarray,
    plant_df: pd.DataFrame,
    title: str = "RGB image and detected plants",
) -> None:
    """
    Display an RGB image with detected plant positions.

    Parameters
    ----------
    image : np.ndarray
        RGB image in the coordinate system used by ``plant_df``.
    plant_df : pd.DataFrame
        Plant detection results containing ``centroid_x``,
        ``centroid_y`` and ``missing_candidate``.
    title : str, default="RGB image and detected plants"
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.imshow(image)

    detected = plant_df[
        ~plant_df["missing_candidate"]
    ].dropna(
        subset=["centroid_x", "centroid_y"]
    )

    ax.scatter(
        detected["centroid_x"],
        detected["centroid_y"],
        s=30,
        label="Detected plants",
        color="red",
    )

    missing = plant_df[
        plant_df["missing_candidate"]
    ]

    ax.scatter(
        missing["expected_x"],
        missing["centroid_y"],
        s=40,
        marker="x",
        label="Missing candidates",
    )

    ax.set_title(title)
    ax.legend()
    ax.axis("off")

    plt.show()

def plot_exg_with_detected_plants(
    exg: np.ndarray,
    plant_df: pd.DataFrame,
    title: str = "ExG and detected plants",
) -> None:
    """
    Display an ExG image with detected plant positions.

    Parameters
    ----------
    exg : np.ndarray
        ExG image in the coordinate system used by ``plant_df``.
    plant_df : pd.DataFrame
        Plant detection results containing ``centroid_x``,
        ``centroid_y`` and ``missing_candidate``.
    title : str, default="ExG and detected plants"
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.imshow(exg)

    detected = plant_df[
        ~plant_df["missing_candidate"]
    ].dropna(
        subset=["centroid_x", "centroid_y"]
    )

    ax.scatter(
        detected["centroid_x"],
        detected["centroid_y"],
        s=30,
        color="red",
        label="Detected plants",
    )

    missing = plant_df[
        plant_df["missing_candidate"]
    ]

    missing = missing.dropna(
        subset=["expected_x"]
    )

    ax.scatter(
        missing["expected_x"],
        missing["centroid_y"],
        s=40,
        marker="x",
        label="Missing candidates",
    )

    ax.set_title(title)
    ax.legend()
    ax.axis("off")

    plt.show()

def plot_rgb_with_annotations(
    image: np.ndarray,
    annotations: list[dict],
    title: str = "RGB image with COCO annotations",
) -> None:
    """
    Display an RGB image with COCO annotation bounding boxes centers.

    Parameters
    ----------
    image : np.ndarray
        RGB image to display.
    annotations : list[dict]
        COCO annotations containing bounding boxes centers in the format
        ``[center_x, center_y]``.
    title : str, default="RGB image with COCO annotations"
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.imshow(image)

    for annotation in annotations:
        ax.scatter(
                annotation["center_x"],
                annotation["center_y"],
                s=20,
            )

    ax.set_title(
        f"{title} — {len(annotations)} plants"
    )

    ax.axis("off")

    plt.show()