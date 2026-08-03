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