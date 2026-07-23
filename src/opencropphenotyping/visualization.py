import matplotlib.pyplot as plt
import numpy as np

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
    plt.axis('off')
    plt.tight_layout()
    plt.show()