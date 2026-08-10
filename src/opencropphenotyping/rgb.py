from pathlib import Path
from PIL import Image
import numpy as np

def read_rgb_image(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an RGB image and return its red, green and blue channels.

    Parameters
    ----------
    image_path : Path
        Path to the RGB image to read.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Three NumPy arrays containing the red, green and blue channels,
        respectively. Each array has the same height and width as the
        input image.

    Raises
    ------
    TypeError
        If the input image is not an RGB image.
    """

    image = Image.open(image_path)

    if image.mode != "RGB":
        raise TypeError("Image must be an RGB image.")

    red, green, blue = image.split()

    red = np.asarray(red)
    green = np.asarray(green)
    blue = np.asarray(blue)

    return red, green, blue