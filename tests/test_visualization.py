import numpy as np
import pytest

from opencropphenotyping.visualization import display_raster


def test_display_raster_success():
    # Create a simple 2D array
    image = np.array([[1, 2], [3, 4]])

    # Call the display_raster function
    display_raster(image, cmap="viridis", title="Test Raster")

    # If no exceptions are raised, the test passes


def test_display_raster_non_2d():
    # Create a 1D array
    image = np.array([1, 2, 3, 4])

    # Expect a ValueError for non-2D input
    with pytest.raises(ValueError, match="Raster image must be a 2D array."):
        display_raster(image, cmap="", title="")


def test_display_raster_3d():
    # Create a 3D array
    image = np.random.rand(10, 10, 3)

    # Expect a ValueError for non-2D input
    with pytest.raises(ValueError, match="Raster image must be a 2D array."):
        display_raster(image)
