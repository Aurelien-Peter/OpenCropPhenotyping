import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure

from opencropphenotyping.visualization import display_raster, plot_statistics


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

def test_plot_statistics_empty_statistics():
    # Create a simple 2D array
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)
    statistics = {}

    # Call the plot_statistics function
    figures = plot_statistics(image, statistics, index_name="Test Index")

    # Check that the function returns a list of figures
    assert isinstance(figures, list)
    assert len(figures) == 2
    assert all(isinstance(fig, Figure) for fig in figures)

def test_plot_statistics_handles_nan():
    image = np.array(
        [[1, 2], [3, np.nan]],
        dtype=np.float32
    )

    statistics = {
        "mean": 2.0,
        "median": 2.0,
    }

    figures = plot_statistics(
        image=image,
        statistics=statistics,
        index_name="NDVI"
    )

    assert len(figures) == 2

def test_plot_statistics_titles():
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)

    statistics = {
        "mean": 2.5,
        "median": 2.5,
    }

    figures = plot_statistics(
        image=image,
        statistics=statistics,
        index_name="NDVI"
    )

    assert figures[0].axes[0].get_title() == "NDVI - Distribution of values"
    assert figures[1].axes[0].get_title() == "NDVI - Boxplot of values"

def test_plot_statistics_adds_mean_and_median_lines():
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)

    statistics = {
        "mean": 2.5,
        "median": 2.5,
    }

    figures = plot_statistics(
        image=image,
        statistics=statistics,
        index_name="NDVI"
    )

    ax = figures[0].axes[0]

    assert len(ax.lines) == 2
    assert np.allclose(ax.lines[0].get_xdata(), [2.5, 2.5])
    assert np.allclose(ax.lines[1].get_xdata(), [2.5, 2.5])