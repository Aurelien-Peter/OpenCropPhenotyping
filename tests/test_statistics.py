import numpy as np
from opencropphenotyping.statistics import compute_statistics

def test_compute_statistics():
    image = np.array([[1, 2], [3, 4]], dtype=np.float32)

    results = compute_statistics(image)

    assert results["min"] == 1
    assert results["max"] == 4
    assert results["mean"] == 2.5
    assert results["median"] == 2.5
    assert results["var"] == 1.25
    assert np.isclose(results["std"], np.nanstd(image))
    assert results["p5"] == 1.15
    assert results["p25"] == 1.75
    assert results["p75"] == 3.25
    assert results["p95"] == 3.85

def test_compute_statistics_with_nan():
    image = np.array([[1, 2], [3, np.nan]], dtype=np.float32)

    results = compute_statistics(image)

    assert results["min"] == 1
    assert results["max"] == 3
    assert results["mean"] == 2
    assert results["median"] == 2
    assert np.isclose(results["var"], np.nanvar(image))
    assert np.isclose(results["std"], np.nanstd(image))
    assert results["p5"] == 1.1
    assert results["p25"] == 1.5
    assert results["p75"] == 2.5
    assert results["p95"] == 2.9

def test_compute_statistics_constant():
    image = np.full((100, 100), 5, dtype=np.float32)

    results = compute_statistics(image)

    assert results["min"] == 5
    assert results["max"] == 5
    assert results["mean"] == 5
    assert results["median"] == 5
    assert results["var"] == 0
    assert results["std"] == 0
    assert results["p5"] == 5
    assert results["p25"] == 5
    assert results["p75"] == 5
    assert results["p95"] == 5