import numpy as np

def threshold_vegetation_index(
    exg: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Create a vegetation mask from an ExG image."""
    return ~np.isnan(exg) & (exg > threshold)