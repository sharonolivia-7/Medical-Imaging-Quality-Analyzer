"""Basic educational image-processing helpers."""

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_filter(image: np.ndarray) -> np.ndarray:
    """Smooth a 2D grayscale image with a Gaussian filter.

    The filter uses a standard deviation (sigma) of 1 pixel. The input is
    converted to floating point so integer images are not rounded during
    filtering.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError("image must be a 2D grayscale array.")
    if image.size == 0:
        raise ValueError("image must not be empty.")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        raise TypeError("image must contain real numeric pixel values.")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite pixel values.")

    # A Gaussian filter averages nearby pixels, with closer pixels receiving
    # more weight. This smooths small, high-frequency variations in the image.
    return gaussian_filter(image.astype(np.float64), sigma=1.0)
