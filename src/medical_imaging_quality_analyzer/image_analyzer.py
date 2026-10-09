"""Basic measurements for grayscale sample images."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


def analyze_image_array(image: np.ndarray) -> dict[str, int | float]:
    """Calculate basic statistics for a valid grayscale NumPy image array."""
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

    pixel_values = image.astype(np.float64)
    return {
        "width": int(pixel_values.shape[1]),
        "height": int(pixel_values.shape[0]),
        "minimum_pixel_value": float(np.min(pixel_values)),
        "maximum_pixel_value": float(np.max(pixel_values)),
        "mean_intensity": float(np.mean(pixel_values)),
        "standard_deviation": float(np.std(pixel_values)),
    }


def analyze_image(image_path: str | Path) -> dict[str, int | float]:
    """Load an image and return its dimensions and grayscale intensity statistics.

    Color images are converted to grayscale before their pixel values are measured.
    """
    try:
        with Image.open(image_path) as image:
            grayscale_image = image.convert("L")
    except OSError as error:
        raise ValueError(f"Could not open image '{image_path}': {error}") from error

    # Pillow provides both the minimum/maximum pixel values and summary statistics.
    minimum_pixel_value, maximum_pixel_value = grayscale_image.getextrema()
    intensity_statistics = ImageStat.Stat(grayscale_image)

    return {
        "width": grayscale_image.width,
        "height": grayscale_image.height,
        "minimum_pixel_value": minimum_pixel_value,
        "maximum_pixel_value": maximum_pixel_value,
        "mean_intensity": intensity_statistics.mean[0],
        "standard_deviation": intensity_statistics.stddev[0],
    }