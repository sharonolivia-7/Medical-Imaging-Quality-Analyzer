"""Display a grayscale image alongside its pixel-intensity histogram."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _display_image_and_histogram(
    grayscale_image: np.ndarray,
    histogram_x: np.ndarray,
    histogram_counts: np.ndarray,
    histogram_xlim: tuple[float, float] | None,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Display grayscale pixels and their intensity distribution."""
    figure, (image_axes, histogram_axes) = plt.subplots(1, 2, figsize=(10, 4))

    image_axes.imshow(grayscale_image, cmap=cmap, vmin=vmin, vmax=vmax)
    image_axes.set_title("Grayscale Image")
    image_axes.set_xlabel("Pixel column")
    image_axes.set_ylabel("Pixel row")

    histogram_axes.plot(histogram_x, histogram_counts)
    histogram_axes.set_title("Pixel-Intensity Histogram")
    histogram_axes.set_xlabel("Pixel intensity")
    histogram_axes.set_ylabel("Number of pixels")
    if histogram_xlim is not None:
        histogram_axes.set_xlim(*histogram_xlim)

    figure.tight_layout()
    plt.show()


def visualize_image(image_path: str | Path) -> None:
    """Show a grayscale image and its intensity histogram in one figure."""
    try:
        with Image.open(image_path) as image:
            grayscale_image = np.asarray(image.convert("L")).copy()
    except OSError as error:
        raise ValueError(f"Could not open image '{image_path}': {error}") from error

    # Each histogram entry counts how many pixels have that intensity (0 through 255).
    histogram_counts = np.asarray(Image.fromarray(grayscale_image).histogram())
    _display_image_and_histogram(
        grayscale_image,
        np.arange(256),
        histogram_counts,
        (0, 255),
        cmap="gray",
        vmin=0,
        vmax=255,
    )


def visualize_image_array(
    image: np.ndarray,
    photometric_interpretation: str = "MONOCHROME2",
) -> None:
    """Display grayscale array data, including rescaled DICOM pixels.

    The image is displayed using its floating-point pixel range. DICOM
    ``MONOCHROME1`` data uses reversed grayscale so the lowest values display
    as white, as specified by that photometric interpretation.
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
    if photometric_interpretation not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(
            "photometric_interpretation must be MONOCHROME1 or MONOCHROME2."
        )

    grayscale_image = image.astype(np.float64, copy=False)
    histogram_counts, histogram_edges = np.histogram(grayscale_image, bins=256)
    histogram_x = (histogram_edges[:-1] + histogram_edges[1:]) / 2
    cmap = "gray_r" if photometric_interpretation == "MONOCHROME1" else "gray"
    _display_image_and_histogram(
        grayscale_image,
        histogram_x,
        histogram_counts,
        None,
        cmap=cmap,
    )
