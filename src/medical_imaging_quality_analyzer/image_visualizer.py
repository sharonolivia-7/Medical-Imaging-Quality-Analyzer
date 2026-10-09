"""Display a grayscale image alongside its pixel-intensity histogram."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def visualize_image(image_path: str | Path) -> None:
    """Show a grayscale image and its intensity histogram in one figure."""
    try:
        with Image.open(image_path) as image:
            grayscale_image = image.convert("L")
    except OSError as error:
        raise ValueError(f"Could not open image '{image_path}': {error}") from error

    # Create two side-by-side areas: one for the image and one for its histogram.
    figure, (image_axes, histogram_axes) = plt.subplots(1, 2, figsize=(10, 4))

    image_axes.imshow(grayscale_image, cmap="gray", vmin=0, vmax=255)
    image_axes.set_title("Grayscale Image")
    image_axes.set_xlabel("Pixel column")
    image_axes.set_ylabel("Pixel row")

    # Each histogram entry counts how many pixels have that intensity (0 through 255).
    histogram_axes.plot(range(256), grayscale_image.histogram())
    histogram_axes.set_title("Pixel-Intensity Histogram")
    histogram_axes.set_xlabel("Pixel intensity")
    histogram_axes.set_ylabel("Number of pixels")
    histogram_axes.set_xlim(0, 255)

    figure.tight_layout()
    plt.show()
