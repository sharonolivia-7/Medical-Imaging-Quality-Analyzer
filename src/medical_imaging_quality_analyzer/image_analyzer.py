"""Basic measurements for grayscale sample images."""

from pathlib import Path

from PIL import Image, ImageStat


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