"""Basic educational checks for grayscale image arrays."""

from numbers import Real
from typing import Any

import numpy as np


def check_image_quality(
    image: np.ndarray,
    *,
    minimum_width: int = 1,
    minimum_height: int = 1,
    minimum_mean_intensity: float | None = None,
    maximum_mean_intensity: float | None = None,
    maximum_standard_deviation: float | None = None,
) -> dict[str, Any]:
    """Measure a grayscale image and report simple, configurable checks.

    Thresholds are provided by the caller because this educational function
    does not define universal or clinically validated image-quality standards.
    Invalid image data is reported in the result instead of raising an exception.
    Invalid threshold configuration raises ``ValueError``.
    """
    if not isinstance(minimum_width, int) or isinstance(minimum_width, bool):
        raise ValueError("minimum_width must be a positive integer.")
    if not isinstance(minimum_height, int) or isinstance(minimum_height, bool):
        raise ValueError("minimum_height must be a positive integer.")
    if minimum_width < 1 or minimum_height < 1:
        raise ValueError("minimum_width and minimum_height must be at least 1.")

    for threshold_name, threshold_value in (
        ("minimum_mean_intensity", minimum_mean_intensity),
        ("maximum_mean_intensity", maximum_mean_intensity),
        ("maximum_standard_deviation", maximum_standard_deviation),
    ):
        if threshold_value is not None and (
            not isinstance(threshold_value, Real)
            or not np.isfinite(threshold_value)
            or isinstance(threshold_value, bool)
        ):
            raise ValueError(f"{threshold_name} must be a finite number or None.")

    if (
        minimum_mean_intensity is not None
        and maximum_mean_intensity is not None
        and minimum_mean_intensity > maximum_mean_intensity
    ):
        raise ValueError(
            "minimum_mean_intensity cannot be greater than maximum_mean_intensity."
        )
    if maximum_standard_deviation is not None and maximum_standard_deviation < 0:
        raise ValueError("maximum_standard_deviation cannot be negative.")

    checks: dict[str, dict[str, str]] = {}

    def invalid_result(message: str) -> dict[str, Any]:
        checks["input"] = {"status": "fail", "explanation": message}
        checks["dimensions"] = {
            "status": "not_checked",
            "explanation": "Dimensions cannot be checked until the image is valid.",
        }
        checks["pixel_variation"] = {
            "status": "not_checked",
            "explanation": "Pixel variation cannot be checked until the image is valid.",
        }
        return {
            "overall_status": "fail",
            "measurements": None,
            "checks": checks,
        }

    if not isinstance(image, np.ndarray):
        return invalid_result("Input must be a NumPy array.")
    if image.ndim != 2:
        return invalid_result("Input must be a 2D grayscale image.")
    if image.size == 0:
        return invalid_result("Image is empty; provide at least one pixel.")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        return invalid_result("Image must contain real numeric pixel values.")
    if not np.isfinite(image).all():
        return invalid_result("Image must contain only finite pixel values.")

    pixel_values = image.astype(np.float64)
    height, width = pixel_values.shape
    mean_intensity = float(np.mean(pixel_values))
    standard_deviation = float(np.std(pixel_values))
    all_pixels_identical = bool(np.all(pixel_values == pixel_values.flat[0]))

    checks["input"] = {
        "status": "pass",
        "explanation": "Image is a non-empty 2D array of finite numeric values.",
    }

    dimensions_pass = width >= minimum_width and height >= minimum_height
    checks["dimensions"] = {
        "status": "pass" if dimensions_pass else "warning",
        "explanation": (
            f"Image dimensions are {width} by {height} pixels; configured minimum "
            f"is {minimum_width} by {minimum_height} pixels."
        ),
    }

    checks["pixel_variation"] = {
        "status": "warning" if all_pixels_identical else "pass",
        "explanation": (
            "All pixel values are identical."
            if all_pixels_identical
            else "Pixel values vary across the image."
        ),
    }

    if minimum_mean_intensity is not None and mean_intensity < minimum_mean_intensity:
        mean_status = "warning"
        mean_explanation = (
            f"Mean intensity {mean_intensity:.4g} is below the configured minimum "
            f"of {minimum_mean_intensity:.4g}."
        )
    elif (
        maximum_mean_intensity is not None
        and mean_intensity > maximum_mean_intensity
    ):
        mean_status = "warning"
        mean_explanation = (
            f"Mean intensity {mean_intensity:.4g} is above the configured maximum "
            f"of {maximum_mean_intensity:.4g}."
        )
    else:
        mean_status = "pass"
        mean_explanation = "Mean intensity is within the configured limits, if any."
    checks["mean_intensity"] = {
        "status": mean_status,
        "explanation": mean_explanation,
    }

    if (
        maximum_standard_deviation is not None
        and standard_deviation > maximum_standard_deviation
    ):
        deviation_status = "warning"
        deviation_explanation = (
            f"Standard deviation {standard_deviation:.4g} is above the configured "
            f"maximum of {maximum_standard_deviation:.4g}."
        )
    else:
        deviation_status = "pass"
        deviation_explanation = (
            "Standard deviation is within the configured limit, if any."
        )
    checks["standard_deviation"] = {
        "status": deviation_status,
        "explanation": deviation_explanation,
    }

    statuses = [check["status"] for check in checks.values()]
    overall_status = "warning" if "warning" in statuses else "pass"

    return {
        "overall_status": overall_status,
        "measurements": {
            "width": width,
            "height": height,
            "mean_intensity": mean_intensity,
            "standard_deviation": standard_deviation,
            "all_pixels_identical": all_pixels_identical,
        },
        "checks": checks,
    }
