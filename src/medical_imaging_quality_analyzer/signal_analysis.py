"""Basic educational signal statistics for grayscale image arrays."""

import numpy as np


def analyze_signal(image: np.ndarray) -> dict[str, float | None]:
    """Return mean intensity, standard deviation, and an estimated SNR.

    The estimated signal-to-noise ratio (SNR) uses the simple formula
    ``mean intensity / standard deviation``. This is a basic educational
    estimate, not a clinically validated medical-imaging measurement.

    A constant positive image has zero standard deviation, so its estimated
    SNR is infinity. For an all-zero image the ratio is undefined and is None.
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

    # Use float64 for straightforward, consistent statistics on common image types.
    pixel_values = image.astype(np.float64)
    mean_intensity = float(np.mean(pixel_values))
    standard_deviation = float(np.std(pixel_values))

    if standard_deviation == 0:
        estimated_snr = float("inf") if mean_intensity != 0 else None
    else:
        estimated_snr = mean_intensity / standard_deviation

    return {
        "mean_intensity": mean_intensity,
        "standard_deviation": standard_deviation,
        "estimated_snr": estimated_snr,
    }
