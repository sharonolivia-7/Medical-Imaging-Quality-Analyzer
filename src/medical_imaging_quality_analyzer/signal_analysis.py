"""Basic educational signal statistics for grayscale image arrays."""

import numpy as np


def analyze_signal(image: np.ndarray) -> dict[str, float | None]:
    """Return mean intensity, standard deviation, and an estimated SNR.

    The estimated signal-to-noise ratio (SNR) uses the simple formula
    ``mean intensity / standard deviation`` only when the mean intensity is
    positive. A negative or zero mean is not a meaningful non-negative signal
    reference, as can occur with signed scales such as CT values, so the
    estimate is ``None`` rather than a negative ratio. This simple estimate is
    educational only: whole-image standard deviation includes image structure
    as well as noise, and it is not a substitute for a defined signal region
    and noise measurement.

    A constant positive image has zero standard deviation, so its estimated
    SNR is infinity. For a non-positive mean or an all-zero image the ratio is
    undefined as an SNR estimate and is None.
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

    if mean_intensity <= 0:
        estimated_snr = None
    elif standard_deviation == 0:
        estimated_snr = float("inf")
    else:
        estimated_snr = mean_intensity / standard_deviation

    return {
        "mean_intensity": mean_intensity,
        "standard_deviation": standard_deviation,
        "estimated_snr": estimated_snr,
    }
