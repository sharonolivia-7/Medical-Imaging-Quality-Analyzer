"""Basic 2D frequency analysis for grayscale image arrays."""

import numpy as np


def analyze_frequency(image: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate the 2D FFT and a centered log-magnitude spectrum.

    The returned dictionary contains:

    - ``fft``: the unshifted complex 2D Fast Fourier Transform.
    - ``shifted_fft``: the FFT with its zero-frequency component at the center.
    - ``log_magnitude_spectrum``: ``log1p(abs(shifted_fft))``, suitable for
      displaying the spectrum while making large and small values easier to see.

    This is an educational signal-processing operation, not a clinically
    validated medical-imaging measurement.
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

    # Floating-point input avoids integer overflow in the transform.
    fft_result = np.fft.fft2(image.astype(np.float64))
    shifted_fft = np.fft.fftshift(fft_result)
    log_magnitude_spectrum = np.log1p(np.abs(shifted_fft))

    return {
        "fft": fft_result,
        "shifted_fft": shifted_fft,
        "log_magnitude_spectrum": log_magnitude_spectrum,
    }
