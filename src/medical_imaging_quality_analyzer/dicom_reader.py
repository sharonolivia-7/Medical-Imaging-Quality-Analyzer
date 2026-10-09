"""Read a small, privacy-conscious set of DICOM image details."""

from pathlib import Path
from typing import Any

import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np


def _read_dataset(dicom_path: str | Path) -> pydicom.Dataset:
    """Read a DICOM dataset and convert common file errors to a clear message."""
    try:
        return pydicom.dcmread(dicom_path)
    except (OSError, InvalidDicomError, ValueError) as error:
        raise ValueError(f"Could not read DICOM file '{dicom_path}': {error}") from error


def read_dicom_info(dicom_path: str | Path) -> dict[str, Any]:
    """Read selected image details without returning patient identifiers.

    The returned dictionary contains height, width, modality, pixel spacing when
    present, and whether pixel data is present. Pixel data is not decoded.
    """
    dataset = _read_dataset(dicom_path)

    # Only copy these explicitly selected fields; never return the full dataset,
    # which may contain patient-identifying metadata.
    result: dict[str, Any] = {
        "height": int(dataset.Rows) if "Rows" in dataset else None,
        "width": int(dataset.Columns) if "Columns" in dataset else None,
        "modality": str(dataset.Modality) if "Modality" in dataset else None,
        "pixel_spacing": None,
        "pixel_data_present": "PixelData" in dataset,
    }

    if "PixelSpacing" in dataset:
        result["pixel_spacing"] = [float(value) for value in dataset.PixelSpacing]

    return result


def read_dicom_image(dicom_path: str | Path) -> dict[str, Any]:
    """Decode a single-frame grayscale DICOM image and apply rescale values.

    Returns only the rescaled ``image`` NumPy array and safe metadata
    (height, width, modality, and pixel spacing). The dataset and its
    patient-identifying metadata are never returned. Rescaling is performed
    on a floating-point copy, leaving stored pixel values unchanged.

    Color images, multi-frame images, missing pixel data, and unsupported or
    corrupted pixel encodings raise a helpful ``ValueError``.
    """
    dataset = _read_dataset(dicom_path)

    if "PixelData" not in dataset:
        raise ValueError(f"DICOM file '{dicom_path}' does not contain pixel data.")

    samples_per_pixel = int(dataset.get("SamplesPerPixel", 1))
    if samples_per_pixel != 1:
        raise ValueError(
            f"DICOM file '{dicom_path}' is not a single-channel grayscale image."
        )

    photometric_interpretation = str(dataset.get("PhotometricInterpretation", ""))
    if photometric_interpretation not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(
            f"DICOM file '{dicom_path}' has unsupported grayscale interpretation "
            f"'{photometric_interpretation or 'not specified'}'."
        )

    try:
        stored_pixels = dataset.pixel_array
    except (AttributeError, NotImplementedError, OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"Could not decode pixel data in DICOM file '{dicom_path}': {error}"
        ) from error

    if stored_pixels.ndim != 2:
        raise ValueError(
            f"DICOM file '{dicom_path}' must contain a single 2D grayscale frame."
        )
    if stored_pixels.size == 0 or not np.issubdtype(stored_pixels.dtype, np.number):
        raise ValueError(
            f"DICOM file '{dicom_path}' contains empty or invalid pixel data."
        )
    if not np.isfinite(stored_pixels).all():
        raise ValueError(
            f"DICOM file '{dicom_path}' contains non-finite pixel values."
        )

    try:
        slope = float(dataset.get("RescaleSlope", 1.0))
        intercept = float(dataset.get("RescaleIntercept", 0.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"DICOM file '{dicom_path}' has invalid rescale slope or intercept."
        ) from error
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError(
            f"DICOM file '{dicom_path}' has invalid rescale slope or intercept."
        )

    expected_shape = (
        int(dataset.Rows) if "Rows" in dataset else stored_pixels.shape[0],
        int(dataset.Columns) if "Columns" in dataset else stored_pixels.shape[1],
    )
    if stored_pixels.shape != expected_shape:
        raise ValueError(
            f"DICOM file '{dicom_path}' pixel dimensions do not match its metadata."
        )

    # Keep the decoded stored values intact and rescale a separate float array.
    rescaled_pixels = stored_pixels.astype(np.float64) * slope + intercept

    safe_metadata: dict[str, Any] = {
        "height": int(dataset.Rows) if "Rows" in dataset else rescaled_pixels.shape[0],
        "width": int(dataset.Columns) if "Columns" in dataset else rescaled_pixels.shape[1],
        "modality": str(dataset.Modality) if "Modality" in dataset else None,
        "pixel_spacing": None,
    }
    if "PixelSpacing" in dataset:
        safe_metadata["pixel_spacing"] = [
            float(value) for value in dataset.PixelSpacing
        ]

    return {"image": rescaled_pixels, **safe_metadata}
