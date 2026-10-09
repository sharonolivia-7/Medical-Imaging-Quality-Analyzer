"""Command-line entry point for the educational image-quality workflow."""

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from medical_imaging_quality_analyzer.dicom_reader import read_dicom_image
from medical_imaging_quality_analyzer.frequency_analysis import analyze_frequency
from medical_imaging_quality_analyzer.image_analyzer import (
    analyze_image,
    analyze_image_array,
)
from medical_imaging_quality_analyzer.image_processing import apply_gaussian_filter
from medical_imaging_quality_analyzer.image_visualizer import (
    visualize_image,
    visualize_image_array,
)
from medical_imaging_quality_analyzer.quality_control import check_image_quality
from medical_imaging_quality_analyzer.report_generator import generate_report
from medical_imaging_quality_analyzer.signal_analysis import analyze_signal


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a sample image with basic educational statistics and checks. "
            "This tool is not clinically validated or intended for diagnosis."
        )
    )
    parser.add_argument("image", type=Path, help="Path to a sample image file.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the image and its intensity histogram.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Apply the example Gaussian filter and print a short summary.",
    )
    parser.add_argument(
        "--frequency",
        action="store_true",
        help="Calculate the centered FFT and log-magnitude spectrum.",
    )
    parser.add_argument(
        "--dicom",
        action="store_true",
        help=(
            "Decode a single-frame grayscale DICOM image even when its filename "
            "does not end in .dcm. DICOM metadata values are not displayed."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional .txt report path. Defaults to a new file inside reports/.",
    )
    return parser.parse_args(arguments)


def _read_grayscale_array(image_path: Path) -> np.ndarray:
    """Load a standard image as a NumPy array of grayscale pixel values."""
    try:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("L")).copy()
    except OSError as error:
        raise ValueError(f"Could not load image '{image_path}': {error}") from error


def _format_estimated_snr(signal_statistics: dict[str, float | None]) -> str:
    """Explain when the simple SNR estimate is unavailable."""
    estimated_snr = signal_statistics["estimated_snr"]
    if estimated_snr is not None:
        return str(estimated_snr)
    if signal_statistics["mean_intensity"] <= 0:
        return "Not available (mean intensity is non-positive)"
    return "Not available"


def _run_dicom_workflow(
    image_path: Path,
    report_path: Path | None,
    apply_filter: bool,
    run_frequency_analysis: bool,
    show_image: bool,
) -> int:
    """Analyze DICOM pixels while reporting only safe metadata and measurements."""
    dicom_result = read_dicom_image(image_path)
    image_array = dicom_result["image"]
    image_statistics: dict[str, Any] = analyze_image_array(image_array)
    signal_statistics = analyze_signal(image_array)
    quality_results = check_image_quality(image_array)
    image_statistics["estimated_snr"] = signal_statistics["estimated_snr"]

    privacy_check = dicom_result["privacy_check"]
    privacy_detected = privacy_check["potentially_sensitive_metadata_present"]
    quality_results["checks"]["dicom_privacy"] = {
        "status": "warning" if privacy_detected else "pass",
        "explanation": (
            "Potentially sensitive DICOM metadata was detected; field values "
            "were omitted."
            if privacy_detected
            else "No known potentially sensitive fields or private tags were detected."
        ),
    }
    if privacy_detected and quality_results["overall_status"] == "pass":
        quality_results["overall_status"] = "warning"

    print("DICOM image analysis (patient-identifying metadata is not displayed):")
    print(f"  Dimensions: {dicom_result['width']} x {dicom_result['height']} pixels")
    print(f"  Modality: {dicom_result['modality'] or 'Not provided'}")
    print(f"  Pixel spacing: {dicom_result['pixel_spacing'] or 'Not provided'}")
    print(f"  Mean intensity: {signal_statistics['mean_intensity']:.4g}")
    print(f"  Standard deviation: {signal_statistics['standard_deviation']:.4g}")
    print(f"  Estimated SNR: {_format_estimated_snr(signal_statistics)}")
    print(f"  Quality-control status: {quality_results['overall_status']}")
    print(
        "  Potentially sensitive metadata detected: "
        f"{'yes' if privacy_detected else 'no'} (values omitted)"
    )

    if run_frequency_analysis:
        frequency_result = analyze_frequency(image_array)
        spectrum_shape = frequency_result["log_magnitude_spectrum"].shape
        image_statistics["frequency_spectrum_shape"] = list(spectrum_shape)
        print(
            "  Centered frequency spectrum calculated; "
            f"shape: {spectrum_shape[1]} x {spectrum_shape[0]}"
        )

    if apply_filter:
        filtered_image = apply_gaussian_filter(image_array)
        print(
            "  Gaussian filter applied; "
            f"output shape: {filtered_image.shape[1]} x {filtered_image.shape[0]}, "
            f"pixel range: {filtered_image.min():.4g} to "
            f"{filtered_image.max():.4g}"
        )

    if show_image:
        visualize_image_array(
        image_array,
        photometric_interpretation=dicom_result["photometric_interpretation"],
        )

    saved_report = generate_report(image_statistics, quality_results, report_path)
    print(f"Report saved to: {saved_report}")
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line image-analysis workflow."""
    options = _parse_arguments(arguments)
    is_dicom = options.dicom or options.image.suffix.lower() == ".dcm"

    try:
        if is_dicom:
            return _run_dicom_workflow(
                options.image,
                options.report,
                options.filter,
                options.frequency,
                options.show,
            )

        image_array = _read_grayscale_array(options.image)
        image_statistics: dict[str, Any] = analyze_image(options.image)
        signal_statistics = analyze_signal(image_array)
        image_statistics["estimated_snr"] = signal_statistics["estimated_snr"]
        quality_results = check_image_quality(image_array)

        print("Image analysis (educational use only):")
        print(f"  Dimensions: {image_statistics['width']} x {image_statistics['height']} pixels")
        print(f"  Mean intensity: {image_statistics['mean_intensity']:.4g}")
        print(f"  Standard deviation: {image_statistics['standard_deviation']:.4g}")
        print(f"  Estimated SNR: {_format_estimated_snr(signal_statistics)}")
        print(f"  Quality-control status: {quality_results['overall_status']}")

        if options.frequency:
            frequency_result = analyze_frequency(image_array)
            spectrum_shape = frequency_result["log_magnitude_spectrum"].shape
            image_statistics["frequency_spectrum_shape"] = list(spectrum_shape)
            print(
                "  Centered frequency spectrum calculated; "
                f"shape: {spectrum_shape[1]} x {spectrum_shape[0]}"
            )

        if options.filter:
            filtered_image = apply_gaussian_filter(image_array)
            print(
                "  Gaussian filter applied; "
                f"output shape: {filtered_image.shape[1]} x {filtered_image.shape[0]}, "
                f"pixel range: {filtered_image.min():.4g} to "
                f"{filtered_image.max():.4g}"
            )

        if options.show:
            visualize_image(options.image)

        saved_report = generate_report(
            image_statistics,
            quality_results,
            options.report,
        )
        print(f"Report saved to: {saved_report}")
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
