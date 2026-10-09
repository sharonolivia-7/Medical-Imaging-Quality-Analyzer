"""Command-line entry point for the educational image-quality workflow."""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from medical_imaging_quality_analyzer.dicom_reader import read_dicom_image
from medical_imaging_quality_analyzer.image_analyzer import analyze_image
from medical_imaging_quality_analyzer.image_processing import apply_gaussian_filter
from medical_imaging_quality_analyzer.image_visualizer import visualize_image
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
        "--dicom",
        action="store_true",
        help=(
            "Decode a single-frame grayscale DICOM image for analysis. "
            "Patient metadata is not displayed."
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


def _run_dicom_workflow(
    image_path: Path,
    report_path: Path | None,
    apply_filter: bool,
) -> int:
    """Analyze DICOM pixels while reporting only safe metadata and measurements."""
    dicom_result = read_dicom_image(image_path)
    image_array = dicom_result["image"]
    signal_statistics = analyze_signal(image_array)
    quality_results = check_image_quality(image_array)
    image_statistics = {
        "width": dicom_result["width"],
        "height": dicom_result["height"],
        "minimum_pixel_value": float(np.min(image_array)),
        "maximum_pixel_value": float(np.max(image_array)),
        "mean_intensity": signal_statistics["mean_intensity"],
        "standard_deviation": signal_statistics["standard_deviation"],
        "estimated_snr": signal_statistics["estimated_snr"],
    }

    print("DICOM image analysis (patient-identifying metadata is not displayed):")
    print(f"  Dimensions: {dicom_result['width']} x {dicom_result['height']} pixels")
    print(f"  Modality: {dicom_result['modality'] or 'Not provided'}")
    print(f"  Pixel spacing: {dicom_result['pixel_spacing'] or 'Not provided'}")
    print(f"  Mean intensity: {signal_statistics['mean_intensity']:.4g}")
    print(f"  Standard deviation: {signal_statistics['standard_deviation']:.4g}")
    print(f"  Estimated SNR: {signal_statistics['estimated_snr']}")
    print(f"  Quality-control status: {quality_results['overall_status']}")

    if apply_filter:
        filtered_image = apply_gaussian_filter(image_array)
        print(
            "  Gaussian filter applied; "
            f"output shape: {filtered_image.shape[1]} x {filtered_image.shape[0]}, "
            f"pixel range: {filtered_image.min():.4g} to "
            f"{filtered_image.max():.4g}"
        )

    saved_report = generate_report(image_statistics, quality_results, report_path)
    print(f"Report saved to: {saved_report}")
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line image-analysis workflow."""
    options = _parse_arguments(arguments)

    if options.dicom and options.show:
        print(
            "Error: --show currently supports regular image files only. "
            "DICOM pixel analysis, filtering, and reporting are available.",
        )
        return 2

    try:
        if options.dicom:
            return _run_dicom_workflow(
                options.image,
                options.report,
                options.filter,
            )

        image_array = _read_grayscale_array(options.image)
        image_statistics = analyze_image(options.image)
        signal_statistics = analyze_signal(image_array)
        image_statistics["estimated_snr"] = signal_statistics["estimated_snr"]
        quality_results = check_image_quality(image_array)

        print("Image analysis (educational use only):")
        print(f"  Dimensions: {image_statistics['width']} x {image_statistics['height']} pixels")
        print(f"  Mean intensity: {image_statistics['mean_intensity']:.4g}")
        print(f"  Standard deviation: {image_statistics['standard_deviation']:.4g}")
        print(f"  Estimated SNR: {image_statistics['estimated_snr']}")
        print(f"  Quality-control status: {quality_results['overall_status']}")

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
