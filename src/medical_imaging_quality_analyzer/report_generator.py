"""Create a simple text report from image measurements and quality checks."""

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

REPORTS_DIRECTORY = Path(__file__).resolve().parents[2] / "reports"
_KNOWN_CHECKS = (
    "input",
    "dimensions",
    "pixel_variation",
    "mean_intensity",
    "standard_deviation",
    "dicom_privacy",
)


def _format_measurement(value: Any) -> str:
    """Format a measurement while handling missing and infinite values."""
    if value is None:
        return "Not provided"
    if isinstance(value, float):
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    return str(value)


def _default_report_path() -> Path:
    """Create a timestamped report path that will not replace another report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return REPORTS_DIRECTORY / f"quality_report_{timestamp}.txt"


def generate_report(
    image_statistics: Mapping[str, Any],
    quality_control_results: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """Write a text report and return its path.

    Only expected image measurements and known quality-check results are
    included. Other metadata fields are ignored to avoid leaking identifiers.
    If no output path is provided, a new report is saved in the project's
    ``reports`` directory.
    """
    if not isinstance(image_statistics, Mapping):
        raise TypeError("image_statistics must be a mapping.")
    if not isinstance(quality_control_results, Mapping):
        raise TypeError("quality_control_results must be a mapping.")

    qc_measurements = quality_control_results.get("measurements")
    if not isinstance(qc_measurements, Mapping):
        qc_measurements = {}

    def get_measurement(name: str) -> Any:
        value = image_statistics.get(name)
        if value is None:
            value = qc_measurements.get(name)
        return value

    width = get_measurement("width")
    height = get_measurement("height")
    if width is None or height is None:
        dimensions = "Not provided"
    else:
        dimensions = f"{width} x {height} pixels"

    snr = image_statistics.get("estimated_snr")
    if snr is None:
        snr = image_statistics.get("snr")
    mean_intensity = get_measurement("mean_intensity")
    if snr is None and isinstance(mean_intensity, (int, float)) and mean_intensity <= 0:
        snr_display = "Not available (mean intensity is non-positive)"
    else:
        snr_display = _format_measurement(snr)

    frequency_shape = image_statistics.get("frequency_spectrum_shape")
    if (
        isinstance(frequency_shape, (list, tuple))
        and len(frequency_shape) == 2
        and all(isinstance(dimension, int) for dimension in frequency_shape)
    ):
        frequency_summary = (
            f"Calculated ({frequency_shape[1]} x {frequency_shape[0]} bins)"
        )
    else:
        frequency_summary = "Not calculated"

    checks = quality_control_results.get("checks")
    if not isinstance(checks, Mapping):
        checks = {}

    warning_lines: list[str] = []
    for check_name in _KNOWN_CHECKS:
        check_result = checks.get(check_name)
        if not isinstance(check_result, Mapping):
            continue
        status = str(check_result.get("status", "")).lower()
        if status in {"warning", "fail"}:
            explanation = check_result.get("explanation")
            if isinstance(explanation, str) and explanation:
                warning_lines.append(f"- {check_name}: {explanation}")
            else:
                warning_lines.append(f"- {check_name}: {status.title()}.")

    if not warning_lines:
        warning_section = "No quality warnings reported."
    else:
        warning_section = "\n".join(warning_lines)

    report_text = (
        "Image Quality Analysis Report\n"
        "==============================\n"
        "Educational use only. This report is not clinically validated and is "
        "not intended for diagnosis.\n\n"
        f"Image dimensions: {dimensions}\n"
        f"Mean intensity: {_format_measurement(get_measurement('mean_intensity'))}\n"
        "Standard deviation: "
        f"{_format_measurement(get_measurement('standard_deviation'))}\n"
        f"Estimated SNR: {snr_display}\n"
        f"Frequency spectrum: {frequency_summary}\n"
        "Quality-control status: "
        f"{_format_measurement(quality_control_results.get('overall_status'))}\n\n"
        "Quality warnings\n"
        "-----------------\n"
        f"{warning_section}\n"
    )

    report_path = Path(output_path) if output_path is not None else _default_report_path()
    if report_path.suffix.lower() != ".txt":
        raise ValueError("The report output path must use the .txt extension.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an existing report.
    with report_path.open("x", encoding="utf-8") as report_file:
        report_file.write(report_text)

    return report_path
