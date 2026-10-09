import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medical_imaging_quality_analyzer.report_generator import generate_report


class GenerateReportTests(unittest.TestCase):
    def setUp(self):
        self.statistics = {
            "width": 4,
            "height": 3,
            "mean_intensity": 12.5,
            "standard_deviation": 2.5,
            "estimated_snr": 5.0,
            "PatientName": "SAMPLE^PERSON",
            "PatientID": "SAMPLE-123",
        }
        self.quality_results = {
            "overall_status": "warning",
            "measurements": {
                "width": 4,
                "height": 3,
                "mean_intensity": 12.5,
                "standard_deviation": 2.5,
            },
            "checks": {
                "dimensions": {
                    "status": "pass",
                    "explanation": "Dimensions are within the configured minimum.",
                },
                "pixel_variation": {
                    "status": "warning",
                    "explanation": "All pixel values are identical.",
                },
                "PatientName": {
                    "status": "warning",
                    "explanation": "SAMPLE^PERSON",
                },
            },
            "PatientID": "SAMPLE-123",
        }

    def test_writes_report_with_measurements_and_quality_warning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "sample_report.txt"

            returned_path = generate_report(
                self.statistics,
                self.quality_results,
                report_path,
            )

            self.assertEqual(returned_path, report_path)
            self.assertTrue(report_path.is_file())
            report_content = report_path.read_text(encoding="utf-8")

        self.assertIn("Image dimensions: 4 x 3 pixels", report_content)
        self.assertIn("Mean intensity: 12.5", report_content)
        self.assertIn("Standard deviation: 2.5", report_content)
        self.assertIn("Estimated SNR: 5.0", report_content)
        self.assertIn(
            "- pixel_variation: All pixel values are identical.", report_content
        )
        self.assertIn("not clinically validated", report_content)
        self.assertNotIn("SAMPLE^PERSON", report_content)
        self.assertNotIn("SAMPLE-123", report_content)
        self.assertNotIn("PatientName", report_content)
        self.assertNotIn("PatientID", report_content)

    def test_missing_optional_measurements_are_described(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "partial_report.txt"
            generate_report({}, {}, report_path)
            report_content = report_path.read_text(encoding="utf-8")

        self.assertIn("Image dimensions: Not provided", report_content)
        self.assertIn("Mean intensity: Not provided", report_content)
        self.assertIn("Standard deviation: Not provided", report_content)
        self.assertIn("Estimated SNR: Not provided", report_content)
        self.assertIn("No quality warnings reported.", report_content)

    def test_default_path_is_inside_reports_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            reports_directory = Path(temporary_directory) / "reports"
            with patch(
                "medical_imaging_quality_analyzer.report_generator.REPORTS_DIRECTORY",
                reports_directory,
            ):
                report_path = generate_report({}, {})

            self.assertEqual(report_path.parent, reports_directory)
            self.assertEqual(report_path.suffix, ".txt")
            self.assertTrue(report_path.is_file())

    def test_does_not_overwrite_an_existing_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "existing.txt"
            report_path.write_text("keep this content", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                generate_report({}, {}, report_path)

            self.assertEqual(
                report_path.read_text(encoding="utf-8"), "keep this content"
            )


if __name__ == "__main__":
    unittest.main()
