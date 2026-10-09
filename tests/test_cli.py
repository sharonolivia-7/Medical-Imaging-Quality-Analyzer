import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from medical_imaging_quality_analyzer.__main__ import main


class CommandLineIntegrationTests(unittest.TestCase):
    def create_sample_image(self, path: Path) -> None:
        sample_image = Image.new("L", (3, 2))
        sample_image.putdata([0, 32, 64, 128, 192, 255])
        sample_image.save(path)

    def create_sample_dicom(self, path: Path) -> None:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(
            str(path),
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.Modality = "OT"
        dataset.Rows = 2
        dataset.Columns = 3
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.PatientName = "SAMPLE^PATIENT"
        dataset.PatientID = "SAMPLE-ID"
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 7
        dataset.PixelRepresentation = 0
        dataset.PixelData = bytes([0, 32, 64, 128, 192, 255])
        dataset.RescaleSlope = 2
        dataset.RescaleIntercept = 1
        dataset.save_as(path, enforce_file_format=True)

    def test_image_workflow_creates_report_without_opening_graphics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            image_path = folder / "sample.png"
            report_path = folder / "analysis.txt"
            self.create_sample_image(image_path)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        str(image_path),
                        "--filter",
                        "--report",
                        str(report_path),
                    ]
                )

            report_content = report_path.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertIn("Mean intensity:", output.getvalue())
        self.assertIn("Gaussian filter applied", output.getvalue())
        self.assertIn("Report saved to:", output.getvalue())
        self.assertIn("Image dimensions: 3 x 2 pixels", report_content)
        self.assertIn("Estimated SNR:", report_content)

    def test_visualization_is_only_called_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            image_path = folder / "sample.png"
            report_path = folder / "analysis.txt"
            self.create_sample_image(image_path)

            output = io.StringIO()
            with patch("medical_imaging_quality_analyzer.__main__.visualize_image") as show:
                with contextlib.redirect_stdout(output):
                    result = main([str(image_path), "--report", str(report_path)])

            show.assert_not_called()

        self.assertEqual(result, 0)

    def test_show_option_calls_visualizer_without_opening_a_window_in_test(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            image_path = folder / "sample.png"
            report_path = folder / "analysis.txt"
            self.create_sample_image(image_path)
            output = io.StringIO()

            with patch("medical_imaging_quality_analyzer.__main__.visualize_image") as show:
                with contextlib.redirect_stdout(output):
                    result = main(
                        [
                            str(image_path),
                            "--show",
                            "--report",
                            str(report_path),
                        ]
                    )

            show.assert_called_once_with(image_path)

        self.assertEqual(result, 0)

    def test_dicom_workflow_analyzes_pixels_without_patient_identifiers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            dicom_path = folder / "sample.dcm"
            report_path = folder / "analysis.txt"
            self.create_sample_dicom(dicom_path)
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = main(
                    [
                        str(dicom_path),
                        "--dicom",
                        "--report",
                        str(report_path),
                    ]
                )

            report_content = report_path.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertIn("3 x 2 pixels", output.getvalue())
        self.assertIn("Mean intensity:", report_content)
        self.assertIn("Estimated SNR:", report_content)
        self.assertIn("Mean intensity: 224.66666666666666", report_content)
        self.assertNotIn("SAMPLE^PATIENT", output.getvalue())
        self.assertNotIn("SAMPLE-ID", output.getvalue())
        self.assertNotIn("SAMPLE^PATIENT", report_content)
        self.assertNotIn("SAMPLE-ID", report_content)

    def test_reports_clear_error_for_missing_image(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "missing.png"
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = main([str(image_path)])

        self.assertEqual(result, 1)
        self.assertIn("Error: Could not load image", output.getvalue())

    def test_dicom_rejects_show_with_clear_error(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["sample.dcm", "--dicom", "--show"])

        self.assertEqual(result, 2)
        self.assertIn("--show currently supports regular image files only", output.getvalue())


if __name__ == "__main__":
    unittest.main()
