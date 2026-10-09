import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from medical_imaging_quality_analyzer.dicom_reader import (
    read_dicom_image,
    read_dicom_info,
)


class ReadDicomInfoTests(unittest.TestCase):
    sample_dicom_path = (
        Path(__file__).resolve().parents[1] / "data" / "sample" / "CT_small.dcm"
    )

    def create_sample_dicom(
        self,
        path: Path,
        include_pixel_data: bool = True,
        *,
        pixels: bytes = bytes([0, 64, 128, 192, 224, 255]),
        slope: float | None = None,
        intercept: float | None = None,
        samples_per_pixel: int = 1,
        photometric_interpretation: str = "MONOCHROME2",
    ) -> None:
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
        dataset.PixelSpacing = [0.5, 0.75]

        # Synthetic values only, used to confirm private fields are never returned.
        dataset.PatientName = "SAMPLE^PATIENT"
        dataset.PatientID = "SAMPLE-ID-001"
        dataset.add_new((0x0011, 0x0010), "LO", "SYNTHETIC_TEST")
        dataset.add_new((0x0011, 0x1010), "LO", "SYNTHETIC_PRIVATE_VALUE")

        if include_pixel_data:
            dataset.SamplesPerPixel = samples_per_pixel
            dataset.PhotometricInterpretation = photometric_interpretation
            dataset.BitsAllocated = 8
            dataset.BitsStored = 8
            dataset.HighBit = 7
            dataset.PixelRepresentation = 0
            dataset.PixelData = pixels
        if slope is not None:
            dataset.RescaleSlope = slope
        if intercept is not None:
            dataset.RescaleIntercept = intercept

        dataset.save_as(path, enforce_file_format=True)

    def test_reads_safe_metadata_from_synthetic_dicom(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "sample.dcm"
            self.create_sample_dicom(dicom_path)

            result = read_dicom_info(dicom_path)

        self.assertEqual(result["height"], 2)
        self.assertEqual(result["width"], 3)
        self.assertEqual(result["modality"], "OT")
        self.assertEqual(result["pixel_spacing"], [0.5, 0.75])
        self.assertTrue(result["pixel_data_present"])
        self.assertIn(
            "PatientName",
            result["privacy_check"]["potentially_sensitive_fields"],
        )
        self.assertIn(
            "PatientID",
            result["privacy_check"]["potentially_sensitive_fields"],
        )
        self.assertTrue(result["privacy_check"]["private_tags_present"])
        self.assertNotIn("SAMPLE^PATIENT", str(result))
        self.assertNotIn("SAMPLE-ID-001", str(result))
        self.assertNotIn("SYNTHETIC_PRIVATE_VALUE", str(result))

    def test_reports_missing_pixel_data_without_decoding_pixels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "no_pixels.dcm"
            self.create_sample_dicom(dicom_path, include_pixel_data=False)

            result = read_dicom_info(dicom_path)

        self.assertFalse(result["pixel_data_present"])
        self.assertEqual(result["height"], 2)
        self.assertEqual(result["width"], 3)

    def test_invalid_file_has_a_helpful_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.dcm"
            invalid_path.write_text("not a DICOM file", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not read DICOM file"):
                read_dicom_info(invalid_path)

    def test_decodes_grayscale_pixels_and_applies_rescale_without_mutating_stored_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "rescaled.dcm"
            original_pixels = bytes([1, 2, 3, 4, 5, 6])
            self.create_sample_dicom(
                dicom_path,
                pixels=original_pixels,
                slope=2.5,
                intercept=-3,
            )
            result = read_dicom_image(dicom_path)
            stored_dataset = pydicom.dcmread(dicom_path)

        expected_pixels = np.array(
            [[-0.5, 2.0, 4.5], [7.0, 9.5, 12.0]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(result["image"], expected_pixels)
        np.testing.assert_array_equal(
            stored_dataset.pixel_array,
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        )
        self.assertEqual(result["height"], 2)
        self.assertEqual(result["width"], 3)
        self.assertEqual(result["modality"], "OT")
        self.assertEqual(result["photometric_interpretation"], "MONOCHROME2")
        self.assertNotIn("PatientName", result)
        self.assertNotIn("PatientID", result)
        self.assertNotIn("SAMPLE^PATIENT", str(result))
        self.assertNotIn("SAMPLE-ID-001", str(result))
        self.assertIn(
            "PatientName",
            result["privacy_check"]["potentially_sensitive_fields"],
        )

    def test_decodes_grayscale_pixels_when_no_rescale_values_exist(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "plain.dcm"
            self.create_sample_dicom(
                dicom_path,
                pixels=bytes([0, 1, 2, 3, 4, 5]),
            )

            result = read_dicom_image(dicom_path)

        np.testing.assert_array_equal(
            result["image"],
            np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float64),
        )

    def test_decoding_rejects_missing_pixel_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "no_pixels.dcm"
            self.create_sample_dicom(dicom_path, include_pixel_data=False)

            with self.assertRaisesRegex(ValueError, "does not contain pixel data"):
                read_dicom_image(dicom_path)

    def test_decoding_rejects_color_pixel_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "color.dcm"
            self.create_sample_dicom(
                dicom_path,
                pixels=bytes([0, 0, 0, 255, 255, 255]),
                samples_per_pixel=3,
                photometric_interpretation="RGB",
            )

            with self.assertRaisesRegex(ValueError, "not a single-channel grayscale"):
                read_dicom_image(dicom_path)

    def test_decoding_rejects_multiframe_pixel_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "multiframe.dcm"
            self.create_sample_dicom(
                dicom_path,
                pixels=bytes(range(12)),
            )
            dataset = pydicom.dcmread(dicom_path)
            dataset.NumberOfFrames = 2
            dataset.save_as(dicom_path, enforce_file_format=True)

            with self.assertRaisesRegex(ValueError, "single 2D grayscale frame"):
                read_dicom_image(dicom_path)

    def test_decoding_rejects_invalid_rescale_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dicom_path = Path(temporary_directory) / "invalid_rescale.dcm"
            self.create_sample_dicom(dicom_path, slope=float("inf"))

            with self.assertRaisesRegex(ValueError, "invalid rescale slope or intercept"):
                read_dicom_image(dicom_path)

    def test_pydicom_sample_metadata_is_read_without_values(self):
        if not self.sample_dicom_path.is_file():
            self.skipTest("The optional CT_small.dcm sample is not present.")

        result = read_dicom_info(self.sample_dicom_path)

        self.assertEqual(result["height"], 128)
        self.assertEqual(result["width"], 128)
        self.assertTrue(result["pixel_data_present"])
        self.assertTrue(
            result["privacy_check"]["potentially_sensitive_metadata_present"]
        )
        self.assertIn(
            "PatientName",
            result["privacy_check"]["potentially_sensitive_fields"],
        )
        self.assertNotIn("PatientName", result)
        self.assertNotIn("PatientID", result)

    def test_pydicom_sample_pixel_array_loads_as_finite_grayscale(self):
        if not self.sample_dicom_path.is_file():
            self.skipTest("The optional CT_small.dcm sample is not present.")

        result = read_dicom_image(self.sample_dicom_path)

        self.assertEqual(result["image"].shape, (128, 128))
        self.assertTrue(np.isfinite(result["image"]).all())
        self.assertTrue(result["privacy_check"]["private_tags_present"])
        self.assertNotIn("PatientName", result)
        self.assertNotIn("PatientID", result)


if __name__ == "__main__":
    unittest.main()
