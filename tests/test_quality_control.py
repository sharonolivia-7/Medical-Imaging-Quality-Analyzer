import unittest

import numpy as np

from medical_imaging_quality_analyzer.quality_control import check_image_quality


class CheckImageQualityTests(unittest.TestCase):
    def test_constant_image_returns_warning_and_measurements(self):
        sample_image = np.full((3, 4), 12, dtype=np.uint8)

        result = check_image_quality(sample_image)

        self.assertEqual(result["overall_status"], "warning")
        self.assertEqual(
            result["measurements"],
            {
                "width": 4,
                "height": 3,
                "mean_intensity": 12.0,
                "standard_deviation": 0.0,
                "all_pixels_identical": True,
            },
        )
        self.assertEqual(result["checks"]["pixel_variation"]["status"], "warning")

    def test_varying_image_passes_with_configured_thresholds(self):
        sample_image = np.array([[0, 10], [20, 30]], dtype=np.uint8)

        result = check_image_quality(
            sample_image,
            minimum_width=2,
            minimum_height=2,
            minimum_mean_intensity=10,
            maximum_mean_intensity=20,
            maximum_standard_deviation=12,
        )

        self.assertEqual(result["overall_status"], "pass")
        self.assertEqual(result["measurements"]["width"], 2)
        self.assertEqual(result["measurements"]["height"], 2)
        self.assertAlmostEqual(result["measurements"]["mean_intensity"], 15.0)
        self.assertFalse(result["measurements"]["all_pixels_identical"])
        self.assertTrue(
            all(check["status"] == "pass" for check in result["checks"].values())
        )

    def test_empty_image_returns_clear_failure_result(self):
        result = check_image_quality(np.empty((0, 3), dtype=np.uint8))

        self.assertEqual(result["overall_status"], "fail")
        self.assertIsNone(result["measurements"])
        self.assertEqual(result["checks"]["input"]["status"], "fail")
        self.assertIn("empty", result["checks"]["input"]["explanation"])

    def test_invalid_input_returns_clear_failure_result(self):
        result = check_image_quality(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertEqual(result["overall_status"], "fail")
        self.assertIsNone(result["measurements"])
        self.assertIn("2D grayscale", result["checks"]["input"]["explanation"])

    def test_configured_threshold_violation_is_a_warning(self):
        result = check_image_quality(
            np.array([[0, 1], [2, 3]], dtype=np.uint8),
            minimum_width=3,
            maximum_mean_intensity=1.4,
        )

        self.assertEqual(result["overall_status"], "warning")
        self.assertEqual(result["checks"]["dimensions"]["status"], "warning")
        self.assertEqual(result["checks"]["mean_intensity"]["status"], "warning")

    def test_rejects_inconsistent_thresholds(self):
        with self.assertRaisesRegex(ValueError, "cannot be greater"):
            check_image_quality(
                np.zeros((2, 2)),
                minimum_mean_intensity=10,
                maximum_mean_intensity=5,
            )


if __name__ == "__main__":
    unittest.main()
