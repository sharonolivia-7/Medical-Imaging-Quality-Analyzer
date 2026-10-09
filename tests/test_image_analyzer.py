import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from medical_imaging_quality_analyzer.image_analyzer import (
    analyze_image,
    analyze_image_array,
)


class AnalyzeImageTests(unittest.TestCase):
    def test_analyzes_grayscale_array(self):
        image = np.array([[1, 3], [5, 7]], dtype=np.int16)

        result = analyze_image_array(image)

        self.assertEqual(result["width"], 2)
        self.assertEqual(result["height"], 2)
        self.assertEqual(result["minimum_pixel_value"], 1.0)
        self.assertEqual(result["maximum_pixel_value"], 7.0)
        self.assertEqual(result["mean_intensity"], 4.0)
        self.assertAlmostEqual(result["standard_deviation"], np.std(image))

    def test_array_analysis_rejects_empty_and_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            analyze_image_array(np.empty((0, 0)))
        with self.assertRaisesRegex(ValueError, "2D grayscale"):
            analyze_image_array(np.zeros((2, 2, 3)))

    def test_returns_dimensions_and_grayscale_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "sample.png"
            sample_image = Image.new("L", (2, 2))
            sample_image.putdata([0, 64, 128, 255])
            sample_image.save(image_path)

            result = analyze_image(image_path)

        self.assertEqual(result["width"], 2)
        self.assertEqual(result["height"], 2)
        self.assertEqual(result["minimum_pixel_value"], 0)
        self.assertEqual(result["maximum_pixel_value"], 255)
        self.assertAlmostEqual(result["mean_intensity"], 111.75)
        self.assertAlmostEqual(result["standard_deviation"], 94.28, places=2)

    def test_missing_image_path_has_a_helpful_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.png"

            with self.assertRaisesRegex(ValueError, "Could not open image"):
                analyze_image(missing_path)


if __name__ == "__main__":
    unittest.main()
