import tempfile
import unittest
from pathlib import Path

from PIL import Image

from medical_imaging_quality_analyzer.image_analyzer import analyze_image


class AnalyzeImageTests(unittest.TestCase):
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
