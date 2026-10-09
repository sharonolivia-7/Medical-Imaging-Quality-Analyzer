import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

# Select a non-interactive backend so tests never open a graphical window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from medical_imaging_quality_analyzer.image_visualizer import visualize_image


class VisualizeImageTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_displays_generated_image_and_histogram(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "sample.png"
            sample_image = Image.new("L", (2, 2))
            sample_image.putdata([0, 64, 64, 255])
            sample_image.save(image_path)

            with patch("matplotlib.pyplot.show") as show:
                visualize_image(image_path)

        show.assert_called_once()
        figure = plt.gcf()
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(figure.axes[0].get_title(), "Grayscale Image")
        self.assertEqual(figure.axes[0].get_xlabel(), "Pixel column")
        self.assertEqual(figure.axes[0].get_ylabel(), "Pixel row")
        self.assertEqual(figure.axes[1].get_title(), "Pixel-Intensity Histogram")
        self.assertEqual(figure.axes[1].get_xlabel(), "Pixel intensity")
        self.assertEqual(figure.axes[1].get_ylabel(), "Number of pixels")

        histogram_line = figure.axes[1].lines[0]
        self.assertEqual(histogram_line.get_ydata()[0], 1)
        self.assertEqual(histogram_line.get_ydata()[64], 2)
        self.assertEqual(histogram_line.get_ydata()[255], 1)

    def test_missing_image_path_has_a_helpful_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.png"

            with self.assertRaisesRegex(ValueError, "Could not open image"):
                visualize_image(missing_path)


if __name__ == "__main__":
    unittest.main()
