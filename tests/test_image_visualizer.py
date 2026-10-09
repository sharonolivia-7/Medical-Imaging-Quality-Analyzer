import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

# Select a non-interactive backend so tests never open a graphical window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from medical_imaging_quality_analyzer.image_visualizer import (
    visualize_image,
    visualize_image_array,
)


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

    def test_displays_signed_array_using_its_values_and_normal_grayscale(self):
        sample_image = np.array([[-100.0, 0.0], [50.0, 100.0]])

        with patch("matplotlib.pyplot.show"):
            visualize_image_array(sample_image)

        figure = plt.gcf()
        displayed = figure.axes[0].images[0]
        np.testing.assert_array_equal(displayed.get_array(), sample_image)
        self.assertEqual(displayed.get_cmap().name, "gray")
        self.assertEqual(figure.axes[0].get_title(), "Grayscale Image")
        self.assertEqual(figure.axes[1].get_title(), "Pixel-Intensity Histogram")

    def test_displays_monochrome1_using_reversed_grayscale(self):
        sample_image = np.array([[0.0, 100.0]])

        with patch("matplotlib.pyplot.show"):
            visualize_image_array(
                sample_image,
                photometric_interpretation="MONOCHROME1",
            )

        self.assertEqual(plt.gcf().axes[0].images[0].get_cmap().name, "gray_r")

    def test_array_display_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            visualize_image_array(np.empty((0, 0)))
        with self.assertRaisesRegex(ValueError, "2D grayscale"):
            visualize_image_array(np.zeros((2, 2, 3)))
        with self.assertRaisesRegex(ValueError, "finite pixel values"):
            visualize_image_array(np.array([[np.nan]]))


if __name__ == "__main__":
    unittest.main()
