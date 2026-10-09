import unittest

import numpy as np

from medical_imaging_quality_analyzer.image_processing import apply_gaussian_filter


class ApplyGaussianFilterTests(unittest.TestCase):
    def test_filters_synthetic_image_and_preserves_dimensions_and_valid_values(self):
        sample_image = np.zeros((7, 7), dtype=np.uint8)
        sample_image[3, 3] = 255

        filtered_image = apply_gaussian_filter(sample_image)

        self.assertIsInstance(filtered_image, np.ndarray)
        self.assertEqual(filtered_image.shape, sample_image.shape)
        self.assertTrue(np.issubdtype(filtered_image.dtype, np.floating))
        self.assertTrue(np.isfinite(filtered_image).all())
        self.assertGreaterEqual(filtered_image.min(), sample_image.min())
        self.assertLessEqual(filtered_image.max(), sample_image.max())
        self.assertGreater(filtered_image[3, 3], filtered_image[3, 2])
        self.assertLess(filtered_image[3, 3], sample_image[3, 3])

    def test_rejects_an_empty_array(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            apply_gaussian_filter(np.empty((0, 0), dtype=np.uint8))

    def test_rejects_non_array_input(self):
        with self.assertRaisesRegex(TypeError, "must be a NumPy array"):
            apply_gaussian_filter([[1, 2], [3, 4]])

    def test_rejects_non_grayscale_dimensions(self):
        with self.assertRaisesRegex(ValueError, "2D grayscale"):
            apply_gaussian_filter(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_rejects_non_finite_pixel_values(self):
        sample_image = np.array([[0.0, 1.0], [2.0, np.inf]])

        with self.assertRaisesRegex(ValueError, "finite pixel values"):
            apply_gaussian_filter(sample_image)


if __name__ == "__main__":
    unittest.main()
