import unittest

import numpy as np

from medical_imaging_quality_analyzer.frequency_analysis import analyze_frequency


class AnalyzeFrequencyTests(unittest.TestCase):
    def test_constant_image_has_only_a_centered_zero_frequency_peak(self):
        sample_image = np.full((8, 8), 3.0)

        result = analyze_frequency(sample_image)
        shifted_magnitude = np.abs(result["shifted_fft"])

        self.assertEqual(result["fft"].shape, sample_image.shape)
        self.assertEqual(result["shifted_fft"].shape, sample_image.shape)
        self.assertEqual(result["log_magnitude_spectrum"].shape, sample_image.shape)
        self.assertAlmostEqual(shifted_magnitude[4, 4], 192.0)
        shifted_magnitude[4, 4] = 0
        np.testing.assert_allclose(shifted_magnitude, 0, atol=1e-12)
        np.testing.assert_allclose(
            result["log_magnitude_spectrum"],
            np.log1p(np.abs(result["shifted_fft"])),
        )

    def test_horizontal_cosine_has_expected_frequency_peaks(self):
        size = 8
        columns = np.arange(size)
        sample_image = np.tile(np.cos(2 * np.pi * 2 * columns / size), (size, 1))

        result = analyze_frequency(sample_image)
        magnitude = np.abs(result["shifted_fft"])

        # The cosine has two horizontal frequency peaks, two bins either side
        # of the centered zero-frequency position.
        self.assertAlmostEqual(magnitude[4, 2], 32.0)
        self.assertAlmostEqual(magnitude[4, 6], 32.0)
        self.assertAlmostEqual(magnitude[4, 4], 0.0, places=10)
        self.assertTrue(np.isfinite(result["log_magnitude_spectrum"]).all())

    def test_rejects_empty_image(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            analyze_frequency(np.empty((0, 0)))

    def test_rejects_non_array_input(self):
        with self.assertRaisesRegex(TypeError, "must be a NumPy array"):
            analyze_frequency([[1, 2], [3, 4]])

    def test_rejects_non_grayscale_dimensions(self):
        with self.assertRaisesRegex(ValueError, "2D grayscale"):
            analyze_frequency(np.zeros((2, 2, 3)))

    def test_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite pixel values"):
            analyze_frequency(np.array([[0.0, np.nan], [1.0, 2.0]]))

    def test_rejects_complex_values(self):
        with self.assertRaisesRegex(TypeError, "real numeric"):
            analyze_frequency(np.array([[1 + 2j]]))


if __name__ == "__main__":
    unittest.main()
