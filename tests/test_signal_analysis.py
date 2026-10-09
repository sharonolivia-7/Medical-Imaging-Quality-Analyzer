import unittest

import numpy as np

from medical_imaging_quality_analyzer.signal_analysis import analyze_signal


class AnalyzeSignalTests(unittest.TestCase):
    def test_calculates_statistics_and_estimated_snr(self):
        sample_image = np.array([[10, 20], [30, 40]], dtype=np.uint8)

        result = analyze_signal(sample_image)

        expected_mean = 25.0
        expected_standard_deviation = float(np.std(sample_image))
        self.assertAlmostEqual(result["mean_intensity"], expected_mean)
        self.assertAlmostEqual(
            result["standard_deviation"], expected_standard_deviation
        )
        self.assertAlmostEqual(
            result["estimated_snr"], expected_mean / expected_standard_deviation
        )

    def test_rejects_an_empty_array(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            analyze_signal(np.array([], dtype=np.uint8).reshape(0, 0))

    def test_rejects_non_array_input(self):
        with self.assertRaisesRegex(TypeError, "must be a NumPy array"):
            analyze_signal([[1, 2], [3, 4]])

    def test_rejects_non_finite_pixel_values(self):
        sample_image = np.array([[0.0, 1.0], [2.0, np.nan]])

        with self.assertRaisesRegex(ValueError, "finite pixel values"):
            analyze_signal(sample_image)


if __name__ == "__main__":
    unittest.main()
