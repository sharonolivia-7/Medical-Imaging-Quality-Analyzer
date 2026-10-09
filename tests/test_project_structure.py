import unittest

from medical_imaging_quality_analyzer import __version__


class ProjectStructureTests(unittest.TestCase):
    def test_package_exposes_version(self):
        self.assertEqual(__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
