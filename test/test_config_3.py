import unittest

import dla_common as common

CFG_ID = 3


class TestConfig3(unittest.TestCase):
    bundle = None

    @classmethod
    def setUpClass(cls):
        cls.bundle = common.run_and_save(CFG_ID)

    def test_completion(self):
        m = self.bundle["metrics"]
        self.assertGreater(m["completion_rate"], 0.60)

    def test_single_cluster(self):
        m = self.bundle["metrics"]
        self.assertEqual(m["n_aggregate"], 1)

    def test_fractal_dimension(self):
        m = self.bundle["metrics"]
        self.assertIsNotNone(m["fractal_dimension"])
        self.assertTrue(1.55 <= m["fractal_dimension"] <= 1.90,
                        f"D_f={m['fractal_dimension']} out of expected range")


if __name__ == "__main__":
    common.run_and_save(CFG_ID)
