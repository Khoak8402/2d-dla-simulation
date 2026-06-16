import unittest

import dla_common as common

CFG_ID = 2


class TestConfig2(unittest.TestCase):
    bundle = None

    @classmethod
    def setUpClass(cls):
        cls.bundle = common.run_and_save(CFG_ID)

    def test_completion(self):
        m = self.bundle["metrics"]
        self.assertGreater(m["completion_rate"], 0.80)

    def test_multiple_clusters(self):
        m = self.bundle["metrics"]
        self.assertGreater(m["n_aggregate"], 1)

    def test_df_not_measured(self):
        m = self.bundle["metrics"]
        self.assertIsNone(m["fractal_dimension"])


if __name__ == "__main__":
    common.run_and_save(CFG_ID)
