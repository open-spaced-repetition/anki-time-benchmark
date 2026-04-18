import unittest

import numpy as np

from script import _compute_r_bucket_precision


class RBucketPrecisionTests(unittest.TestCase):
    def test_bucket_precision_5pct(self) -> None:
        r = np.array([0.01, 0.03, 0.07, 0.99], dtype=float)
        y = np.array([10.0, 10.0, 10.0, 10.0], dtype=float)
        p = np.array([11.0, 13.0, 9.5, 13.0], dtype=float)

        out = _compute_r_bucket_precision(r, y, p, bucket_step=0.05, tolerance_sec=2.0)

        b0 = next(x for x in out if x["bucket_start"] == 0.0 and x["bucket_end"] == 0.05)
        self.assertEqual(b0["count"], 2)
        self.assertAlmostEqual(b0["mean_true_sec"], 10.0, places=6)
        self.assertAlmostEqual(b0["mean_pred_sec"], 12.0, places=6)
        self.assertAlmostEqual(b0["mse_sec"], 5.0, places=6)
        self.assertAlmostEqual(b0["rmse_sec"], np.sqrt(5.0), places=6)
        self.assertAlmostEqual(b0["mae_sec"], 2.0, places=6)
        self.assertAlmostEqual(b0["precise_enough_pct"], 50.0, places=6)

        blast = next(x for x in out if x["bucket_start"] == 0.95 and x["bucket_end"] == 1.0)
        self.assertEqual(blast["count"], 1)
        self.assertAlmostEqual(blast["precise_enough_pct"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
