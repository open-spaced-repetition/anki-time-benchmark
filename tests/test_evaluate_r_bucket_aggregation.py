import unittest

from evaluate import _aggregate_r_bucket_precision


class EvaluateRBucketAggregationTests(unittest.TestCase):
    def test_weighted_aggregation(self) -> None:
        lists = [
            [
                {
                    "bucket_start": 0.0,
                    "bucket_end": 0.05,
                    "count": 10,
                    "mean_true_sec": 8.0,
                    "mean_pred_sec": 9.0,
                    "mse_sec": 2.0,
                    "mae_sec": 1.0,
                    "precise_enough_pct": 80.0,
                    "tolerance_sec": 2.0,
                }
            ],
            [
                {
                    "bucket_start": 0.0,
                    "bucket_end": 0.05,
                    "count": 30,
                    "mean_true_sec": 10.0,
                    "mean_pred_sec": 12.0,
                    "mse_sec": 8.0,
                    "mae_sec": 2.0,
                    "precise_enough_pct": 50.0,
                    "tolerance_sec": 2.0,
                }
            ],
        ]
        out = _aggregate_r_bucket_precision(lists)
        self.assertEqual(len(out), 1)
        row = out[0]
        self.assertEqual(row["count"], 40)
        self.assertAlmostEqual(row["mean_true_sec"], 9.5, places=6)
        self.assertAlmostEqual(row["mean_pred_sec"], 11.25, places=6)
        self.assertAlmostEqual(row["rmse_sec"], (6.5 ** 0.5), places=6)
        self.assertAlmostEqual(row["mae_sec"], 1.75, places=6)
        self.assertAlmostEqual(row["precise_enough_pct"], 57.5, places=6)


if __name__ == "__main__":
    unittest.main()
