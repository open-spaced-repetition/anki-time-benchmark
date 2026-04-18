import unittest

from evaluate import _bucket_corr_summary


class EvaluateBucketCorrelationTests(unittest.TestCase):
    def test_bucket_corr_summary_negative(self) -> None:
        rows = [
            {"bucket_start": 0.80, "bucket_end": 0.85, "count": 10, "mean_true_sec": 8.0, "mean_pred_sec": 7.5},
            {"bucket_start": 0.85, "bucket_end": 0.90, "count": 10, "mean_true_sec": 7.0, "mean_pred_sec": 6.8},
            {"bucket_start": 0.90, "bucket_end": 0.95, "count": 10, "mean_true_sec": 5.0, "mean_pred_sec": 5.5},
        ]
        out = _bucket_corr_summary(rows)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertLess(out["pearson_true"], 0.0)
        self.assertLess(out["pearson_pred"], 0.0)


if __name__ == "__main__":
    unittest.main()
