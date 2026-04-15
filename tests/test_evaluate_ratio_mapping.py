import unittest

from evaluate import _add_ratio_mapping_pct


class EvaluateRatioMappingTests(unittest.TestCase):
    def test_ratio_mapping_matches_example(self) -> None:
        rows = [
            {
                "bucket_start": 0.50,
                "bucket_end": 0.55,
                "mean_true_sec": 15.0,
                "mean_pred_sec": 10.0,
            },
            {
                "bucket_start": 0.85,
                "bucket_end": 0.90,
                "mean_true_sec": 7.5,
                "mean_pred_sec": 5.0,
            },
        ]
        out = _add_ratio_mapping_pct(rows, ref_bucket_start=0.85, ref_bucket_end=0.90)
        row = next(r for r in out if r["bucket_start"] == 0.50)
        self.assertAlmostEqual(row["pred_ratio_to_ref"], 2.0, places=6)
        self.assertAlmostEqual(row["actual_ratio_to_ref"], 2.0, places=6)
        self.assertAlmostEqual(row["ratio_mapping_pct"], 100.0, places=6)

    def test_ratio_mapping_penalizes_mismatch(self) -> None:
        rows = [
            {
                "bucket_start": 0.50,
                "bucket_end": 0.55,
                "mean_true_sec": 20.0,
                "mean_pred_sec": 10.0,
            },
            {
                "bucket_start": 0.85,
                "bucket_end": 0.90,
                "mean_true_sec": 10.0,
                "mean_pred_sec": 8.0,
            },
        ]
        out = _add_ratio_mapping_pct(rows, ref_bucket_start=0.85, ref_bucket_end=0.90)
        row = next(r for r in out if r["bucket_start"] == 0.50)
        # pred ratio = 1.25, actual ratio = 2.0 => score 62.5%
        self.assertAlmostEqual(row["ratio_mapping_pct"], 62.5, places=6)


if __name__ == "__main__":
    unittest.main()
