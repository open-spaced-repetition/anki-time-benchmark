import json
import tempfile
import unittest
from pathlib import Path

from calibration_plots import build_from_result_dir, build_ordered_methods


class CalibrationResultDirTests(unittest.TestCase):
    def test_build_from_result_dir_reads_suffix_and_sorts_by_mae(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            a_rows = [
                {
                    "user": 1,
                    "size": 100,
                    "metrics": {"MAE": 3.0, "RMSE": 4.0, "MAPE": 50.0},
                    "r_bucket_precision": [
                        {
                            "bucket_start": 0.80,
                            "bucket_end": 0.85,
                            "count": 10,
                            "mean_true_sec": 8.0,
                            "mean_pred_sec": 9.0,
                            "mse_sec": 1.0,
                            "mae_sec": 1.0,
                            "precise_enough_pct": 10.0,
                        }
                    ],
                }
            ]
            b_rows = [
                {
                    "user": 1,
                    "size": 100,
                    "metrics": {"MAE": 6.0, "RMSE": 7.0, "MAPE": 60.0},
                    "r_bucket_precision": [
                        {
                            "bucket_start": 0.80,
                            "bucket_end": 0.85,
                            "count": 10,
                            "mean_true_sec": 8.0,
                            "mean_pred_sec": 7.0,
                            "mse_sec": 1.0,
                            "mae_sec": 1.0,
                            "precise_enough_pct": 10.0,
                        }
                    ],
                }
            ]

            (root / "A_NO_FIRST_REVIEWS.jsonl").write_text("\n".join(json.dumps(x) for x in a_rows) + "\n", encoding="utf-8")
            (root / "B_NO_FIRST_REVIEWS.jsonl").write_text("\n".join(json.dumps(x) for x in b_rows) + "\n", encoding="utf-8")

            methods_data, maes = build_from_result_dir(
                result_dir=root,
                suffix="NO_FIRST_REVIEWS",
                methods_arg=["A", "B"],
            )

            self.assertEqual(set(methods_data.keys()), {"A", "B"})
            self.assertAlmostEqual(maes["A"], 3.0, places=6)
            self.assertAlmostEqual(maes["B"], 6.0, places=6)

            ordered = build_ordered_methods(methods_data, maes)
            self.assertEqual([m for m, _, _, _ in ordered], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
