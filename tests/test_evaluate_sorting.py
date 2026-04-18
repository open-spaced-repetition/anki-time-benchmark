import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluate import print_table_for_suffix


class EvaluateSortingTests(unittest.TestCase):
    def test_main_table_sorted_by_mae_ascending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result_dir = Path(td)

            method_a = {
                "user": 1,
                "size": 100,
                "metrics": {"RMSE": 20.0, "MAE": 4.0, "MAPE": 80.0},
            }
            method_b = {
                "user": 1,
                "size": 100,
                "metrics": {"RMSE": 10.0, "MAE": 6.0, "MAPE": 70.0},
            }

            (result_dir / "A_NO_FIRST_REVIEWS.jsonl").write_text(json.dumps(method_a) + "\n", encoding="utf-8")
            (result_dir / "B_NO_FIRST_REVIEWS.jsonl").write_text(json.dumps(method_b) + "\n", encoding="utf-8")

            buf = io.StringIO()
            with redirect_stdout(buf):
                print_table_for_suffix(
                    result_dir=result_dir,
                    methods_arg=["A", "B"],
                    use_default_methods=False,
                    suffix="NO_FIRST_REVIEWS",
                    weight_by="reviews",
                )

            lines = buf.getvalue().splitlines()
            method_rows = [
                line
                for line in lines
                if line.startswith("|") and "Method" not in line and "---" not in line and "±" in line
            ]
            self.assertGreaterEqual(len(method_rows), 2)
            self.assertTrue(method_rows[0].startswith("| A "))


if __name__ == "__main__":
    unittest.main()
