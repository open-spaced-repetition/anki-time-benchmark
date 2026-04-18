import io
import unittest
from contextlib import redirect_stdout

from evaluate import _print_aligned_markdown_table


class EvaluateTableAlignmentTests(unittest.TestCase):
    def test_table_rows_have_consistent_width(self) -> None:
        headers = ["Method", "RMSE", "MAE", "MAPE"]
        rows = [
            ["CONST", "10.03±0.00 s", "5.28±0.00 s", "97.28%±0.00%"],
            ["FSRS7_R_LINEAR", "9.70±0.00 s", "5.95±0.00 s", "123.94%±0.00%"],
        ]

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_aligned_markdown_table(headers, rows)

        lines = [line for line in buf.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(len(line) == len(lines[0]) for line in lines))


if __name__ == "__main__":
    unittest.main()
