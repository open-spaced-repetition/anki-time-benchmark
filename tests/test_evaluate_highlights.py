import unittest

from evaluate import _format_highlights


class EvaluateHighlightsTests(unittest.TestCase):
    def test_no_single_winner_message(self) -> None:
        rows = [
            ("A", 1.0, 2.0, 3.0, "1.0", "2.0", "3.0"),
            ("B", 2.0, 1.0, 2.0, "2.0", "1.0", "2.0"),
            ("C", 3.0, 3.0, 1.0, "3.0", "3.0", "1.0"),
        ]
        lines = _format_highlights(rows)
        self.assertIn("Best RMSE: A", lines[0])
        self.assertIn("Best MAE: B", lines[1])
        self.assertIn("Best MAPE: C", lines[2])
        self.assertIn("no single method", lines[3])

    def test_single_winner_message(self) -> None:
        rows = [
            ("A", 1.0, 1.0, 1.0, "1.0", "1.0", "1.0"),
            ("B", 2.0, 2.0, 2.0, "2.0", "2.0", "2.0"),
        ]
        lines = _format_highlights(rows)
        self.assertIn("Best RMSE: A", lines[0])
        self.assertIn("Best MAE: A", lines[1])
        self.assertIn("Best MAPE: A", lines[2])
        self.assertIn("A is best on RMSE, MAE, and MAPE", lines[3])


if __name__ == "__main__":
    unittest.main()
