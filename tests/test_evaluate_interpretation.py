import unittest

from evaluate import _metric_interpretation_line


class EvaluateInterpretationTests(unittest.TestCase):
    def test_interpretation_line_mentions_units_and_example(self) -> None:
        line = _metric_interpretation_line()
        self.assertIn("10s MAE", line)
        self.assertIn("10s RMSE", line)
        self.assertIn("MAPE is percentage", line)
        self.assertIn("true time is 20s", line)


if __name__ == "__main__":
    unittest.main()
