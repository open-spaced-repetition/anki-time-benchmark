import unittest

from calibration_plots import build_ordered_methods, parse_method_mae, parse_methods_data, safe_name


SAMPLE = """
| Method         | RMSE         | MAE         | MAPE          |
| -------------- | ------------ | ----------- | ------------- |
| FSRS7_R_LINEAR | 9.70±0.00 s  | 5.95±0.00 s | 123.94%±0.00% |
| CONST          | 10.03±0.00 s | 5.28±0.00 s | 97.28%±0.00%  |

Method: CONST (Within 2s uses |pred-true| <= 2.0s)
| R bucket  | Count | Mean true (s) | Mean pred (s) | RMSE (s) |
| --------- | ----- | ------------- | ------------- | -------- |
| 0.25-0.30 | 9     | 8.7818        | 7.0000        | 6.5221   |
| 0.30-0.35 | 1296  | 15.8337       | N/A           | N/A      |

Method: FSRS7_R_LINEAR (Within 2s uses |pred-true| <= 2.0s)
| R bucket  | Count | Mean true (s) | Mean pred (s) | RMSE (s) |
| --------- | ----- | ------------- | ------------- | -------- |
| 0.25-0.30 | 9     | 8.7818        | 21.6645       | 10.0334  |
| 0.30-0.35 | 1296  | 15.8337       | 20.4699       | 13.6881  |
"""


class CalibrationParserTests(unittest.TestCase):
    def test_parse_methods_data(self) -> None:
        out = parse_methods_data(SAMPLE)
        self.assertEqual(set(out.keys()), {"CONST", "FSRS7_R_LINEAR"})
        self.assertEqual(out["CONST"], [(8.7818, 7.0)])
        self.assertEqual(out["FSRS7_R_LINEAR"][0], (8.7818, 21.6645))
        self.assertEqual(len(out["FSRS7_R_LINEAR"]), 2)

    def test_safe_name(self) -> None:
        self.assertEqual(safe_name("FSRS7 R/Linear"), "FSRS7_R_Linear")

    def test_parse_method_mae(self) -> None:
        maes = parse_method_mae(SAMPLE)
        self.assertEqual(maes["CONST"], 5.28)
        self.assertEqual(maes["FSRS7_R_LINEAR"], 5.95)

    def test_build_ordered_methods_sorts_by_mae(self) -> None:
        methods = parse_methods_data(SAMPLE)
        maes = parse_method_mae(SAMPLE)
        ordered = build_ordered_methods(methods, maes)
        self.assertEqual([m for m, _, _, _ in ordered], ["CONST", "FSRS7_R_LINEAR"])
        self.assertEqual(ordered[0][3], 1)
        self.assertEqual(ordered[1][3], 2)


if __name__ == "__main__":
    unittest.main()
