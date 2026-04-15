import unittest

from script import Config, evaluate


class ResultRegressionParametersTests(unittest.TestCase):
    def test_evaluate_includes_regression_parameters_when_save_weights_enabled(self) -> None:
        config = Config(save_weights=True)
        stats, raw = evaluate(
            y_true=[1.0, 2.0, 3.0],
            y_pred=[1.0, 2.0, 3.0],
            user_id=123,
            config=config,
            algorithm_weights_last_split=None,
            regression_parameters_last_split={"a": 1.23456789, "b": 9.87654321},
        )

        self.assertIn("regression_parameters", stats)
        self.assertEqual(stats["regression_parameters"]["a"], 1.234568)
        self.assertEqual(stats["regression_parameters"]["b"], 9.876543)
        self.assertIsNone(raw)

    def test_evaluate_includes_r_bucket_precision(self) -> None:
        config = Config(save_weights=False)
        stats, _ = evaluate(
            y_true=[1.0, 2.0],
            y_pred=[1.1, 2.2],
            user_id=7,
            config=config,
            r_bucket_precision=[{"bucket_start": 0.0, "bucket_end": 0.05, "count": 1}],
        )
        self.assertIn("r_bucket_precision", stats)
        self.assertEqual(stats["r_bucket_precision"][0]["bucket_end"], 0.05)


if __name__ == "__main__":
    unittest.main()
