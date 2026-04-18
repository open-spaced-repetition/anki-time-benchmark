import unittest
from unittest import mock

import numpy as np
import pandas as pd

from script import _fit_linear, _fit_ols, _predict_fsrs_r_linear


class LinearFitLossTests(unittest.TestCase):
    def test_fit_linear_mse_matches_ols(self) -> None:
        X = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ],
            dtype=float,
        )
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

        got = _fit_linear(X, y, loss="mse")
        expected = _fit_ols(X, y)
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)

    def test_fit_linear_mae_not_worse_than_ols_on_mae_objective(self) -> None:
        X = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [1.0, 5.0],
            ],
            dtype=float,
        )
        y = np.array([1.0, 1.3, 1.6, 1.9, 2.2, 12.0], dtype=float)

        ols = _fit_ols(X, y)
        mae_fit = _fit_linear(X, y, loss="mae")

        ols_mae = float(np.mean(np.abs((X @ ols) - y)))
        mae_obj = float(np.mean(np.abs((X @ mae_fit) - y)))
        self.assertLessEqual(mae_obj, ols_mae + 1e-9)

    def test_fsrs_r_linear_passes_linear_loss_to_fit(self) -> None:
        train_eval = pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4],
                "rating": [3, 3, 3, 3],
                "first_review": [False, False, False, False],
                "duration_sec": [10.0, 9.0, 8.0, 7.0],
            }
        )
        test_eval = pd.DataFrame(
            {
                "event_id": [5],
                "rating": [3],
                "first_review": [False],
                "duration_sec": [0.0],
            }
        )
        train_r = {1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}
        test_r = {5: 0.8}

        with mock.patch("script._fit_linear", wraps=_fit_linear) as fit_linear_mock:
            _predict_fsrs_r_linear(
                train_eval=train_eval,
                test_eval=test_eval,
                train_R_map=train_r,
                test_R_map=test_r,
                with_first_reviews=False,
                linear_loss="mae",
            )

        self.assertGreaterEqual(len(fit_linear_mock.call_args_list), 1)
        self.assertTrue(any(call.kwargs.get("loss") == "mae" for call in fit_linear_mock.call_args_list))


if __name__ == "__main__":
    unittest.main()
