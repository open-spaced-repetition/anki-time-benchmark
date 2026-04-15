import unittest

import numpy as np
import pandas as pd

from script import _predict_fsrs_r_ridge, _predict_fsrs_one_minus_r_s_reps_d_ridge


class RidgeMethodTests(unittest.TestCase):
    def test_fsrs_r_ridge_returns_coefficients(self) -> None:
        train = pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4],
                "rating": [3, 3, 3, 3],
                "first_review": [False, False, False, False],
                "duration_sec": [10.0, 9.0, 8.0, 7.0],
            }
        )
        test = pd.DataFrame(
            {
                "event_id": [5, 6],
                "rating": [3, 3],
                "first_review": [False, False],
                "duration_sec": [0.0, 0.0],
            }
        )
        train_r = {1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9}
        test_r = {5: 0.65, 6: 0.95}

        pred, coeffs = _predict_fsrs_r_ridge(
            train_eval=train,
            test_eval=test,
            train_R_map=train_r,
            test_R_map=test_r,
            with_first_reviews=False,
            ridge_alpha=1.0,
            return_coefficients=True,
        )

        self.assertEqual(pred.shape[0], 2)
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(set(coeffs.keys()), {"a", "b", "ridge_alpha"})

    def test_fsrs_one_minus_r_s_reps_d_ridge_returns_coefficients(self) -> None:
        train = pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4, 5, 6],
                "rating": [3, 3, 3, 3, 3, 3],
                "first_review": [False, False, False, False, False, False],
                "duration_sec": [12.0, 10.0, 9.0, 8.5, 7.8, 7.2],
                "total_reps_before": [1, 2, 3, 4, 5, 6],
            }
        )
        test = pd.DataFrame(
            {
                "event_id": [7, 8],
                "rating": [3, 3],
                "first_review": [False, False],
                "duration_sec": [0.0, 0.0],
                "total_reps_before": [7, 8],
            }
        )
        train_dsr = {
            1: (3.0, 2.0, 0.60),
            2: (2.5, 2.5, 0.65),
            3: (2.2, 2.8, 0.70),
            4: (2.0, 3.0, 0.75),
            5: (1.8, 3.2, 0.80),
            6: (1.6, 3.4, 0.85),
        }
        test_dsr = {
            7: (1.5, 3.5, 0.88),
            8: (1.4, 3.6, 0.90),
        }

        pred, coeffs = _predict_fsrs_one_minus_r_s_reps_d_ridge(
            train_eval=train,
            test_eval=test,
            train_dsr_map=train_dsr,
            test_dsr_map=test_dsr,
            with_first_reviews=False,
            ridge_alpha=1.0,
            return_coefficients=True,
        )

        self.assertEqual(pred.shape[0], 2)
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(set(coeffs.keys()), {"a", "b", "c", "d", "e", "ridge_alpha"})


if __name__ == "__main__":
    unittest.main()
