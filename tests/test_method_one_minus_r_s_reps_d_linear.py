import unittest

import numpy as np
import pandas as pd

from script import _predict_fsrs_one_minus_r_s_reps_d_linear


class OneMinusRSRepsDLinearTests(unittest.TestCase):
    def test_predicts_from_linear_formula(self) -> None:
        # t = a + b*(1-R) + c*S + d*reps + e*D

        train_eval = pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4, 5, 6],
                "rating": [3, 3, 3, 3, 3, 3],
                "first_review": [False, False, False, False, False, False],
                "duration_sec": [0.0] * 6,
                "total_reps_before": [1, 2, 3, 4, 5, 6],
            }
        )
        train_dsr_map = {
            1: (1.2, 3.1, 0.92),
            2: (2.7, 1.4, 0.81),
            3: (3.4, 2.9, 0.73),
            4: (0.8, 4.2, 0.61),
            5: (2.1, 1.7, 0.54),
            6: (3.9, 3.6, 0.43),
        }
        a, b, c, d, e = 2.0, 4.0, 1.5, 0.3, 0.8

        for i, row in train_eval.iterrows():
            D, S, R = train_dsr_map[int(row["event_id"])]
            reps = float(row["total_reps_before"])
            train_eval.loc[i, "duration_sec"] = a + b * (1.0 - R) + c * S + d * reps + e * D

        test_eval = pd.DataFrame(
            {
                "event_id": [101, 102],
                "rating": [3, 3],
                "first_review": [False, False],
                "duration_sec": [0.0, 0.0],
                "total_reps_before": [7, 8],
            }
        )
        test_dsr_map = {
            101: (2.2, 3.3, 0.65),
            102: (1.4, 4.2, 0.55),
        }

        train_R = np.array([train_dsr_map[i][2] for i in train_eval["event_id"]], dtype=float)
        train_S = np.array([train_dsr_map[i][1] for i in train_eval["event_id"]], dtype=float)
        train_D = np.array([train_dsr_map[i][0] for i in train_eval["event_id"]], dtype=float)
        train_reps = train_eval["total_reps_before"].to_numpy(dtype=float)
        train_y = train_eval["duration_sec"].to_numpy(dtype=float)
        X_train = np.column_stack([np.ones(len(train_eval)), 1.0 - train_R, train_S, train_reps, train_D])
        coef, _, _, _ = np.linalg.lstsq(X_train, train_y, rcond=None)

        out = _predict_fsrs_one_minus_r_s_reps_d_linear(
            train_eval=train_eval,
            test_eval=test_eval,
            train_dsr_map=train_dsr_map,
            test_dsr_map=test_dsr_map,
            with_first_reviews=False,
            return_coefficients=True,
        )
        pred, coeffs = out

        X_test = np.array(
            [
                [1.0, 1.0 - 0.65, 3.3, 7.0, 2.2],
                [1.0, 1.0 - 0.55, 4.2, 8.0, 1.4],
            ],
            dtype=float,
        )
        expected = X_test @ coef
        np.testing.assert_allclose(pred, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(set(coeffs.keys()), {"a", "b", "c", "d", "e"})

    def test_fallback_to_non_first_grade_median_when_dsr_missing(self) -> None:
        train_eval = pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4, 5],
                "rating": [3, 3, 3, 3, 3],
                "first_review": [False, False, False, False, False],
                "duration_sec": [6.0, 8.0, 10.0, 12.0, 14.0],
                "total_reps_before": [1, 2, 3, 4, 5],
            }
        )
        train_dsr_map = {
            1: (2.0, 3.0, 0.9),
            2: (3.0, 2.0, 0.8),
            3: (4.0, 1.0, 0.7),
            4: (1.0, 5.0, 0.6),
            5: (2.5, 2.5, 0.5),
        }
        test_eval = pd.DataFrame(
            {
                "event_id": [100],
                "rating": [3],
                "first_review": [False],
                "duration_sec": [0.0],
                "total_reps_before": [6],
            }
        )
        test_dsr_map = {}

        pred = _predict_fsrs_one_minus_r_s_reps_d_linear(
            train_eval=train_eval,
            test_eval=test_eval,
            train_dsr_map=train_dsr_map,
            test_dsr_map=test_dsr_map,
            with_first_reviews=False,
        )

        self.assertEqual(float(pred[0]), float(np.median([6.0, 8.0, 10.0, 12.0, 14.0])))


if __name__ == "__main__":
    unittest.main()
