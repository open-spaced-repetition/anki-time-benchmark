import unittest

import numpy as np
import pandas as pd

from script import (
    _predict_fsrs_r_linear_by_grades,
    _predict_fsrs_one_minus_r_s_reps_d_linear_by_grade,
)


class LinearByGradeMethodTests(unittest.TestCase):
    def test_fsrs_r_linear_by_grades_runs_and_returns_coefficients(self) -> None:
        train = pd.DataFrame(
            {
                "event_id": [1, 2, 3, 4, 5, 6, 7, 8],
                "rating": [1, 1, 2, 2, 3, 3, 4, 4],
                "first_review": [False] * 8,
                "duration_sec": [12.0, 11.0, 10.0, 9.8, 8.5, 8.2, 7.5, 7.2],
            }
        )
        test = pd.DataFrame(
            {
                "event_id": [9, 10, 11, 12],
                "rating": [1, 2, 3, 4],
                "first_review": [False, False, False, False],
                "duration_sec": [0.0, 0.0, 0.0, 0.0],
            }
        )
        train_r = {1: 0.4, 2: 0.5, 3: 0.5, 4: 0.6, 5: 0.7, 6: 0.8, 7: 0.8, 8: 0.9}
        test_r = {9: 0.45, 10: 0.55, 11: 0.75, 12: 0.85}

        pred, coeffs = _predict_fsrs_r_linear_by_grades(
            train_eval=train,
            test_eval=test,
            train_R_map=train_r,
            test_R_map=test_r,
            with_first_reviews=False,
            return_coefficients=True,
        )
        self.assertEqual(pred.shape[0], 4)
        self.assertTrue(np.isfinite(pred).all())
        for k in ["again_a", "again_b", "hard_a", "hard_b", "good_a", "good_b", "easy_a", "easy_b"]:
            self.assertIn(k, coeffs)

    def test_fsrs_one_minus_r_s_reps_d_linear_by_grade_runs(self) -> None:
        train = pd.DataFrame(
            {
                "event_id": list(range(1, 17)),
                "rating": [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4,
                "first_review": [False] * 16,
                "duration_sec": [14, 13, 12, 11, 12, 11, 10, 9.5, 10, 9, 8.5, 8, 9, 8.5, 7.5, 7],
                "total_reps_before": list(range(1, 17)),
            }
        )
        train_dsr = {}
        for eid in range(1, 17):
            train_dsr[eid] = (max(1.0, 4.0 - 0.1 * eid), 1.5 + 0.1 * eid, min(0.98, 0.3 + 0.04 * eid))
        test = pd.DataFrame(
            {
                "event_id": [17, 18, 19, 20],
                "rating": [1, 2, 3, 4],
                "first_review": [False, False, False, False],
                "duration_sec": [0.0, 0.0, 0.0, 0.0],
                "total_reps_before": [17, 18, 19, 20],
            }
        )
        test_dsr = {
            17: (2.2, 3.2, 0.82),
            18: (2.0, 3.3, 0.85),
            19: (1.8, 3.4, 0.88),
            20: (1.6, 3.5, 0.90),
        }

        pred, coeffs = _predict_fsrs_one_minus_r_s_reps_d_linear_by_grade(
            train_eval=train,
            test_eval=test,
            train_dsr_map=train_dsr,
            test_dsr_map=test_dsr,
            with_first_reviews=False,
            return_coefficients=True,
        )
        self.assertEqual(pred.shape[0], 4)
        self.assertTrue(np.isfinite(pred).all())
        for k in ["again_a", "hard_a", "good_a", "easy_a"]:
            self.assertIn(k, coeffs)


if __name__ == "__main__":
    unittest.main()
