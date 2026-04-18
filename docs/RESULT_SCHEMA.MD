# Result JSONL Schema

Per-user results are written to:

- `result/<METHOD>_<SUFFIX>.jsonl`

Each line contains:

- `user` (int): user id
- `size` (int): number of evaluation reviews used for metrics
- `metrics` (object):
  - `RMSE` (float)
  - `MAE` (float)
  - `MAPE` (float)

When `--save-weights` is enabled:

- `algorithm_parameters` (array[float]): FSRS model parameters (for methods that fit FSRS weights)
- `regression_parameters` (object): linear regression coefficients for linear methods
  - `fsrs_r_linear`: `a`, `b` in `t = a*R + b`
  - `fsrs_r_linear_by_grades`: per-grade coefficients (`again_*`, `hard_*`, `good_*`, `easy_*`)
  - `fsrs_r_ridge`: `a`, `b`, `ridge_alpha` in `t = a*R + b` with Ridge
  - `fsrs_one_minus_r_s_reps_d_linear`: `a`, `b`, `c`, `d`, `e` in
    `t = a + b*(1-R) + c*S + d*reps + e*D`
  - `fsrs_one_minus_r_s_reps_d_linear_by_grade`: per-grade coefficients
    (`again_*`, `hard_*`, `good_*`, `easy_*`) for the same formula
  - `fsrs_one_minus_r_s_reps_d_ridge`: `a`, `b`, `c`, `d`, `e`, `ridge_alpha`

For all methods:

- `r_bucket_precision` (array): prediction quality by R bucket (5% buckets)
  - `bucket_start` (float)
  - `bucket_end` (float)
  - `count` (int)
  - `mean_true_sec` (float): average true review time in the bucket
  - `mean_pred_sec` (float): average predicted review time in the bucket
  - `mse_sec` (float): average squared error in the bucket
  - `rmse_sec` (float): root mean squared error in the bucket
  - `mae_sec` (float)
  - `precise_enough_pct` (float), where precise enough means `|pred-true| <= 2.0s`
  - `tolerance_sec` (float)
