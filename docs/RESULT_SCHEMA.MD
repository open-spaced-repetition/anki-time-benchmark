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
  - `fsrs_one_minus_r_s_reps_d_linear`: `a`, `b`, `c`, `d`, `e` in
    `t = a + b*(1-R) + c*S + d*reps + e*D`

For `fsrs_r_linear`:

- `r_bucket_precision` (array): prediction quality by R bucket (5% buckets)
  - `bucket_start` (float)
  - `bucket_end` (float)
  - `count` (int)
  - `mae_sec` (float)
  - `precise_enough_pct` (float), where precise enough means `|pred-true| <= 2.0s`
  - `tolerance_sec` (float)
