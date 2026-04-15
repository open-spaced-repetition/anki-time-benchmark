from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore


def moving_avg_seconds(
    dataset: pd.DataFrame,
    n_splits: int = 5,
    *,
    target_col: str = "duration_sec",
    alpha: float = 0.3,
    init_value: Optional[float] = None,
    log_space: bool = True,
    min_seconds: float = 0.05,
) -> tuple[list[float], list[float], pd.DataFrame]:
    """
    MOVING-AVG adapted for seconds (unbounded positive target).

    Why this adaptation:
      - Recall probability is naturally in [0, 1].
      - Review time in seconds has no natural upper bound.
      - We therefore average in log-seconds by default:
            z <- (1 - alpha) * z + alpha * log(y)
            y_pred = exp(z)
        This keeps predictions positive and avoids forcing an arbitrary max cap.

    Returns:
      y_true, y_pred, evaluated_subset_df_with_t_pred
    """
    if target_col not in dataset.columns:
        raise ValueError(f"Missing required column: {target_col}")

    data = dataset.reset_index(drop=True).copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col]).reset_index(drop=True)

    if len(data) < n_splits + 1:
        raise ValueError("Not enough rows for TimeSeriesSplit.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    save_tmp: list[pd.DataFrame] = []

    first_test_index = int(1e9)
    for _, test_index in tscv.split(data):
        first_test_index = min(first_test_index, int(test_index.min()))
        save_tmp.append(data.iloc[test_index].copy())

    observed = np.clip(data[target_col].to_numpy(dtype=float), min_seconds, None)

    if init_value is None:
        init_slice = observed[:first_test_index] if first_test_index > 0 else observed[:1]
        init_sec = float(np.median(init_slice)) if len(init_slice) else float(np.median(observed))
    else:
        init_sec = float(max(init_value, min_seconds))

    if log_space:
        state = float(np.log(init_sec))
    else:
        state = float(init_sec)

    y_true: list[float] = []
    y_pred: list[float] = []

    for i in range(len(observed)):
        pred = float(np.exp(state) if log_space else state)

        if i >= first_test_index:
            y_true.append(float(observed[i]))
            y_pred.append(max(min_seconds, pred))

        if log_space:
            y_i = float(np.log(observed[i]))
        else:
            y_i = float(observed[i])

        state = (1.0 - alpha) * state + alpha * y_i

    save_tmp_df = pd.concat(save_tmp).reset_index(drop=True)
    save_tmp_df["t_pred"] = y_pred
    return y_true, y_pred, save_tmp_df