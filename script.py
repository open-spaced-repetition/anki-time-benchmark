"""
Benchmark review-time prediction methods on Anki revlogs.

Methods:
- const
- user_median
- grade_median_4
- grade_median_4_4
- grade_median_8
- fsrs_r_linear
- fsrs_r_linear_by_grades
- fsrs_r_ridge
- fsrs_r_grade_interact
- fsrs_one_minus_r_s_reps_d_linear
- fsrs_one_minus_r_s_reps_d_linear_by_grade
- fsrs_one_minus_r_s_reps_d_ridge
- fsrs_dsr_grade_nn
- poor_mans_fsrs

Behavior for fsrs_dsr_grade_nn:
1) Pretrain once and save .pth checkpoint if it does not exist.
2) If checkpoint exists, load it and skip pretraining.
3) Then run per-user fine-tuning/evaluation.

Usage example:
python script.py --method grade_median_4 --processes 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from torch import Tensor, nn
from tqdm.auto import tqdm  # type: ignore

from fsrs_optimizer import BatchDataset, BatchLoader, DevicePrefetchLoader  # type: ignore
from fsrs_v7 import FSRS7
from moving_avg import moving_avg_seconds
from data import create_features
from review_time_nn import (
    ReviewTimeNN,
    Normalizer,
    featurize_dsrg,
    train_regressor,
    predict_seconds,
)

try:
    from scipy.optimize import curve_fit  # type: ignore
except Exception:
    curve_fit = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


METHOD_NAMES = {
    "const": "CONST",
    "user_median": "USER_MEDIAN",
    "grade_median_4": "GRADE_MEDIAN_4",
    "grade_median_4_4": "GRADE_MEDIAN_4_4",
    "grade_median_8": "GRADE_MEDIAN_8",  # identical to GRADE_MEDIAN_4 when --with_first_reviews is false
    "fsrs_r_linear": "FSRS7_R_LINEAR",
    "fsrs_r_linear_by_grades": "FSRS7_R_LINEAR_BY_GRADES",
    "fsrs_r_ridge": "FSRS7_R_RIDGE",
    "fsrs_r_grade_interact": "FSRS7_R_GRADE_INTERACT",
    "fsrs_one_minus_r_s_reps_d_linear": "FSRS7_ONE_MINUS_R_S_REPS_D_LINEAR",
    "fsrs_one_minus_r_s_reps_d_linear_by_grade": "FSRS7_ONE_MINUS_R_S_REPS_D_LINEAR_BY_GRADE",
    "fsrs_one_minus_r_s_reps_d_ridge": "FSRS7_ONE_MINUS_R_S_REPS_D_RIDGE",
    "fsrs_dsr_grade_nn": "FSRS7_DSR_GRADE_NN",
    "poor_mans_fsrs": "POOR_MANS_FSRS",
    "moving_avg": "MOVING_AVG",
}

ALL_METHODS = list(METHOD_NAMES.keys())
R_BUCKET_STEP = 0.05
R_PRECISION_TOLERANCE_SEC = 2.0


@dataclass
class Config:
    data_path: Path = Path("../anki-revlogs-10k")
    max_user_id: Optional[int] = None
    user_id: Optional[int] = None

    method: str = "const"

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    batch_size: int = 512
    n_splits: int = 5
    seed: int = 42
    default_params: bool = False
    use_recency_weighting: bool = False

    # For MOVING-AVG
    moving_avg_alpha: float = 0.3
    moving_avg_init_value: Optional[float] = None
    moving_avg_log_space: bool = True
    moving_avg_min_seconds: float = 0.05
    ridge_alpha: float = 1.0

    include_short_term: bool = True
    two_buttons: bool = False
    max_seq_len: int = 64

    nn_ckpt_path: Path = Path("checkpoints/review_time_pretrained.pth")
    nn_pretrain_users: int = 250
    nn_pretrain_epochs: int = 8
    nn_pretrain_lr: float = 1e-3
    nn_pretrain_batch_size: int = 2048
    nn_pretrain_max_samples_per_user: int = 2000
    nn_finetune_epochs: int = 40
    nn_finetune_lr: float = 3e-3
    nn_finetune_batch_size: int = 512

    train_equals_test: bool = False
    with_first_reviews: bool = False

    s_min: float = 0.0001
    init_s_max: float = 100.0

    save_evaluation_file: bool = False
    save_raw_output: bool = False
    save_weights: bool = False
    verbose_inadequate_data: bool = False

    num_processes: int = 1
    cache_fsrs_weights: bool = True
    fsrs_weights_cache_dir: Path = Path(".cache/fsrs_weights")

    def get_evaluation_file_name(self) -> str:
        suffix = "WITH_FIRST_REVIEWS" if self.with_first_reviews else "NO_FIRST_REVIEWS"
        return f"{METHOD_NAMES[self.method]}_{suffix}"


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_path=Path(args.data),
        max_user_id=args.max_user_id,
        user_id=args.user_id,
        method=args.method,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        seed=args.seed,
        default_params=args.default_params,
        use_recency_weighting=args.recency_weighting,
        save_evaluation_file=args.save_evaluation_file,
        save_raw_output=args.save_raw_output,
        save_weights=args.save_weights,
        verbose_inadequate_data=args.verbose,
        num_processes=args.processes,
        with_first_reviews=args.with_first_reviews,
        nn_ckpt_path=Path(args.nn_ckpt),
        nn_pretrain_users=args.nn_pretrain_users,
        nn_pretrain_epochs=args.nn_pretrain_epochs,
        nn_pretrain_lr=args.nn_pretrain_lr,
        nn_pretrain_batch_size=args.nn_pretrain_batch_size,
        nn_pretrain_max_samples_per_user=args.nn_pretrain_max_samples_per_user,
        nn_finetune_epochs=args.nn_finetune_epochs,
        nn_finetune_lr=args.nn_finetune_lr,
        nn_finetune_batch_size=args.nn_finetune_batch_size,
        moving_avg_alpha=args.moving_avg_alpha,
        moving_avg_init_value=args.moving_avg_init_value,
        moving_avg_log_space=(not args.moving_avg_linear_space),
        moving_avg_min_seconds=args.moving_avg_min_seconds,
        ridge_alpha=args.ridge_alpha,
        cache_fsrs_weights=(not args.no_cache_fsrs_weights),
        fsrs_weights_cache_dir=Path(args.fsrs_weights_cache_dir),
    )


def _drop_frequency_jump_tail(
    df: pd.DataFrame,
    duration_col: str = "duration",
    jump_ratio: float = 10.0,
    require_whole_seconds: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    vc = out[duration_col].value_counts(dropna=False)
    if vc.empty or len(vc) < 2:
        return out

    vc_sorted = vc.sort_values(kind="stable")
    counts = vc_sorted.to_numpy(dtype=np.int64)
    jump_pos = np.where(counts[:-1] * jump_ratio <= counts[1:])[0]
    if len(jump_pos) == 0:
        return out

    j = int(jump_pos[-1])
    tail_values = vc_sorted.index[(j + 1):]

    if require_whole_seconds:
        tail_values = [v for v in tail_values if (int(v) % 1000 == 0)]

    if len(tail_values) == 0:
        return out

    return out[~out[duration_col].isin(tail_values)].copy()


def _is_inadequate_exception(e: Exception) -> bool:
    s = str(e).lower()
    return (
        s.endswith("inadequate.")
        or "non-empty tensorlist" in s
        or "stack expects a non-empty tensorlist" in s
        or ("empty" in s and "batch" in s)
    )


def load_user_frames(user_id: int, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(config.data_path / "revlogs" / f"user_id={user_id}").copy()
    raw = raw[raw["rating"].isin([1, 2, 3, 4])].copy()

    if config.two_buttons:
        raw["rating"] = raw["rating"].replace({2: 3, 4: 3})

    raw = raw.reset_index(drop=True)
    raw["event_id"] = np.arange(1, len(raw) + 1)

    raw = raw.sort_values(by=["card_id", "event_id"]).copy()
    raw["prev_rating"] = raw.groupby("card_id")["rating"].shift(1)
    raw["i_raw"] = raw.groupby("card_id").cumcount() + 1
    raw["first_review"] = raw["i_raw"] == 1

    is_again = (raw["rating"] == 1).astype(int)
    raw["again_count_before"] = is_again.groupby(raw["card_id"]).cumsum() - is_again
    raw["again_count_before"] = raw["again_count_before"].astype(int)
    raw["total_reps_before"] = raw.groupby("card_id").cumcount().astype(int)

    raw = raw[raw["i_raw"] <= config.max_seq_len * 2].copy()

    raw_base = raw.sort_values("event_id").copy()

    algorithm_input = raw_base.copy()
    algorithm_df = create_features(algorithm_input, config)

    eval_df = raw_base.copy()
    eval_df["duration"] = pd.to_numeric(eval_df["duration"], errors="coerce")
    eval_df = eval_df.dropna(subset=["duration"]).copy()
    eval_df["duration"] = eval_df["duration"].astype("int64")

    eval_df = eval_df[(eval_df["duration"] > 0) & (eval_df["duration"] <= 1_800_000)].copy()
    eval_df = _drop_frequency_jump_tail(
        eval_df,
        duration_col="duration",
        jump_ratio=10.0,
        require_whole_seconds=True,
    )

    eval_df["duration_sec"] = (eval_df["duration"] / 1000.0).round(3)

    eval_df["interval_days"] = np.nan

    # Try interval from raw columns first
    raw_interval_candidates = [
        "delta_t",
        "elapsed_days",
        "elapsed",
        "interval",
        "ivl",
    ]
    for c in raw_interval_candidates:
        if c in eval_df.columns:
            eval_df["interval_days"] = pd.to_numeric(eval_df[c], errors="coerce")
            break

    # Fill/override with algorithm delta_t when available (usually the best source for interval)
    if "delta_t" in algorithm_df.columns and "event_id" in algorithm_df.columns:
        dt_map = dict(
            zip(
                algorithm_df["event_id"].astype(int).tolist(),
                pd.to_numeric(algorithm_df["delta_t"], errors="coerce").tolist(),
            )
        )
        mapped = eval_df["event_id"].map(dt_map)
        eval_df["interval_days"] = eval_df["interval_days"].where(eval_df["interval_days"].notna(), mapped)

    eval_df["interval_days"] = pd.to_numeric(eval_df["interval_days"], errors="coerce")
    eval_df["interval_days"] = eval_df["interval_days"].clip(lower=0)

    eval_df = eval_df.sort_values("event_id").copy()

    if len(eval_df) < 6:
        raise Exception(f"{user_id} does not have enough usable (non-censored) data.")

    return eval_df.reset_index(drop=True), algorithm_df.reset_index(drop=True)


class AlgorithmTrainer:
    def __init__(
        self,
        algorithm: FSRS7,
        train_set: pd.DataFrame,
        batch_size: int = 512,
        max_seq_len: int = 64,
    ) -> None:
        self.algorithm = algorithm
        self.batch_size = getattr(algorithm, "batch_size", batch_size)
        self.betas = getattr(algorithm, "betas", (0.9, 0.999))
        self.n_epoch = algorithm.n_epoch
        self.loss_fn = nn.BCELoss(reduction="none")

        algorithm.initialize_parameters(train_set)
        filtered = algorithm.filter_training_data(train_set)
        if filtered is None or len(filtered) == 0:
            raise Exception("Training data inadequate.")

        self.train_dataset = BatchDataset(filtered.copy(), self.batch_size, max_seq_len=max_seq_len)
        self.train_loader = BatchLoader(self.train_dataset)
        if getattr(self.train_loader, "batch_nums", 0) == 0:
            raise Exception("Training data inadequate.")

        self.optimizer = algorithm.get_optimizer(lr=algorithm.lr, wd=algorithm.wd, betas=self.betas)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_loader.batch_nums * self.n_epoch
        )

    def _batch_process(self, batch: tuple) -> dict[str, Tensor]:
        sequences, delta_ts, labels, seq_lens, weights = batch
        real_batch_size = seq_lens.shape[0]
        result = self.algorithm.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
        result["labels"] = labels
        result["weights"] = weights
        return result

    def train(self) -> list:
        best_loss = np.inf
        best_w = self.algorithm.state_dict()
        epoch_len = len(self.train_dataset.y_train)

        for _ in range(self.n_epoch):
            loss_val, w = self._eval()
            if loss_val < best_loss:
                best_loss = loss_val
                best_w = w

            for batch in self.train_loader:
                self.algorithm.train()
                self.optimizer.zero_grad()
                seq_lens = batch[3]
                if seq_lens.shape[0] == 0 or torch.max(seq_lens).item() <= 0:
                    continue
                result = self._batch_process(batch)
                loss = (self.loss_fn(result["retentions"], result["labels"]) * result["weights"]).sum()
                if "penalty" in result:
                    loss += result["penalty"] / epoch_len
                loss.backward()
                self.algorithm.apply_gradient_constraints()
                self.optimizer.step()
                self.scheduler.step()
                self.algorithm.apply_parameter_clipper()

        loss_val, w = self._eval()
        if loss_val < best_loss:
            best_w = w
        return best_w

    def _eval(self) -> tuple[float, list]:
        self.algorithm.eval()
        self.train_loader.shuffle = False
        total_loss = 0.0
        total_items = 0
        epoch_len = len(self.train_dataset.y_train)

        with torch.no_grad():
            for batch in self.train_loader:
                seq_lens = batch[3]
                if seq_lens is None or seq_lens.shape[0] == 0:
                    continue
                if torch.max(seq_lens).item() <= 0:
                    continue
                result = self._batch_process(batch)
                total_loss += ((self.loss_fn(result["retentions"], result["labels"]) * result["weights"]).sum().item())
                if "penalty" in result:
                    total_loss += (result["penalty"] / epoch_len).item()
                total_items += batch[3].shape[0]

        self.train_loader.shuffle = True
        return total_loss / max(total_items, 1), self.algorithm.state_dict()


def batch_predict_dsr(algorithm: FSRS7, dataset: pd.DataFrame, config: Config) -> tuple[list[float], list[float], list[float]]:
    if dataset is None or len(dataset) == 0:
        return [], [], []

    algorithm.eval()
    ds = BatchDataset(dataset, batch_size=8192, sort_by_length=False)
    loader = BatchLoader(ds, shuffle=False)
    dev_loader = DevicePrefetchLoader(loader, target_device=config.device)

    d_list, s_list, r_list = [], [], []
    with torch.no_grad():
        for batch in dev_loader:
            sequences, delta_ts, labels, seq_lens, weights = batch
            real_batch_size = seq_lens.shape[0]
            result = algorithm.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
            d_list.extend(result["difficulties"].cpu().tolist())
            s_list.extend(result["stabilities"].cpu().tolist())
            r_list.extend(result["retentions"].cpu().tolist())
    return d_list, s_list, r_list


def batch_predict_retention(algorithm: FSRS7, dataset: pd.DataFrame, config: Config) -> list[float]:
    if dataset is None or len(dataset) == 0:
        return []

    algorithm.eval()
    ds = BatchDataset(dataset, batch_size=8192, sort_by_length=False)
    loader = BatchLoader(ds, shuffle=False)
    dev_loader = DevicePrefetchLoader(loader, target_device=config.device)

    out = []
    with torch.no_grad():
        for batch in dev_loader:
            sequences, delta_ts, labels, seq_lens, weights = batch
            real_batch_size = seq_lens.shape[0]
            result = algorithm.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
            out.extend(result["retentions"].cpu().tolist())
    return out


def _median_by_grade(df: pd.DataFrame) -> dict[int, float]:
    return df.groupby("rating")["duration_sec"].median().to_dict()


def _fill_grade_medians(base: dict[int, float], fallback: float) -> dict[int, float]:
    return {g: float(base.get(g, fallback)) for g in [1, 2, 3, 4]}


def _duration_training_scope(train_df: pd.DataFrame, with_first_reviews: bool) -> pd.DataFrame:
    if with_first_reviews:
        return train_df

    non_first = train_df[~train_df["first_review"]].copy()
    if len(non_first) == 0:
        raise Exception("No non-first training rows; skip this split.")
    return non_first


def _predict_const7(test_df: pd.DataFrame) -> np.ndarray:
    return np.full(len(test_df), 7.0, dtype=float)


def _predict_user_median(train_df: pd.DataFrame, test_df: pd.DataFrame, with_first_reviews: bool) -> np.ndarray:
    src = _duration_training_scope(train_df, with_first_reviews=with_first_reviews)
    med = float(src["duration_sec"].median())
    return np.full(len(test_df), med, dtype=float)


def _predict_grade_median_4(train_df: pd.DataFrame, test_df: pd.DataFrame, with_first_reviews: bool) -> np.ndarray:
    src = _duration_training_scope(train_df, with_first_reviews=with_first_reviews)
    global_med = float(src["duration_sec"].median())
    med = _fill_grade_medians(_median_by_grade(src), global_med)
    return test_df["rating"].map(med).astype(float).to_numpy()


def _predict_grade_median_4_4(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    with_first_reviews: bool = False,  # kept for call compatibility; intentionally unused
) -> np.ndarray:
    """Predict by (prev_rating, current_rating) pair — 16 bins.

    Training scope: rows that have a prev_rating (i.e. not first reviews).
    Fallback for missing pairs: global median across those same rows.
    First reviews in test get the global median (they are excluded from
    metrics anyway).
    """
    src = train_df[train_df["prev_rating"].notna()].copy()
    if len(src) == 0:
        fallback = float(train_df["duration_sec"].median())
        return np.full(len(test_df), fallback, dtype=float)

    src["prev_rating"] = src["prev_rating"].astype(int)
    src["rating"] = src["rating"].astype(int)

    global_med = float(src["duration_sec"].median())
    pair_median = src.groupby(["prev_rating", "rating"])["duration_sec"].median()

    pred = np.full(len(test_df), global_med, dtype=float)
    has_prev = test_df["prev_rating"].notna().to_numpy()
    if has_prev.any():
        keys = list(
            zip(
                test_df.loc[has_prev, "prev_rating"].astype(int).to_numpy(),
                test_df.loc[has_prev, "rating"].astype(int).to_numpy(),
            )
        )
        pred_vals = np.array([float(pair_median.get(k, global_med)) for k in keys], dtype=float)
        pred[has_prev] = pred_vals
    return pred


def _predict_grade_median_8(train_df: pd.DataFrame, test_df: pd.DataFrame, with_first_reviews: bool) -> np.ndarray:
    if not with_first_reviews:
        return _predict_grade_median_4(train_df, test_df, with_first_reviews=False)

    global_med = float(train_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(train_df), global_med)

    first_df = train_df[train_df["first_review"]]
    non_first_df = train_df[~train_df["first_review"]]

    first_map_raw = first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()

    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}
    non_first_map = {g: float(non_first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    pred = []
    for _, row in test_df.iterrows():
        g = int(row["rating"])
        pred.append(first_map[g] if bool(row["first_review"]) else non_first_map[g])
    return np.array(pred, dtype=float)


def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _poor_mans_fsrs_model(
    xdata: np.ndarray,
    a0: float,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
    a5: float,
) -> np.ndarray:
    # xdata rows: [ln_agains, ln_total_reps, interval_days, grade]
    ln_agains = xdata[0]
    ln_total = xdata[1]
    interval = np.maximum(xdata[2], 0.0)
    grade = xdata[3]
    return a0 + a1 * ln_agains + a2 * ln_total + a3 * np.exp(-a4 * interval) + a5 * grade


def _predict_poor_mans_fsrs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    with_first_reviews: bool,
) -> np.ndarray:
    # Fallback baseline
    fallback = _predict_grade_median_4(train_df, test_df, with_first_reviews=with_first_reviews)
    if curve_fit is None:
        return fallback

    src = _duration_training_scope(train_df, with_first_reviews=with_first_reviews).copy()
    if len(src) < 12:
        return fallback

    if "again_count_before" not in src.columns or "total_reps_before" not in src.columns or "interval_days" not in src.columns:
        return fallback

    src["again_count_before"] = pd.to_numeric(src["again_count_before"], errors="coerce")
    src["total_reps_before"] = pd.to_numeric(src["total_reps_before"], errors="coerce")
    src["interval_days"] = pd.to_numeric(src["interval_days"], errors="coerce")
    src["rating"] = pd.to_numeric(src["rating"], errors="coerce")
    src["duration_sec"] = pd.to_numeric(src["duration_sec"], errors="coerce")

    src = src.dropna(subset=["again_count_before", "total_reps_before", "rating", "duration_sec"]).copy()
    if len(src) < 12:
        return fallback

    if src["interval_days"].notna().any():
        interval_fill = float(np.nanmedian(src["interval_days"].to_numpy(dtype=float)))
    else:
        interval_fill = 0.0
    if not np.isfinite(interval_fill):
        interval_fill = 0.0

    src["interval_days"] = src["interval_days"].fillna(interval_fill).clip(lower=0)

    ln_agains = np.log1p(np.clip(src["again_count_before"].to_numpy(dtype=float), 0, None))
    ln_total = np.log1p(np.clip(src["total_reps_before"].to_numpy(dtype=float), 0, None))
    interval = src["interval_days"].to_numpy(dtype=float)
    grade = src["rating"].to_numpy(dtype=float)
    y = src["duration_sec"].to_numpy(dtype=float)

    xdata = np.vstack([ln_agains, ln_total, interval, grade])

    y_med = float(np.median(y))
    y_std = float(np.std(y)) if len(y) > 1 else 1.0

    p0 = np.array([y_med, 0.5, -0.5, max(0.1, y_std), 0.1, 0.5], dtype=float)
    lower = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 1e-8, -np.inf], dtype=float)  # enforce a4 > 0
    upper = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=float)

    try:
        params, _ = curve_fit(
            _poor_mans_fsrs_model,
            xdata,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
    except Exception:
        return fallback

    test_tmp = test_df.copy()
    if "again_count_before" not in test_tmp.columns or "total_reps_before" not in test_tmp.columns or "interval_days" not in test_tmp.columns:
        return fallback

    test_tmp["again_count_before"] = pd.to_numeric(test_tmp["again_count_before"], errors="coerce").fillna(0).clip(lower=0)
    test_tmp["total_reps_before"] = pd.to_numeric(test_tmp["total_reps_before"], errors="coerce").fillna(0).clip(lower=0)
    test_tmp["interval_days"] = pd.to_numeric(test_tmp["interval_days"], errors="coerce").fillna(interval_fill).clip(lower=0)
    test_tmp["rating"] = pd.to_numeric(test_tmp["rating"], errors="coerce").fillna(3.0)

    x_test = np.vstack(
        [
            np.log1p(test_tmp["again_count_before"].to_numpy(dtype=float)),
            np.log1p(test_tmp["total_reps_before"].to_numpy(dtype=float)),
            test_tmp["interval_days"].to_numpy(dtype=float),
            test_tmp["rating"].to_numpy(dtype=float),
        ]
    )
    pred = _poor_mans_fsrs_model(x_test, *params)
    return pred.astype(float)


def _fit_algorithm_weights(train_algorithm_df: pd.DataFrame, config: Config) -> list:
    if len(train_algorithm_df) == 0:
        return FSRS7(config).state_dict()

    train_alg = train_algorithm_df.copy()

    if config.use_recency_weighting:
        x = np.linspace(0, 1, len(train_alg))
        train_alg["weights"] = 0.25 + 0.75 * np.power(x, 3)

    algorithm = FSRS7(config).to(config.device)

    if config.default_params:
        return algorithm.state_dict()

    try:
        trainer = AlgorithmTrainer(
            algorithm=algorithm,
            train_set=train_alg,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
        )
        return trainer.train()
    except Exception as e:
        if _is_inadequate_exception(e):
            if config.verbose_inadequate_data:
                tqdm.write("Training data inadequate; using default FSRS parameters.")
            return FSRS7(config).state_dict()
        raise


def _fit_algorithm_weights_cached(
    train_algorithm_df: pd.DataFrame,
    config: Config,
    user_id: int,
    split_key: int,
) -> list:
    if not config.cache_fsrs_weights:
        return _fit_algorithm_weights(train_algorithm_df, config)

    cache_dir = config.fsrs_weights_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    event_min = int(train_algorithm_df["event_id"].min()) if len(train_algorithm_df) > 0 else -1
    event_max = int(train_algorithm_df["event_id"].max()) if len(train_algorithm_df) > 0 else -1
    cache_payload = {
        "user_id": int(user_id),
        "split_key": int(split_key),
        "data_path": str(config.data_path),
        "default_params": bool(config.default_params),
        "use_recency_weighting": bool(config.use_recency_weighting),
        "batch_size": int(config.batch_size),
        "max_seq_len": int(config.max_seq_len),
        "seed": int(config.seed),
        "s_min": float(config.s_min),
        "init_s_max": float(config.init_s_max),
        "train_rows": int(len(train_algorithm_df)),
        "event_min": event_min,
        "event_max": event_max,
    }
    cache_key = hashlib.sha1(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    cache_file = cache_dir / f"fsrs_weights_{cache_key}.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) > 0:
                return [float(x) for x in cached]
        except Exception:
            pass

    w = _fit_algorithm_weights(train_algorithm_df, config)
    try:
        cache_file.write_text(json.dumps([float(x) for x in w]), encoding="utf-8")
    except Exception:
        pass
    return w


def _predict_DSR_maps(
    train_algorithm_df: pd.DataFrame,
    test_algorithm_df: pd.DataFrame,
    config: Config,
    user_id: int,
    split_key: int,
) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]]]:
    if len(train_algorithm_df) == 0:
        return {}, {}

    w = _fit_algorithm_weights_cached(train_algorithm_df, config, user_id=user_id, split_key=split_key)
    algorithm = FSRS7(config, w=w).to(config.device)

    train_map: dict[int, tuple[float, float, float]] = {}
    test_map: dict[int, tuple[float, float, float]] = {}

    if len(train_algorithm_df) > 0:
        d, s, r = batch_predict_dsr(algorithm, train_algorithm_df, config)
        for eid, dd, ss, rr in zip(train_algorithm_df["event_id"].tolist(), d, s, r):
            train_map[int(eid)] = (float(dd), float(ss), float(rr))

    if len(test_algorithm_df) > 0:
        d, s, r = batch_predict_dsr(algorithm, test_algorithm_df, config)
        for eid, dd, ss, rr in zip(test_algorithm_df["event_id"].tolist(), d, s, r):
            test_map[int(eid)] = (float(dd), float(ss), float(rr))

    return train_map, test_map


def _predict_R_maps(
    train_algorithm_df: pd.DataFrame,
    test_algorithm_df: pd.DataFrame,
    config: Config,
    user_id: int,
    split_key: int,
) -> tuple[dict[int, float], dict[int, float], list]:
    if len(train_algorithm_df) == 0:
        return {}, {}, FSRS7(config).state_dict()

    w = _fit_algorithm_weights_cached(train_algorithm_df, config, user_id=user_id, split_key=split_key)
    algorithm = FSRS7(config, w=w).to(config.device)

    train_map: dict[int, float] = {}
    test_map: dict[int, float] = {}

    if len(train_algorithm_df) > 0:
        r_train = batch_predict_retention(algorithm, train_algorithm_df, config)
        for eid, r in zip(train_algorithm_df["event_id"].tolist(), r_train):
            train_map[int(eid)] = float(r)

    if len(test_algorithm_df) > 0:
        r_test = batch_predict_retention(algorithm, test_algorithm_df, config)
        for eid, r in zip(test_algorithm_df["event_id"].tolist(), r_test):
            test_map[int(eid)] = float(r)

    return train_map, test_map, w


def _predict_fsrs_r_linear(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_R_map: dict[int, float],
    test_R_map: dict[int, float],
    with_first_reviews: bool,
    return_coefficients: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_df = train_eval[~train_eval["first_review"]].copy()
    non_first_df["R"] = non_first_df["event_id"].map(train_R_map)
    fit_df = non_first_df.dropna(subset=["R"]).copy()

    if len(fit_df) >= 2:
        X = np.column_stack([fit_df["R"].to_numpy(), np.ones(len(fit_df))])
        y = fit_df["duration_sec"].to_numpy()
        a, b = _fit_ols(X, y)
    else:
        a, b = 0.0, global_med

    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g: float(non_first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    pred = []
    for _, row in test_eval.iterrows():
        g = int(row["rating"])
        if bool(row["first_review"]):
            t = first_map[g]
        else:
            R = test_R_map.get(int(row["event_id"]))
            t = non_first_map[g] if R is None else float(a) * float(R) + float(b)
        pred.append(max(0.0, float(t)))
    pred_arr = np.array(pred, dtype=float)
    if return_coefficients:
        return pred_arr, {"a": float(a), "b": float(b)}
    return pred_arr


def _predict_fsrs_r_ridge(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_R_map: dict[int, float],
    test_R_map: dict[int, float],
    with_first_reviews: bool,
    ridge_alpha: float,
    return_coefficients: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_df = train_eval[~train_eval["first_review"]].copy()
    non_first_df["R"] = non_first_df["event_id"].map(train_R_map)
    fit_df = non_first_df.dropna(subset=["R"]).copy()

    if len(fit_df) >= 2:
        X = fit_df["R"].to_numpy(dtype=float).reshape(-1, 1)
        y = fit_df["duration_sec"].to_numpy(dtype=float)
        model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
        model.fit(X, y)
        a = float(model.coef_[0])
        b = float(model.intercept_)
    else:
        a, b = 0.0, global_med

    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g: float(non_first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    pred = []
    for _, row in test_eval.iterrows():
        g = int(row["rating"])
        if bool(row["first_review"]):
            t = first_map[g]
        else:
            R = test_R_map.get(int(row["event_id"]))
            t = non_first_map[g] if R is None else float(a) * float(R) + float(b)
        pred.append(max(0.0, float(t)))
    pred_arr = np.array(pred, dtype=float)
    if return_coefficients:
        return pred_arr, {"a": float(a), "b": float(b), "ridge_alpha": float(ridge_alpha)}
    return pred_arr


def _predict_fsrs_r_linear_by_grades(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_R_map: dict[int, float],
    test_R_map: dict[int, float],
    with_first_reviews: bool,
    return_coefficients: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)
    grade_labels = {1: "again", 2: "hard", 3: "good", 4: "easy"}

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_df = train_eval[~train_eval["first_review"]].copy()
    non_first_df["R"] = non_first_df["event_id"].map(train_R_map)
    fit_df = non_first_df.dropna(subset=["R"]).copy()

    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g: float(non_first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    coeffs_by_grade: dict[int, tuple[float, float]] = {}
    for g in [1, 2, 3, 4]:
        gdf = fit_df[fit_df["rating"] == g]
        if len(gdf) >= 2:
            X = np.column_stack([gdf["R"].to_numpy(dtype=float), np.ones(len(gdf))])
            y = gdf["duration_sec"].to_numpy(dtype=float)
            a, b = _fit_ols(X, y)
        else:
            a, b = 0.0, non_first_map[g]
        coeffs_by_grade[g] = (float(a), float(b))

    pred = []
    for _, row in test_eval.iterrows():
        g = int(row["rating"])
        if bool(row["first_review"]):
            t = first_map[g]
        else:
            R = test_R_map.get(int(row["event_id"]))
            if R is None:
                t = non_first_map[g]
            else:
                a_g, b_g = coeffs_by_grade[g]
                t = float(a_g) * float(R) + float(b_g)
        pred.append(max(0.0, float(t)))
    pred_arr = np.array(pred, dtype=float)
    if return_coefficients:
        flat: dict[str, float] = {}
        for g in [1, 2, 3, 4]:
            a_g, b_g = coeffs_by_grade[g]
            label = grade_labels[g]
            flat[f"{label}_a"] = float(a_g)
            flat[f"{label}_b"] = float(b_g)
        return pred_arr, flat
    return pred_arr


def _predict_fsrs_r_grade_interact(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_R_map: dict[int, float],
    test_R_map: dict[int, float],
    with_first_reviews: bool,
) -> np.ndarray:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_df = train_eval[~train_eval["first_review"]].copy()
    non_first_df["R"] = non_first_df["event_id"].map(train_R_map)
    fit_df = non_first_df.dropna(subset=["R"]).copy()

    if len(fit_df) >= 4:
        g = fit_df["rating"].to_numpy().astype(float)
        R = fit_df["R"].to_numpy().astype(float)
        X = np.column_stack([np.ones(len(fit_df)), g, R, g * R])
        y = fit_df["duration_sec"].to_numpy()
        a0, a1, a2, a3 = _fit_ols(X, y)
    else:
        a0, a1, a2, a3 = global_med, 0.0, 0.0, 0.0

    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g_: float(non_first_map_raw.get(g_, by_grade_all[g_])) for g_ in [1, 2, 3, 4]}

    pred = []
    for _, row in test_eval.iterrows():
        g = float(row["rating"])
        if bool(row["first_review"]):
            t = first_map[int(g)]
        else:
            R = test_R_map.get(int(row["event_id"]))
            if R is None:
                t = non_first_map[int(g)]
            else:
                t = float(a0) + float(a1) * g + float(a2) * float(R) + float(a3) * g * float(R)
        pred.append(max(0.0, float(t)))
    return np.array(pred, dtype=float)


def _predict_fsrs_one_minus_r_s_reps_d_linear(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_dsr_map: dict[int, tuple[float, float, float]],
    test_dsr_map: dict[int, tuple[float, float, float]],
    with_first_reviews: bool,
    return_coefficients: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_df = train_eval[~train_eval["first_review"]].copy()
    non_first_df["dsr"] = non_first_df["event_id"].map(train_dsr_map)
    fit_df = non_first_df.dropna(subset=["dsr"]).copy()

    if len(fit_df) >= 5:
        R = np.array([x[2] for x in fit_df["dsr"].tolist()], dtype=float)
        S = np.array([x[1] for x in fit_df["dsr"].tolist()], dtype=float)
        D = np.array([x[0] for x in fit_df["dsr"].tolist()], dtype=float)
        reps = pd.to_numeric(fit_df["total_reps_before"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = fit_df["duration_sec"].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(fit_df)), 1.0 - R, S, reps, D])
        a, b, c, d, e = _fit_ols(X, y)
    else:
        a, b, c, d, e = global_med, 0.0, 0.0, 0.0, 0.0

    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g_: float(non_first_map_raw.get(g_, by_grade_all[g_])) for g_ in [1, 2, 3, 4]}

    pred = []
    for _, row in test_eval.iterrows():
        g = int(row["rating"])
        if bool(row["first_review"]):
            t = first_map[g]
        else:
            dsr = test_dsr_map.get(int(row["event_id"]))
            if dsr is None:
                t = non_first_map[g]
            else:
                D, S, R = float(dsr[0]), float(dsr[1]), float(dsr[2])
                reps = float(pd.to_numeric(row.get("total_reps_before", 0.0), errors="coerce"))
                if not np.isfinite(reps):
                    reps = 0.0
                t = float(a) + float(b) * (1.0 - R) + float(c) * S + float(d) * reps + float(e) * D
        pred.append(max(0.0, float(t)))
    pred_arr = np.array(pred, dtype=float)
    if return_coefficients:
        return pred_arr, {"a": float(a), "b": float(b), "c": float(c), "d": float(d), "e": float(e)}
    return pred_arr


def _predict_fsrs_one_minus_r_s_reps_d_ridge(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_dsr_map: dict[int, tuple[float, float, float]],
    test_dsr_map: dict[int, tuple[float, float, float]],
    with_first_reviews: bool,
    ridge_alpha: float,
    return_coefficients: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_df = train_eval[~train_eval["first_review"]].copy()
    non_first_df["dsr"] = non_first_df["event_id"].map(train_dsr_map)
    fit_df = non_first_df.dropna(subset=["dsr"]).copy()

    if len(fit_df) >= 5:
        R = np.array([x[2] for x in fit_df["dsr"].tolist()], dtype=float)
        S = np.array([x[1] for x in fit_df["dsr"].tolist()], dtype=float)
        D = np.array([x[0] for x in fit_df["dsr"].tolist()], dtype=float)
        reps = pd.to_numeric(fit_df["total_reps_before"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = fit_df["duration_sec"].to_numpy(dtype=float)
        X = np.column_stack([1.0 - R, S, reps, D])
        model = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
        model.fit(X, y)
        b, c, d, e = [float(x) for x in model.coef_.tolist()]
        a = float(model.intercept_)
    else:
        a, b, c, d, e = global_med, 0.0, 0.0, 0.0, 0.0

    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g_: float(non_first_map_raw.get(g_, by_grade_all[g_])) for g_ in [1, 2, 3, 4]}

    pred = []
    for _, row in test_eval.iterrows():
        g = int(row["rating"])
        if bool(row["first_review"]):
            t = first_map[g]
        else:
            dsr = test_dsr_map.get(int(row["event_id"]))
            if dsr is None:
                t = non_first_map[g]
            else:
                D_, S_, R_ = float(dsr[0]), float(dsr[1]), float(dsr[2])
                reps = float(pd.to_numeric(row.get("total_reps_before", 0.0), errors="coerce"))
                if not np.isfinite(reps):
                    reps = 0.0
                t = float(a) + float(b) * (1.0 - R_) + float(c) * S_ + float(d) * reps + float(e) * D_
        pred.append(max(0.0, float(t)))
    pred_arr = np.array(pred, dtype=float)
    if return_coefficients:
        return pred_arr, {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "e": float(e),
            "ridge_alpha": float(ridge_alpha),
        }
    return pred_arr


def _predict_fsrs_one_minus_r_s_reps_d_linear_by_grade(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_dsr_map: dict[int, tuple[float, float, float]],
    test_dsr_map: dict[int, tuple[float, float, float]],
    with_first_reviews: bool,
    return_coefficients: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)
    grade_labels = {1: "again", 2: "hard", 3: "good", 4: "easy"}

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_df = train_eval[~train_eval["first_review"]].copy()
    non_first_df["dsr"] = non_first_df["event_id"].map(train_dsr_map)
    fit_df = non_first_df.dropna(subset=["dsr"]).copy()

    non_first_map_raw = non_first_df.groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g: float(non_first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    coeffs_by_grade: dict[int, tuple[float, float, float, float, float]] = {}
    for g in [1, 2, 3, 4]:
        gdf = fit_df[fit_df["rating"] == g].copy()
        if len(gdf) >= 5:
            R = np.array([x[2] for x in gdf["dsr"].tolist()], dtype=float)
            S = np.array([x[1] for x in gdf["dsr"].tolist()], dtype=float)
            D = np.array([x[0] for x in gdf["dsr"].tolist()], dtype=float)
            reps = pd.to_numeric(gdf["total_reps_before"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            y = gdf["duration_sec"].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(gdf)), 1.0 - R, S, reps, D])
            a, b, c, d, e = _fit_ols(X, y)
        else:
            a, b, c, d, e = non_first_map[g], 0.0, 0.0, 0.0, 0.0
        coeffs_by_grade[g] = (float(a), float(b), float(c), float(d), float(e))

    pred = []
    for _, row in test_eval.iterrows():
        g = int(row["rating"])
        if bool(row["first_review"]):
            t = first_map[g]
        else:
            dsr = test_dsr_map.get(int(row["event_id"]))
            if dsr is None:
                t = non_first_map[g]
            else:
                D_, S_, R_ = float(dsr[0]), float(dsr[1]), float(dsr[2])
                reps = float(pd.to_numeric(row.get("total_reps_before", 0.0), errors="coerce"))
                if not np.isfinite(reps):
                    reps = 0.0
                a_g, b_g, c_g, d_g, e_g = coeffs_by_grade[g]
                t = float(a_g) + float(b_g) * (1.0 - R_) + float(c_g) * S_ + float(d_g) * reps + float(e_g) * D_
        pred.append(max(0.0, float(t)))

    pred_arr = np.array(pred, dtype=float)
    if return_coefficients:
        flat: dict[str, float] = {}
        for g in [1, 2, 3, 4]:
            label = grade_labels[g]
            a_g, b_g, c_g, d_g, e_g = coeffs_by_grade[g]
            flat[f"{label}_a"] = float(a_g)
            flat[f"{label}_b"] = float(b_g)
            flat[f"{label}_c"] = float(c_g)
            flat[f"{label}_d"] = float(d_g)
            flat[f"{label}_e"] = float(e_g)
        return pred_arr, flat
    return pred_arr


def _list_user_ids(data_path: Path, max_user_id: Optional[int], user_id: Optional[int]) -> list[int]:
    ids = []
    for p in (data_path / "revlogs").glob("user_id=*"):
        try:
            ids.append(int(p.name.split("=")[1]))
        except Exception:
            continue
    ids = sorted(set(ids))
    if user_id is not None:
        ids = [u for u in ids if u == user_id]
    elif max_user_id is not None:
        ids = [u for u in ids if u <= max_user_id]
    return ids


def _resolve_unprocessed_user_ids(config: Config, processed_users: set[int]) -> list[int]:
    all_user_ids = _list_user_ids(config.data_path, config.max_user_id, config.user_id)
    return [uid for uid in all_user_ids if uid not in processed_users]


def _build_pretrain_nn_state(config: Config) -> dict:
    user_ids = _list_user_ids(config.data_path, config.max_user_id, config.user_id)[: config.nn_pretrain_users]

    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    for uid in tqdm(user_ids, desc="NN pretrain users", leave=False):
        try:
            eval_df, algorithm_df = load_user_frames(uid, config)
            if len(algorithm_df) == 0:
                continue

            w = _fit_algorithm_weights(algorithm_df, config)
            algo = FSRS7(config, w=w).to(config.device)
            d, s, r = batch_predict_dsr(algo, algorithm_df, config)

            dsr_map: dict[int, tuple[float, float, float]] = {}
            for eid, dd, ss, rr in zip(algorithm_df["event_id"].tolist(), d, s, r):
                dsr_map[int(eid)] = (float(dd), float(ss), float(rr))

            rows = eval_df[~eval_df["first_review"]].copy()
            rows["dsr"] = rows["event_id"].map(dsr_map)
            rows = rows.dropna(subset=["dsr"])
            if len(rows) == 0:
                continue

            if len(rows) > config.nn_pretrain_max_samples_per_user:
                rows = rows.sample(n=config.nn_pretrain_max_samples_per_user, random_state=config.seed)

            dd = np.array([x[0] for x in rows["dsr"].tolist()], dtype=np.float32)
            ss = np.array([x[1] for x in rows["dsr"].tolist()], dtype=np.float32)
            rr = np.array([x[2] for x in rows["dsr"].tolist()], dtype=np.float32)
            gg = rows["rating"].to_numpy(dtype=np.float32)
            tt = rows["duration_sec"].to_numpy(dtype=np.float32)

            X_all.append(featurize_dsrg(dd, ss, rr, gg))
            y_all.append(np.log(np.clip(tt, 0.05, None)))
        except Exception:
            continue

    if len(X_all) == 0:
        raise RuntimeError("NN pretraining failed: no usable samples.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    norm = Normalizer.fit(X)
    Xn = norm.transform(X)

    model = ReviewTimeNN()
    train_regressor(
        model,
        Xn,
        y,
        device=config.device,
        lr=config.nn_pretrain_lr,
        weight_decay=1e-5,
        epochs=config.nn_pretrain_epochs,
        batch_size=config.nn_pretrain_batch_size,
        train_head_only=False,
    )

    return {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "norm_mean": norm.mean,
        "norm_std": norm.std,
    }


def _load_or_pretrain_nn_state(config: Config) -> dict:
    ckpt = config.nn_ckpt_path
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if ckpt.exists():
        tqdm.write(f"Found pretrained checkpoint: {ckpt}")
        try:
            # PyTorch >=2.6 defaults to weights_only=True, but this checkpoint also
            # stores numpy arrays (normalizer stats), so we need full unpickling.
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            # Compatibility for older PyTorch versions without `weights_only`.
            state = torch.load(ckpt, map_location="cpu")
        if not isinstance(state, dict) or "state_dict" not in state or "norm_mean" not in state or "norm_std" not in state:
            raise RuntimeError(f"Checkpoint format invalid: {ckpt}")
        tqdm.write("Loaded pretrained NN; skipping pretraining.")
        return state

    tqdm.write(f"No checkpoint found at {ckpt}. Starting pretraining...")
    state = _build_pretrain_nn_state(config)
    torch.save(state, ckpt)
    tqdm.write(f"Saved pretrained checkpoint to: {ckpt}")
    return state


def _predict_fsrs_dsr_grade_nn(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_dsr_map: dict[int, tuple[float, float, float]],
    test_dsr_map: dict[int, tuple[float, float, float]],
    nn_state: dict,
    config: Config,
    with_first_reviews: bool,
) -> np.ndarray:
    scope_df = _duration_training_scope(train_eval, with_first_reviews=with_first_reviews)
    global_med = float(scope_df["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(scope_df), global_med)

    first_map_raw = train_eval[train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    first_map = {g: float(first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    non_first_map_raw = train_eval[~train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
    non_first_map = {g: float(non_first_map_raw.get(g, by_grade_all[g])) for g in [1, 2, 3, 4]}

    model = ReviewTimeNN()
    model.load_state_dict(nn_state["state_dict"])
    norm = Normalizer(
        mean=np.array(nn_state["norm_mean"], dtype=np.float32),
        std=np.array(nn_state["norm_std"], dtype=np.float32),
    )

    ft_rows = train_eval[~train_eval["first_review"]].copy()
    ft_rows["dsr"] = ft_rows["event_id"].map(train_dsr_map)
    ft_rows = ft_rows.dropna(subset=["dsr"])

    if len(ft_rows) >= 16:
        dd = np.array([x[0] for x in ft_rows["dsr"].tolist()], dtype=np.float32)
        ss = np.array([x[1] for x in ft_rows["dsr"].tolist()], dtype=np.float32)
        rr = np.array([x[2] for x in ft_rows["dsr"].tolist()], dtype=np.float32)
        gg = ft_rows["rating"].to_numpy(dtype=np.float32)
        tt = ft_rows["duration_sec"].to_numpy(dtype=np.float32)

        X = featurize_dsrg(dd, ss, rr, gg)
        X = norm.transform(X)
        y = np.log(np.clip(tt, 0.05, None))

        train_regressor(
            model,
            X,
            y,
            device=config.device,
            lr=config.nn_finetune_lr,
            weight_decay=0.0,
            epochs=config.nn_finetune_epochs,
            batch_size=config.nn_finetune_batch_size,
            train_head_only=True,
        )

    pred: list[float] = []
    for _, row in test_eval.iterrows():
        g = int(row["rating"])
        if bool(row["first_review"]):
            pred.append(first_map[g])
            continue

        dsr = test_dsr_map.get(int(row["event_id"]))
        if dsr is None:
            pred.append(non_first_map[g])
            continue

        x = featurize_dsrg(
            np.array([dsr[0]], dtype=np.float32),
            np.array([dsr[1]], dtype=np.float32),
            np.array([dsr[2]], dtype=np.float32),
            np.array([g], dtype=np.float32),
        )
        x = norm.transform(x)
        t = float(predict_seconds(model, x, device=config.device, min_sec=0.05, max_sec=1800.0)[0])
        pred.append(t)

    return np.array(pred, dtype=float)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else None
    return {"MAE": round(mae, 6), "RMSE": round(rmse, 6), "MAPE": None if mape is None else round(mape, 6)}


def _compute_r_bucket_precision(
    r_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bucket_step: float = R_BUCKET_STEP,
    tolerance_sec: float = R_PRECISION_TOLERANCE_SEC,
) -> list[dict]:
    r = np.asarray(r_values, dtype=float)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(r) & np.isfinite(y) & np.isfinite(p)
    r = r[mask]
    y = y[mask]
    p = p[mask]
    if len(r) == 0:
        return []

    edges = np.arange(0.0, 1.0 + bucket_step + 1e-12, bucket_step)
    edges[-1] = 1.0
    out: list[dict] = []

    abs_err = np.abs(p - y)

    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == len(edges) - 2:
            in_bucket = (r >= lo) & (r <= hi)
        else:
            in_bucket = (r >= lo) & (r < hi)
        if not np.any(in_bucket):
            continue

        bucket_err = abs_err[in_bucket]
        bucket_sq_err = np.square(p[in_bucket] - y[in_bucket])
        bucket_true = y[in_bucket]
        bucket_pred = p[in_bucket]
        precise_pct = float(np.mean(bucket_err <= tolerance_sec) * 100.0)
        out.append(
            {
                "bucket_start": round(lo, 2),
                "bucket_end": round(hi, 2),
                "count": int(np.sum(in_bucket)),
                "mean_true_sec": round(float(np.mean(bucket_true)), 6),
                "mean_pred_sec": round(float(np.mean(bucket_pred)), 6),
                "mse_sec": round(float(np.mean(bucket_sq_err)), 6),
                "rmse_sec": round(float(np.sqrt(np.mean(bucket_sq_err))), 6),
                "mae_sec": round(float(np.mean(bucket_err)), 6),
                "precise_enough_pct": round(precise_pct, 6),
                "tolerance_sec": float(tolerance_sec),
            }
        )
    return out


def evaluate(
    y_true: list[float],
    y_pred: list[float],
    user_id: int,
    config: Config,
    algorithm_weights_last_split: Optional[list] = None,
    regression_parameters_last_split: Optional[dict[str, float]] = None,
    r_bucket_precision: Optional[list[dict]] = None,
) -> tuple[dict, Optional[dict]]:
    y = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)

    stats: dict = {
        "metrics": _compute_metrics(y, p),
        "user": int(user_id),
        "size": int(len(y_true)),
    }

    if config.save_weights and algorithm_weights_last_split is not None:
        if isinstance(algorithm_weights_last_split, list):
            stats["algorithm_parameters"] = [round(float(x), 6) for x in algorithm_weights_last_split]
        else:
            stats["algorithm_parameters"] = algorithm_weights_last_split

    if config.save_weights and regression_parameters_last_split is not None:
        stats["regression_parameters"] = {
            k: round(float(v), 6) for k, v in regression_parameters_last_split.items()
        }

    if r_bucket_precision is not None:
        stats["r_bucket_precision"] = r_bucket_precision

    raw: Optional[dict] = None
    if config.save_raw_output:
        raw = {
            "user": int(user_id),
            "t_pred": [round(float(v), 4) for v in y_pred],
            "t_true": [round(float(v), 4) for v in y_true],
        }

    return stats, raw


def save_evaluation_file(user_id: int, df: pd.DataFrame, config: Config) -> None:
    if config.save_evaluation_file:
        df.to_csv(f"evaluation/{config.get_evaluation_file_name()}/{user_id}.tsv", sep="\t", index=False)


def sort_jsonl(file: Path) -> list:
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("", encoding="utf-8")
        return []

    lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = [json.loads(line) for line in lines]
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8", newline="\n") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return data


def _catch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception:
            user_id = args[0] if args else kwargs.get("user_id")
            msg = traceback.format_exc()
            if user_id is not None:
                msg = f"User {user_id}:\n{msg}"
            return None, msg

    return wrapper


@_catch
def process(user_id: int, config: Config, nn_state: Optional[dict] = None) -> tuple[dict, Optional[dict]]:
    eval_df, algorithm_df = load_user_frames(user_id, config)

    if config.method == "moving_avg":
        _, _, save_df = moving_avg_seconds(
            dataset=eval_df,
            n_splits=config.n_splits,
            target_col="duration_sec",
            alpha=config.moving_avg_alpha,
            init_value=config.moving_avg_init_value,
            log_space=config.moving_avg_log_space,
            min_seconds=config.moving_avg_min_seconds,
        )

        save_df = save_df.copy()
        save_df["t_true"] = pd.to_numeric(save_df["duration_sec"], errors="coerce")
        save_df["t_pred"] = pd.to_numeric(save_df["t_pred"], errors="coerce")
        save_df = save_df.dropna(subset=["t_true", "t_pred"]).copy()

        if not config.with_first_reviews and "first_review" in save_df.columns:
            save_df["used_for_metrics"] = ~save_df["first_review"]
            metric_df = save_df[~save_df["first_review"]].copy()
        else:
            save_df["used_for_metrics"] = True
            metric_df = save_df.copy()

        if len(metric_df) == 0:
            raise Exception(f"{user_id}: no evaluation rows for moving_avg after filtering.")

        # Build a reference R-map so moving_avg can also report R-bucket precision.
        _, test_R_map, _ = _predict_R_maps(
            train_algorithm_df=algorithm_df,
            test_algorithm_df=algorithm_df,
            config=config,
            user_id=user_id,
            split_key=int(eval_df["event_id"].max()) if len(eval_df) > 0 else -1,
        )
        metric_df = metric_df.copy()
        metric_df["R"] = metric_df["event_id"].map(test_R_map)
        r_bucket_precision = _compute_r_bucket_precision(
            r_values=pd.to_numeric(metric_df["R"], errors="coerce").to_numpy(dtype=float),
            y_true=metric_df["t_true"].astype(float).to_numpy(dtype=float),
            y_pred=metric_df["t_pred"].astype(float).to_numpy(dtype=float),
        )

        save_evaluation_file(user_id, save_df, config)
        return evaluate(
            metric_df["t_true"].astype(float).tolist(),
            metric_df["t_pred"].astype(float).tolist(),
            user_id,
            config,
            None,
            r_bucket_precision=r_bucket_precision,
        )

    if not config.with_first_reviews and (~eval_df["first_review"]).sum() == 0:
        raise Exception(f"{user_id}: only first reviews; skipping user.")

    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    y_all: list[float] = []
    p_all: list[float] = []
    save_tmp: list[pd.DataFrame] = []
    last_algorithm_weights: Optional[list] = None
    last_regression_parameters: Optional[dict[str, float]] = None
    last_r_bucket_precision: Optional[list[dict]] = None

    method_always_excludes_first = config.method == "grade_median_4_4"

    for _, (train_idx, test_idx) in enumerate(tscv.split(eval_df)):
        if not config.train_equals_test:
            train_eval = eval_df.iloc[train_idx].copy()
            test_eval = eval_df.iloc[test_idx].copy()
        else:
            train_eval = eval_df.copy()
            test_eval = eval_df.iloc[test_idx].copy()

        if len(train_eval) == 0 or len(test_eval) == 0:
            continue

        if not config.with_first_reviews and (~train_eval["first_review"]).sum() == 0:
            continue

        first_test_event = int(test_eval["event_id"].min())
        train_algorithm_df = algorithm_df[algorithm_df["event_id"] < first_test_event].copy()

        test_event_ids = set(test_eval["event_id"].tolist())
        test_algorithm_df = algorithm_df[algorithm_df["event_id"].isin(test_event_ids)].copy()
        train_R_map, test_R_map, r_weights = _predict_R_maps(
            train_algorithm_df,
            test_algorithm_df,
            config,
            user_id=user_id,
            split_key=first_test_event,
        )
        test_eval = test_eval.copy()
        test_eval["R"] = test_eval["event_id"].map(test_R_map)

        if config.method == "const":
            pred = _predict_const7(test_eval)

        elif config.method == "user_median":
            pred = _predict_user_median(train_eval, test_eval, with_first_reviews=config.with_first_reviews)

        elif config.method == "grade_median_4":
            pred = _predict_grade_median_4(train_eval, test_eval, with_first_reviews=config.with_first_reviews)

        elif config.method == "grade_median_4_4":
            pred = _predict_grade_median_4_4(train_eval, test_eval, with_first_reviews=config.with_first_reviews)

        elif config.method == "grade_median_8":
            pred = _predict_grade_median_8(train_eval, test_eval, with_first_reviews=config.with_first_reviews)

        elif config.method == "poor_mans_fsrs":
            pred = _predict_poor_mans_fsrs(
                train_df=train_eval,
                test_df=test_eval,
                with_first_reviews=config.with_first_reviews,
            )

        elif config.method in ("fsrs_r_linear", "fsrs_r_linear_by_grades", "fsrs_r_ridge", "fsrs_r_grade_interact"):
            last_algorithm_weights = r_weights if r_weights else last_algorithm_weights
            if config.method == "fsrs_r_linear":
                linear_out = _predict_fsrs_r_linear(
                    train_eval,
                    test_eval,
                    train_R_map,
                    test_R_map,
                    with_first_reviews=config.with_first_reviews,
                    return_coefficients=config.save_weights,
                )
                if config.save_weights:
                    pred, coeffs = linear_out  # type: ignore[assignment]
                    last_regression_parameters = coeffs
                else:
                    pred = linear_out  # type: ignore[assignment]
            elif config.method == "fsrs_r_linear_by_grades":
                linear_out = _predict_fsrs_r_linear_by_grades(
                    train_eval,
                    test_eval,
                    train_R_map,
                    test_R_map,
                    with_first_reviews=config.with_first_reviews,
                    return_coefficients=config.save_weights,
                )
                if config.save_weights:
                    pred, coeffs = linear_out  # type: ignore[assignment]
                    last_regression_parameters = coeffs
                else:
                    pred = linear_out  # type: ignore[assignment]
            elif config.method == "fsrs_r_ridge":
                linear_out = _predict_fsrs_r_ridge(
                    train_eval,
                    test_eval,
                    train_R_map,
                    test_R_map,
                    with_first_reviews=config.with_first_reviews,
                    ridge_alpha=config.ridge_alpha,
                    return_coefficients=config.save_weights,
                )
                if config.save_weights:
                    pred, coeffs = linear_out  # type: ignore[assignment]
                    last_regression_parameters = coeffs
                else:
                    pred = linear_out  # type: ignore[assignment]
            else:
                pred = _predict_fsrs_r_grade_interact(
                    train_eval,
                    test_eval,
                    train_R_map,
                    test_R_map,
                    with_first_reviews=config.with_first_reviews,
                )

        elif config.method in (
            "fsrs_one_minus_r_s_reps_d_linear",
            "fsrs_one_minus_r_s_reps_d_linear_by_grade",
            "fsrs_one_minus_r_s_reps_d_ridge",
        ):
            train_dsr_map, test_dsr_map = _predict_DSR_maps(
                train_algorithm_df,
                test_algorithm_df,
                config,
                user_id=user_id,
                split_key=first_test_event,
            )
            if config.method == "fsrs_one_minus_r_s_reps_d_linear":
                linear_out = _predict_fsrs_one_minus_r_s_reps_d_linear(
                    train_eval=train_eval,
                    test_eval=test_eval,
                    train_dsr_map=train_dsr_map,
                    test_dsr_map=test_dsr_map,
                    with_first_reviews=config.with_first_reviews,
                    return_coefficients=config.save_weights,
                )
            elif config.method == "fsrs_one_minus_r_s_reps_d_linear_by_grade":
                linear_out = _predict_fsrs_one_minus_r_s_reps_d_linear_by_grade(
                    train_eval=train_eval,
                    test_eval=test_eval,
                    train_dsr_map=train_dsr_map,
                    test_dsr_map=test_dsr_map,
                    with_first_reviews=config.with_first_reviews,
                    return_coefficients=config.save_weights,
                )
            else:
                linear_out = _predict_fsrs_one_minus_r_s_reps_d_ridge(
                    train_eval=train_eval,
                    test_eval=test_eval,
                    train_dsr_map=train_dsr_map,
                    test_dsr_map=test_dsr_map,
                    with_first_reviews=config.with_first_reviews,
                    ridge_alpha=config.ridge_alpha,
                    return_coefficients=config.save_weights,
                )
            if config.save_weights:
                pred, coeffs = linear_out  # type: ignore[assignment]
                last_regression_parameters = coeffs
            else:
                pred = linear_out  # type: ignore[assignment]

        elif config.method == "fsrs_dsr_grade_nn":
            if nn_state is None:
                raise RuntimeError("NN state is required for fsrs_dsr_grade_nn but is None.")
            train_dsr_map, test_dsr_map = _predict_DSR_maps(
                train_algorithm_df,
                test_algorithm_df,
                config,
                user_id=user_id,
                split_key=first_test_event,
            )
            pred = _predict_fsrs_dsr_grade_nn(
                train_eval=train_eval,
                test_eval=test_eval,
                train_dsr_map=train_dsr_map,
                test_dsr_map=test_dsr_map,
                nn_state=nn_state,
                config=config,
                with_first_reviews=config.with_first_reviews,
            )

        else:
            raise ValueError(f"Unknown method: {config.method}")

        pred = np.maximum(pred.astype(float), 0.0)
        out_df = test_eval.copy()
        out_df["t_true"] = out_df["duration_sec"].astype(float)
        out_df["t_pred"] = pred.astype(float)

        if method_always_excludes_first:
            out_df["used_for_metrics"] = ~out_df["first_review"]
            metric_df = out_df[~out_df["first_review"]].copy()
        elif config.with_first_reviews:
            out_df["used_for_metrics"] = True
            metric_df = out_df.copy()
        else:
            out_df["used_for_metrics"] = ~out_df["first_review"]
            metric_df = out_df[~out_df["first_review"]].copy()

        if len(metric_df) > 0:
            y_all.extend(metric_df["t_true"].tolist())
            p_all.extend(metric_df["t_pred"].tolist())

        save_tmp.append(out_df)

        if config.train_equals_test:
            break

    if len(save_tmp) == 0:
        raise Exception(f"{user_id}: no valid splits after filtering.")

    if len(y_all) == 0:
        raise Exception(f"{user_id}: no evaluation rows after filtering by with_first_reviews={config.with_first_reviews}.")

    save_df = pd.concat(save_tmp, ignore_index=True)
    metric_rows = save_df[save_df["used_for_metrics"]].copy()
    if "R" in metric_rows.columns:
        last_r_bucket_precision = _compute_r_bucket_precision(
            r_values=pd.to_numeric(metric_rows["R"], errors="coerce").to_numpy(dtype=float),
            y_true=metric_rows["t_true"].to_numpy(dtype=float),
            y_pred=metric_rows["t_pred"].to_numpy(dtype=float),
        )
    save_evaluation_file(user_id, save_df, config)
    return evaluate(
        y_all,
        p_all,
        user_id,
        config,
        algorithm_weights_last_split=last_algorithm_weights,
        regression_parameters_last_split=last_regression_parameters,
        r_bucket_precision=last_r_bucket_precision,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review-time benchmark")
    p.add_argument("--data", default="../anki-revlogs-10k")
    p.add_argument("--processes", type=int, default=1)
    p.add_argument("--all-methods", action="store_true")
    p.add_argument("--no-cache-fsrs-weights", action="store_true")
    p.add_argument("--fsrs-weights-cache-dir", default=".cache/fsrs_weights")

    p.add_argument(
        "--method",
        default="const",
        choices=[
            "const",
            "user_median",
            "grade_median_4",
            "grade_median_4_4",
            "grade_median_8",
            "poor_mans_fsrs",
            "moving_avg",
            "fsrs_r_linear",
            "fsrs_r_linear_by_grades",
            "fsrs_r_ridge",
            "fsrs_r_grade_interact",
            "fsrs_one_minus_r_s_reps_d_linear",
            "fsrs_one_minus_r_s_reps_d_linear_by_grade",
            "fsrs_one_minus_r_s_reps_d_ridge",
            "fsrs_dsr_grade_nn",
        ],
    )

    p.add_argument("--moving_avg_alpha", type=float, default=0.3)
    p.add_argument("--moving_avg_init_value", type=float, default=None)
    p.add_argument("--moving_avg_linear_space", action="store_true")
    p.add_argument("--moving_avg_min_seconds", type=float, default=0.05)
    p.add_argument("--ridge-alpha", dest="ridge_alpha", type=float, default=1.0)

    p.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default-params", dest="default_params", action="store_true")
    p.add_argument("--recency-weighting", dest="recency_weighting", action="store_true")
    p.add_argument("--with_first_reviews", action="store_true")
    p.add_argument("--save-evaluation-file", dest="save_evaluation_file", action="store_true")
    p.add_argument("--save-raw", dest="save_raw_output", action="store_true")
    p.add_argument("--save-weights", dest="save_weights", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--max-user-id", dest="max_user_id", type=int, default=None)
    p.add_argument("--user-id", dest="user_id", type=int, default=None)

    p.add_argument("--nn_ckpt", type=str, default="checkpoints/review_time_pretrained.pth")
    p.add_argument("--nn_pretrain_users", type=int, default=250)
    p.add_argument("--nn_pretrain_epochs", type=int, default=8)
    p.add_argument("--nn_pretrain_lr", type=float, default=1e-3)
    p.add_argument("--nn_pretrain_batch_size", type=int, default=2048)
    p.add_argument("--nn_pretrain_max_samples_per_user", type=int, default=2000)
    p.add_argument("--nn_finetune_epochs", type=int, default=40)
    p.add_argument("--nn_finetune_lr", type=float, default=3e-3)
    p.add_argument("--nn_finetune_batch_size", type=int, default=512)

    return p.parse_args()


def _run_single_method(config: Config) -> None:
    file_name = config.get_evaluation_file_name()

    Path(f"evaluation/{file_name}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)

    result_file = Path(f"result/{file_name}.jsonl")
    raw_file = Path(f"raw/{file_name}.jsonl")

    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.touch(exist_ok=True)
    if config.save_raw_output:
        raw_file.parent.mkdir(parents=True, exist_ok=True)
        raw_file.touch(exist_ok=True)

    processed_users: set[int] = set()
    if result_file.exists():
        processed_users = {d["user"] for d in sort_jsonl(result_file)}
    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)

    unprocessed = _resolve_unprocessed_user_ids(config, processed_users)

    global_nn_state: Optional[dict] = None
    if config.method == "fsrs_dsr_grade_nn":
        global_nn_state = _load_or_pretrain_nn_state(config)

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [executor.submit(process, uid, config, global_nn_state) for uid in unprocessed]
        pbar = tqdm(as_completed(futures), total=len(futures), smoothing=0.03)

        for future in pbar:
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                    continue

                stats, raw = result
                with open(result_file, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(stats, ensure_ascii=False) + "\n")

                if raw:
                    with open(raw_file, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(raw, ensure_ascii=False) + "\n")

                pbar.set_description(f"Processed user {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    sort_jsonl(result_file)
    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)


def _resolve_methods_to_run(method: str, all_methods: bool) -> list[str]:
    return ALL_METHODS.copy() if all_methods else [method]


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = _parse_args()
    base_config = build_config(args)
    torch.manual_seed(base_config.seed)

    for method in _resolve_methods_to_run(base_config.method, args.all_methods):
        config = replace(base_config, method=method)
        _run_single_method(config)


if __name__ == "__main__":
    main()
