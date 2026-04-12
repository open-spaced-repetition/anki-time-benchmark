"""
Benchmark review-time prediction methods on Anki revlogs.

Supported methods (select via --method):
- const                   : global constant 7s
- user_median             : user median from train split
- grade_median_4          : per-grade median (Again/Hard/Good/Easy)
- grade_median_8          : 8 medians (first/non-first x 4 grades)
- fsrs_r_linear           : first->first-grade median, else t=b+aR
- fsrs_r_grade_interact   : first->first-grade median, else t=a0+a1*g+a2*R+a3*g*R

Notes:
- duration in parquet is milliseconds, converted to seconds with 3 decimals.
- censored durations are excluded from train/test evaluation targets by iteratively
  finding the mode and removing it when it is exactly xxx.000 seconds (ms % 1000 == 0).
- zero/non-positive duration rows are dropped from evaluation targets.
- FSRS feature generation/training uses unfiltered-by-duration review history.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from torch import Tensor, nn
from tqdm.auto import tqdm  # type: ignore

from fsrs_optimizer import BatchDataset, BatchLoader, DevicePrefetchLoader  # type: ignore
from fsrs_v7 import FSRS7
from data import create_features

warnings.filterwarnings("ignore", category=UserWarning)


# ── configuration ─────────────────────────────────────────────────────────────


METHOD_NAMES = {
    "const": "CONST",
    "user_median": "USER_MEDIAN",
    "grade_median_4": "GRADE_MEDIAN_4",
    "grade_median_8": "GRADE_MEDIAN_8",
    "fsrs_r_linear": "FSRS7_R_LINEAR",
    "fsrs_r_grade_interact": "FSRS7_R_GRADE_INTERACT",
}


@dataclass
class Config:
    # Data
    data_path: Path = Path("../anki-revlogs-10k")
    max_user_id: Optional[int] = None

    # Selected benchmark method
    method: str = "const"

    # Algorithm (FSRS-7) training options
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    batch_size: int = 512
    n_splits: int = 5
    seed: int = 42
    default_params: bool = False
    use_recency_weighting: bool = False

    # FSRS-7 feature flags
    include_short_term: bool = True
    two_buttons: bool = False
    max_seq_len: int = 64

    # Split options
    train_equals_test: bool = False
    no_test_same_day: bool = False
    no_train_same_day: bool = False

    # Partitioning ("none" | "preset" | "deck")
    partitions: str = "none"

    # Algorithm constraints
    s_min: float = 0.0001
    init_s_max: float = 100.0

    # Output
    save_evaluation_file: bool = False
    save_raw_output: bool = False
    save_weights: bool = False
    verbose_inadequate_data: bool = False

    # Parallelism
    num_processes: int = 1

    def get_evaluation_file_name(self) -> str:
        return METHOD_NAMES[self.method]


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_path=Path(args.data),
        max_user_id=args.max_user_id,
        method=args.method,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        seed=args.seed,
        default_params=args.default_params,
        use_recency_weighting=args.recency_weighting,
        partitions=args.partitions,
        save_evaluation_file=args.save_evaluation_file,
        save_raw_output=args.save_raw_output,
        save_weights=args.save_weights,
        verbose_inadequate_data=args.verbose,
        num_processes=args.processes,
        no_test_same_day=args.no_test_same_day,
        no_train_same_day=args.no_train_same_day,
    )


# ── data loading / preprocessing ──────────────────────────────────────────────


def _drop_iterative_capped_modes(df: pd.DataFrame, duration_col: str = "duration") -> pd.DataFrame:
    """
    Iteratively:
      1) find the mode duration value
      2) if that mode has .000 seconds (i.e., ms % 1000 == 0), remove all rows with that value
    Stop when:
      - no duplicate mode remains (mode frequency <= 1), or
      - mode is not xxx.000 seconds.
    """
    out = df.copy()

    while len(out) > 0:
        vc = out[duration_col].value_counts(dropna=False)
        if vc.empty:
            break

        top_freq = int(vc.iloc[0])
        if top_freq <= 1:
            break

        mode_candidates = vc[vc == top_freq].index.to_numpy(dtype="int64")
        mode_val = int(np.min(mode_candidates))

        if mode_val % 1000 != 0:
            break

        out = out[out[duration_col] != mode_val].copy()

    return out


def _is_inadequate_exception(e: Exception) -> bool:
    s = str(e).lower()
    return (
        s.endswith("inadequate.")
        or "non-empty tensorlist" in s
        or "stack expects a non-empty tensorlist" in s
        or "empty" in s and "batch" in s
    )


def _build_partition_map(raw_base: pd.DataFrame, user_id: int, config: Config) -> pd.DataFrame:
    out = raw_base[["event_id"]].copy()

    if config.partitions == "none":
        out["partition"] = 0
        return out

    # Build with minimal columns to avoid duplicate-name issues in feature pipeline.
    card_cols = ["card_id", "deck_id"]
    df_cards = pd.read_parquet(
        config.data_path / "cards",
        filters=[("user_id", "=", user_id)],
        columns=["user_id", *card_cols],
    ).drop(columns=["user_id"], errors="ignore")

    tmp = raw_base[["event_id", "card_id"]].merge(
        df_cards.drop_duplicates("card_id"),
        on="card_id",
        how="left",
    )

    if config.partitions == "deck":
        tmp["partition"] = tmp["deck_id"].fillna(-1).astype(int)
    else:
        df_decks = pd.read_parquet(
            config.data_path / "decks",
            filters=[("user_id", "=", user_id)],
            columns=["user_id", "deck_id", "preset_id"],
        ).drop(columns=["user_id"], errors="ignore")
        tmp = tmp.merge(
            df_decks.drop_duplicates("deck_id"),
            on="deck_id",
            how="left",
        )
        tmp["partition"] = tmp["preset_id"].fillna(-1).astype(int)

    return tmp[["event_id", "partition"]]


def load_user_frames(user_id: int, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      eval_df: rows used for benchmarking (includes first reviews), duration-filtered.
      algorithm_df: feature-engineered rows for FSRS-7, built from unfiltered durations.
    """
    raw = pd.read_parquet(config.data_path / "revlogs" / f"user_id={user_id}").copy()

    raw = raw[raw["rating"].isin([1, 2, 3, 4])].copy()

    if config.two_buttons:
        raw["rating"] = raw["rating"].replace({2: 3, 4: 3})

    raw = raw.reset_index(drop=True)
    raw["event_id"] = np.arange(1, len(raw) + 1)

    raw = raw.sort_values(by=["card_id", "event_id"]).copy()
    raw["i_raw"] = raw.groupby("card_id").cumcount() + 1
    raw["first_review"] = raw["i_raw"] == 1
    raw = raw[raw["i_raw"] <= config.max_seq_len * 2].copy()

    # Shared base after structural filtering; keep duration unfiltered for FSRS path.
    raw_base = raw.sort_values("event_id").copy()

    # Partition map (by event_id) computed separately to avoid introducing duplicate columns.
    partition_map = _build_partition_map(raw_base, user_id, config)

    # FSRS path: unfiltered-by-duration history.
    algorithm_input = raw_base.copy()
    algorithm_df = create_features(algorithm_input, config)
    algorithm_df = algorithm_df.merge(partition_map, on="event_id", how="left")
    algorithm_df["partition"] = algorithm_df["partition"].fillna(-1).astype(int)

    # Time-target path: duration filtering for regression evaluation.
    eval_df = raw_base.copy()
    eval_df["duration"] = pd.to_numeric(eval_df["duration"], errors="coerce")
    eval_df = eval_df.dropna(subset=["duration"]).copy()
    eval_df["duration"] = eval_df["duration"].astype("int64")

    non_positive = eval_df["duration"] <= 0
    afk_too_long = eval_df["duration"] > 1_800_000  # > 30 min in ms
    eval_df = eval_df[~non_positive & ~afk_too_long].copy()

    eval_df = _drop_iterative_capped_modes(eval_df, duration_col="duration")

    eval_df["duration_sec"] = (eval_df["duration"] / 1000.0).round(3)
    eval_df = eval_df.merge(partition_map, on="event_id", how="left")
    eval_df["partition"] = eval_df["partition"].fillna(-1).astype(int)
    eval_df = eval_df.sort_values("event_id").copy()

    if eval_df.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough usable (non-censored) data.")

    return eval_df.reset_index(drop=True), algorithm_df.reset_index(drop=True)


# ── algorithm trainer (FSRS-7 BCE training) ──────────────────────────────────


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

        self.train_dataset = BatchDataset(
            filtered.copy(), self.batch_size, max_seq_len=max_seq_len
        )
        self.train_loader = BatchLoader(self.train_dataset)
        if getattr(self.train_loader, "batch_nums", 0) == 0:
            raise Exception("Training data inadequate.")

        self.optimizer = algorithm.get_optimizer(
            lr=algorithm.lr, wd=algorithm.wd, betas=self.betas
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.train_loader.batch_nums * self.n_epoch,
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
                loss = (
                    self.loss_fn(result["retentions"], result["labels"])
                    * result["weights"]
                ).sum()
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
                total_loss += (
                    (
                        self.loss_fn(result["retentions"], result["labels"])
                        * result["weights"]
                    )
                    .sum()
                    .item()
                )
                if "penalty" in result:
                    total_loss += (result["penalty"] / epoch_len).item()
                total_items += batch[3].shape[0]
        self.train_loader.shuffle = True
        w = self.algorithm.state_dict()
        return total_loss / max(total_items, 1), w


# ── FSRS-7 retention helper ───────────────────────────────────────────────────


def batch_predict_retention(algorithm: FSRS7, dataset: pd.DataFrame, config: Config) -> list[float]:
    if dataset is None or len(dataset) == 0:
        return []

    algorithm.eval()
    ds = BatchDataset(dataset, batch_size=8192, sort_by_length=False)
    loader = BatchLoader(ds, shuffle=False)
    dev_loader = DevicePrefetchLoader(loader, target_device=config.device)
    retentions: list[float] = []
    with torch.no_grad():
        for batch in dev_loader:
            sequences, delta_ts, labels, seq_lens, weights = batch
            real_batch_size = seq_lens.shape[0]
            result = algorithm.batch_process(sequences, delta_ts, seq_lens, real_batch_size)
            retentions.extend(result["retentions"].cpu().tolist())
    return retentions


# ── method helpers ────────────────────────────────────────────────────────────


def _median_by_grade(df: pd.DataFrame) -> dict[int, float]:
    return df.groupby("rating")["duration_sec"].median().to_dict()


def _fill_grade_medians(base: dict[int, float], fallback: float) -> dict[int, float]:
    out = {}
    for g in [1, 2, 3, 4]:
        out[g] = float(base.get(g, fallback))
    return out


def _predict_const7(test_df: pd.DataFrame) -> np.ndarray:
    return np.full(len(test_df), 7.0, dtype=float)


def _predict_user_median(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    med = float(train_df["duration_sec"].median())
    return np.full(len(test_df), med, dtype=float)


def _predict_grade_median_4(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    global_med = float(train_df["duration_sec"].median())
    med = _fill_grade_medians(_median_by_grade(train_df), global_med)
    return test_df["rating"].map(med).astype(float).to_numpy()


def _predict_grade_median_8(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
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


def _fit_algorithm_partition_weights(train_algorithm_df: pd.DataFrame, config: Config) -> dict[int, list]:
    partition_weights: dict[int, list] = {}
    for partition in train_algorithm_df["partition"].unique():
        train_partition = train_algorithm_df[train_algorithm_df["partition"] == partition].copy()
        if len(train_partition) == 0:
            continue

        if config.use_recency_weighting:
            x = np.linspace(0, 1, len(train_partition))
            train_partition["weights"] = 0.25 + 0.75 * np.power(x, 3)

        algorithm = FSRS7(config).to(config.device)

        if config.default_params:
            partition_weights[int(partition)] = algorithm.state_dict()
            continue

        try:
            trainer = AlgorithmTrainer(
                algorithm=algorithm,
                train_set=train_partition,
                batch_size=config.batch_size,
                max_seq_len=config.max_seq_len,
            )
            partition_weights[int(partition)] = trainer.train()
        except Exception as e:
            if _is_inadequate_exception(e):
                if config.verbose_inadequate_data:
                    tqdm.write(f"Skipping partition {partition} due to inadequate data.")
                partition_weights[int(partition)] = FSRS7(config).state_dict()
            else:
                raise
    return partition_weights


def _predict_R_maps(
    train_algorithm_df: pd.DataFrame,
    test_algorithm_df: pd.DataFrame,
    config: Config,
) -> tuple[dict[int, float], dict[int, float], dict[int, list]]:
    if len(train_algorithm_df) == 0:
        return {}, {}, {}

    weights = _fit_algorithm_partition_weights(train_algorithm_df, config)
    train_map: dict[int, float] = {}
    test_map: dict[int, float] = {}

    for partition, w in weights.items():
        algorithm = FSRS7(config, w=w).to(config.device)

        tr_part = train_algorithm_df[train_algorithm_df["partition"] == partition].copy()
        if len(tr_part) > 0:
            r_train = batch_predict_retention(algorithm, tr_part, config)
            for eid, r in zip(tr_part["event_id"].tolist(), r_train):
                train_map[int(eid)] = float(r)

        te_part = test_algorithm_df[test_algorithm_df["partition"] == partition].copy()
        if len(te_part) > 0:
            r_test = batch_predict_retention(algorithm, te_part, config)
            for eid, r in zip(te_part["event_id"].tolist(), r_test):
                test_map[int(eid)] = float(r)

    return train_map, test_map, weights


def _predict_fsrs_r_linear(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_R_map: dict[int, float],
    test_R_map: dict[int, float],
) -> np.ndarray:
    global_med = float(train_eval["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(train_eval), global_med)
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

    non_first_map_raw = train_eval[~train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
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
    return np.array(pred, dtype=float)


def _predict_fsrs_r_grade_interact(
    train_eval: pd.DataFrame,
    test_eval: pd.DataFrame,
    train_R_map: dict[int, float],
    test_R_map: dict[int, float],
) -> np.ndarray:
    global_med = float(train_eval["duration_sec"].median())
    by_grade_all = _fill_grade_medians(_median_by_grade(train_eval), global_med)
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

    non_first_map_raw = train_eval[~train_eval["first_review"]].groupby("rating")["duration_sec"].median().to_dict()
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


# ── metrics / output ──────────────────────────────────────────────────────────


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else None
    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MAPE": None if mape is None else round(mape, 6),
    }


def evaluate(
    y_true: list[float],
    y_pred: list[float],
    user_id: int,
    config: Config,
    algorithm_weights_last_split: Optional[dict[int, list]] = None,
) -> tuple[dict, Optional[dict]]:
    y = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)

    stats: dict = {
        "metrics": _compute_metrics(y, p),
        "user": int(user_id),
        "size": int(len(y_true)),
    }

    if algorithm_weights_last_split:
        stats["algorithm_parameters"] = {
            int(partition): list(map(lambda x: round(float(x), 6), w))
            for partition, w in algorithm_weights_last_split.items()
            if isinstance(w, list)
        }

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
        df.to_csv(
            f"evaluation/{config.get_evaluation_file_name()}/{user_id}.tsv",
            sep="\t",
            index=False,
        )


def sort_jsonl(file: Path) -> list:
    data = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8", newline="\n") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return data


# ── per-user processing ───────────────────────────────────────────────────────


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
def process(user_id: int, config: Config) -> tuple[dict, Optional[dict]]:
    eval_df, algorithm_df = load_user_frames(user_id, config)

    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    y_all: list[float] = []
    p_all: list[float] = []
    save_tmp: list[pd.DataFrame] = []
    last_algorithm_weights: Optional[dict[int, list]] = None

    for _, (train_idx, test_idx) in enumerate(tscv.split(eval_df)):
        if not config.train_equals_test:
            train_eval = eval_df.iloc[train_idx].copy()
            test_eval = eval_df.iloc[test_idx].copy()
        else:
            train_eval = eval_df.copy()
            test_eval = eval_df.iloc[test_idx].copy()

        if config.no_test_same_day:
            test_eval = test_eval[test_eval["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_eval = train_eval[train_eval["elapsed_days"] > 0].copy()

        if len(test_eval) == 0 or len(train_eval) == 0:
            continue

        # FSRS uses unfiltered-by-duration history, sliced by temporal event windows.
        train_end = int(train_eval["event_id"].max())
        test_start = int(test_eval["event_id"].min())
        test_end = int(test_eval["event_id"].max())

        train_algorithm_df = algorithm_df[algorithm_df["event_id"] <= train_end].copy()
        test_algorithm_df = algorithm_df[
            (algorithm_df["event_id"] >= test_start) & (algorithm_df["event_id"] <= test_end)
        ].copy()

        if config.method == "const":
            pred = _predict_const7(test_eval)

        elif config.method == "user_median":
            pred = _predict_user_median(train_eval, test_eval)

        elif config.method == "grade_median_4":
            pred = _predict_grade_median_4(train_eval, test_eval)

        elif config.method == "grade_median_8":
            pred = _predict_grade_median_8(train_eval, test_eval)

        elif config.method in ("fsrs_r_linear", "fsrs_r_grade_interact"):
            train_R_map, test_R_map, weights = _predict_R_maps(train_algorithm_df, test_algorithm_df, config)
            last_algorithm_weights = weights if weights else last_algorithm_weights

            if config.method == "fsrs_r_linear":
                pred = _predict_fsrs_r_linear(train_eval, test_eval, train_R_map, test_R_map)
            else:
                pred = _predict_fsrs_r_grade_interact(train_eval, test_eval, train_R_map, test_R_map)

        else:
            raise ValueError(f"Unknown method: {config.method}")

        pred = np.maximum(pred.astype(float), 0.0)

        test_eval = test_eval.copy()
        test_eval["t_true"] = test_eval["duration_sec"].astype(float)
        test_eval["t_pred"] = pred.astype(float)

        y_all.extend(test_eval["t_true"].tolist())
        p_all.extend(test_eval["t_pred"].tolist())
        save_tmp.append(test_eval)

        if config.train_equals_test:
            break

    if len(save_tmp) == 0:
        raise Exception(f"{user_id}: no valid splits after filtering.")

    save_df = pd.concat(save_tmp, ignore_index=True)
    save_evaluation_file(user_id, save_df, config)

    return evaluate(y_all, p_all, user_id, config, last_algorithm_weights)


# ── main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review-time benchmark")
    p.add_argument("--data", default="../anki-revlogs-10k", help="Path to dataset")
    p.add_argument("--processes", type=int, default=1, help="Number of parallel workers")

    p.add_argument(
        "--method",
        default="const",
        choices=[
            "const",
            "user_median",
            "grade_median_4",
            "grade_median_8",
            "fsrs_r_linear",
            "fsrs_r_grade_interact",
        ],
        help="Benchmark method",
    )

    p.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    p.add_argument("--n-splits", dest="n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--default-params",
        dest="default_params",
        action="store_true",
        help="Skip algorithm training, use default FSRS-7 parameters",
    )
    p.add_argument("--recency-weighting", dest="recency_weighting", action="store_true")
    p.add_argument("--partitions", default="none", choices=["none", "preset", "deck"])
    p.add_argument("--no-test-same-day", dest="no_test_same_day", action="store_true")
    p.add_argument("--no-train-same-day", dest="no_train_same_day", action="store_true")
    p.add_argument("--save-evaluation-file", dest="save_evaluation_file", action="store_true")
    p.add_argument("--save-raw", dest="save_raw_output", action="store_true")
    p.add_argument("--save-weights", dest="save_weights", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--max-user-id", dest="max_user_id", type=int, default=None)
    return p.parse_args()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = _parse_args()
    config = build_config(args)
    torch.manual_seed(config.seed)

    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    file_name = config.get_evaluation_file_name()

    Path(f"evaluation/{file_name}").mkdir(parents=True, exist_ok=True)
    Path("result").mkdir(parents=True, exist_ok=True)
    Path("raw").mkdir(parents=True, exist_ok=True)

    result_file = Path(f"result/{file_name}.jsonl")
    raw_file = Path(f"raw/{file_name}.jsonl")

    processed_users: set = set()
    if result_file.exists():
        processed_users = {d["user"] for d in sort_jsonl(result_file)}
    if config.save_raw_output and raw_file.exists():
        sort_jsonl(raw_file)

    unprocessed = []
    for user_id in dataset.partitioning.dictionaries[0]:
        uid = user_id.as_py()
        if config.max_user_id is not None and uid > config.max_user_id:
            continue
        if uid not in processed_users:
            unprocessed.append(uid)
    unprocessed.sort()

    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [executor.submit(process, uid, config) for uid in unprocessed]
        pbar = tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        for future in pbar:
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                else:
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
    if config.save_raw_output:
        sort_jsonl(raw_file)


if __name__ == "__main__":
    main()
