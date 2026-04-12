import argparse
import json
import math
import pathlib
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy  # type: ignore


def sigdig(value: float, CI: float) -> Tuple[str, str]:
    def num_lead_zeros(x: float) -> float:
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = int(n_lead_zeros_CI + CI_sigdigs)
    rounded_CI = round(CI, decimals)
    rounded_value = round(value, decimals)
    if n_lead_zeros_CI > num_lead_zeros(rounded_CI):
        d = max(decimals - 1, 0)
        return str(f"{round(value, d):.{d}f}"), str(f"{round(CI, d):.{d}f}")
    return str(f"{rounded_value:.{max(decimals, 0)}f}"), str(
        f"{rounded_CI:.{max(decimals, 0)}f}"
    )


def confidence_interval(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0

    identifiers = np.arange(len(values))
    dict_x_w = {
        int(i): (float(v), float(w))
        for i, (v, w) in enumerate(zip(values, weights))
    }

    def weighted_mean(z, axis):
        data = np.vectorize(dict_x_w.get)(z)
        return np.average(data[0], weights=data[1], axis=axis)

    try:
        ci_99 = scipy.stats.bootstrap(
            (identifiers,),
            statistic=weighted_mean,
            confidence_level=0.99,
            axis=0,
            method="BCa",
            random_state=42,
        )
        low = float(ci_99.confidence_interval.low)
        high = float(ci_99.confidence_interval.high)
        return max((high - low) / 2, 0.0)
    except Exception:
        return 0.0


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def discover_methods(result_dir: pathlib.Path) -> List[str]:
    return [f.stem for f in sorted(result_dir.glob("*.jsonl"))]


def load_method_user_metrics(method: str, result_dir: pathlib.Path) -> Dict[str, Dict]:
    """
    Returns per-user metrics for one method:
      {
        user_id: {"MAE": float, "RMSE": float, "MAPE": float, "size": int},
        ...
      }
    """
    fp = result_dir / f"{method}.jsonl"
    if not fp.exists():
        return {}

    with fp.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out: Dict[str, Dict] = {}
    for r in rows:
        user = r.get("user")
        m = r.get("metrics", {})
        if user is None:
            continue
        if "MAE" not in m or "RMSE" not in m or "MAPE" not in m:
            continue
        if m["MAE"] is None or m["RMSE"] is None or m["MAPE"] is None:
            continue

        out[str(user)] = {
            "MAE": float(m["MAE"]),
            "RMSE": float(m["RMSE"]),
            "MAPE": float(m["MAPE"]),
            "size": int(r.get("size", 0)),
        }
    return out


def _metric_mean_ci(
    metrics_list: List[Dict], metric: str, weight_by: str
) -> Tuple[Optional[float], Optional[float]]:
    vals = np.array([x[metric] for x in metrics_list], dtype=float)

    if weight_by == "reviews":
        w = np.array([x["size"] for x in metrics_list], dtype=float)
        if np.all(w == 0):
            w = np.ones_like(w)
    else:
        w = np.ones(len(metrics_list), dtype=float)

    mask = ~np.isnan(vals)
    vals = vals[mask]
    w = w[mask]

    if len(vals) == 0:
        return None, None

    mean = weighted_mean(vals, w)
    ci = confidence_interval(vals, w)
    return mean, ci


def fmt_mean_ci(mean: Optional[float], ci: Optional[float], suffix: str = "") -> str:
    if mean is None or ci is None:
        return "N/A"
    m, c = sigdig(mean, ci)
    return f"{m}±{c}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="./result", help="Directory with *.jsonl result files")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method names (file stems). If omitted, auto-discover from result dir",
    )
    parser.add_argument(
        "--weight-by",
        choices=["reviews", "users"],
        default="reviews",
        help="Aggregate per-user metrics weighted by number of reviews or equally by users",
    )
    args = parser.parse_args()

    result_dir = pathlib.Path(args.result_dir)
    if not result_dir.exists():
        print(f"Result directory not found: {result_dir}")
        return

    methods = args.methods if args.methods else discover_methods(result_dir)
    if not methods:
        print("No result files found.")
        return

    # Load all method user maps first
    method_user_data: Dict[str, Dict[str, Dict]] = {}
    for method in methods:
        d = load_method_user_metrics(method, result_dir)
        if d:
            method_user_data[method] = d

    if not method_user_data:
        print("No valid method files with MAE/RMSE/MAPE found.")
        return

    # Intersection of users across all loaded method files
    user_sets: List[Set[str]] = [set(d.keys()) for d in method_user_data.values()]
    common_users = set.intersection(*user_sets) if user_sets else set()

    if not common_users:
        print("No common users found across selected method files.")
        return

    print(f"Aggregation weight: {args.weight_by} (99% CI, bootstrap BCa)")
    print(f"Methods compared: {len(method_user_data)}")
    print(f"Common users in all method files: {len(common_users)}")

    # Print common review count once (using first method as reference)
    ref_method = next(iter(method_user_data.keys()))
    common_reviews = sum(method_user_data[ref_method][u]["size"] for u in common_users)
    print(f"Common reviews (from shared users): {common_reviews}\n")

    rows = []
    for method, user_map in method_user_data.items():
        metrics_list = [user_map[u] for u in common_users]

        mae_mean, mae_ci = _metric_mean_ci(metrics_list, "MAE", args.weight_by)
        rmse_mean, rmse_ci = _metric_mean_ci(metrics_list, "RMSE", args.weight_by)
        mape_mean, mape_ci = _metric_mean_ci(metrics_list, "MAPE", args.weight_by)

        rows.append(
            (
                method,
                float("inf") if rmse_mean is None else rmse_mean,  # sort key
                fmt_mean_ci(rmse_mean, rmse_ci, " s"),
                fmt_mean_ci(mae_mean, mae_ci, " s"),
                fmt_mean_ci(mape_mean, mape_ci, "%"),
            )
        )

    rows.sort(key=lambda x: x[1])  # lowest RMSE first

    print("| Method | RMSE | MAE | MAPE |")
    print("| --- | --- | --- | --- |")
    for method, _, rmse, mae, mape in rows:
        print(f"| {method} | {rmse} | {mae} | {mape} |")


if __name__ == "__main__":
    main()