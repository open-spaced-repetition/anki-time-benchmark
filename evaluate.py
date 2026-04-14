import argparse
import json
import math
import pathlib
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy  # type: ignore


DEFAULT_METHODS = [
    "CONST",
    "USER_MEDIAN",
    "GRADE_MEDIAN_4",
    "GRADE_MEDIAN_4_4",
    "POOR_MANS_FSRS"
    "GRADE_MEDIAN_8",  # same as GRADE_MEDIAN_4 if first reviews are excluded
    "FSRS7_R_LINEAR",
    "FSRS7_R_GRADE_INTERACT",
    "FSRS7_DSR_GRADE_NN",
]


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

    def weighted_mean_bootstrap(z, axis):
        data = np.vectorize(dict_x_w.get)(z)
        return np.average(data[0], weights=data[1], axis=axis)

    try:
        ci_99 = scipy.stats.bootstrap(
            (identifiers,),
            statistic=weighted_mean_bootstrap,
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


def assert_user_size_consistency(
    method_user_data: Dict[str, Dict[str, Dict]],
    method_to_file: Dict[str, str],
    suffix_label: str,
) -> None:
    """
    For a single suffix, verify that if a user appears in multiple method files,
    their `size` is identical across those files.
    """
    sizes_by_user: Dict[str, Dict[int, List[str]]] = {}

    for method, user_map in method_user_data.items():
        src = method_to_file.get(method, method)
        for user, row in user_map.items():
            size = int(row.get("size", 0))
            sizes_by_user.setdefault(user, {}).setdefault(size, []).append(src)

    mismatches = []
    for user, size_groups in sizes_by_user.items():
        if len(size_groups) > 1:
            parts = []
            for sz, files in sorted(size_groups.items(), key=lambda x: x[0]):
                parts.append(f"size={sz}: {', '.join(sorted(files))}")
            mismatches.append(f"user={user} -> " + " | ".join(parts))

    if mismatches:
        preview = "\n".join(mismatches[:20])
        more = f"\n... and {len(mismatches) - 20} more" if len(mismatches) > 20 else ""
        raise ValueError(
            f"Inconsistent per-user `size` detected for suffix '{suffix_label}'.\n"
            f"A user has different review counts across method files (possible bug).\n"
            f"{preview}{more}"
        )

def discover_methods(result_dir: pathlib.Path, suffix: Optional[str]) -> List[str]:
    stems = [f.stem for f in sorted(result_dir.glob("*.jsonl"))]
    if suffix:
        tag = f"_{suffix}"
        methods = [s[:-len(tag)] for s in stems if s.endswith(tag)]
        return sorted(set(methods))
    return stems


def resolve_result_file(method: str, result_dir: pathlib.Path, suffix: Optional[str]) -> Optional[pathlib.Path]:
    candidates: List[pathlib.Path] = []
    if suffix:
        candidates.append(result_dir / f"{method}_{suffix}.jsonl")
    candidates.append(result_dir / f"{method}.jsonl")

    for fp in candidates:
        if fp.exists():
            return fp
    return None


def load_method_user_metrics(
    method: str, result_dir: pathlib.Path, suffix: Optional[str]
) -> Dict[str, Dict]:
    """
    Returns per-user metrics for one method:
      {
        user_id: {"MAE": float, "RMSE": float, "MAPE": float, "size": int},
        ...
      }
    """
    fp = resolve_result_file(method, result_dir, suffix)
    if fp is None:
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
    if suffix == "s":
        return f"{m}±{c} s"
    if suffix == "%":
        return f"{m}%±{c}%"
    raise Exception("Unknown suffix")


def print_table_for_suffix(
    result_dir: pathlib.Path,
    methods_arg: Optional[List[str]],
    use_default_methods: bool,
    suffix: str,
    weight_by: str,
) -> None:
    suffix = suffix.strip()
    suffix_for_match: Optional[str] = suffix or None

    if methods_arg:
        methods = methods_arg
    elif use_default_methods:
        methods = DEFAULT_METHODS
    else:
        methods = discover_methods(result_dir, suffix=suffix_for_match)

    print(f"\n=== {suffix if suffix else '(no suffix)'} ===")

    if not methods:
        print("No result files found.")
        return

    method_user_data: Dict[str, Dict[str, Dict]] = {}
    method_to_file: Dict[str, str] = {}

    for method in methods:
        fp = resolve_result_file(method, result_dir, suffix_for_match)
        if fp is None:
            continue

        d = load_method_user_metrics(method, result_dir, suffix=suffix_for_match)
        if d:
            method_user_data[method] = d
            method_to_file[method] = fp.name

    if not method_user_data:
        print("No valid method files with MAE/RMSE/MAPE found.")
        return

    assert_user_size_consistency(
        method_user_data=method_user_data,
        method_to_file=method_to_file,
        suffix_label=suffix if suffix else "(no suffix)",
    )

    user_sets: List[Set[str]] = [set(d.keys()) for d in method_user_data.values()]
    common_users = set.intersection(*user_sets) if user_sets else set()

    if not common_users:
        print("No common users found across selected method files.")
        return

    print(f"Aggregation weight: {weight_by} (99% CI, bootstrap BCa)")
    print(f"Methods compared: {len(method_user_data)}")
    print(f"Common users in all method files: {len(common_users)}")

    ref_method = next(iter(method_user_data.keys()))
    common_reviews = sum(method_user_data[ref_method][u]["size"] for u in common_users)
    print(f"Common reviews (from shared users): {common_reviews}\n")

    rows = []
    for method, user_map in method_user_data.items():
        metrics_list = [user_map[u] for u in common_users]

        mae_mean, mae_ci = _metric_mean_ci(metrics_list, "MAE", weight_by)
        rmse_mean, rmse_ci = _metric_mean_ci(metrics_list, "RMSE", weight_by)
        mape_mean, mape_ci = _metric_mean_ci(metrics_list, "MAPE", weight_by)

        rows.append(
            (
                method,
                float("inf") if rmse_mean is None else rmse_mean,
                fmt_mean_ci(rmse_mean, rmse_ci, "s"),
                fmt_mean_ci(mae_mean, mae_ci, "s"),
                fmt_mean_ci(mape_mean, mape_ci, "%"),
            )
        )

    rows.sort(key=lambda x: x[1])

    print("| Method | RMSE | MAE | MAPE |")
    print("| --- | --- | --- | --- |")
    for method, _, rmse, mae, mape in rows:
        print(f"| {method} | {rmse} | {mae} | {mape} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="./result", help="Directory with *.jsonl result files")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional base method names (e.g., CONST FSRS7_R_LINEAR). "
            "If omitted, uses --default-methods or auto-discovery."
        ),
    )
    parser.add_argument(
        "--default-methods",
        action="store_true",
        help="Use built-in default method list instead of auto-discovery.",
    )
    parser.add_argument(
        "--weight-by",
        choices=["reviews", "users"],
        default="reviews",
        help="Aggregate per-user metrics weighted by number of reviews or equally by users",
    )
    parser.add_argument(
        "--without-first-suffix",
        default="NO_FIRST_REVIEWS",
        help="Filename suffix for runs excluding first reviews.",
    )
    parser.add_argument(
        "--with-first-suffix",
        default="WITH_FIRST_REVIEWS",
        help="Filename suffix for runs including first reviews.",
    )
    args = parser.parse_args()

    result_dir = pathlib.Path(args.result_dir)
    if not result_dir.exists():
        print(f"Result directory not found: {result_dir}")
        return

    print_table_for_suffix(
        result_dir=result_dir,
        methods_arg=args.methods,
        use_default_methods=args.default_methods,
        suffix=args.without_first_suffix,
        weight_by=args.weight_by,
    )
    print_table_for_suffix(
        result_dir=result_dir,
        methods_arg=args.methods,
        use_default_methods=args.default_methods,
        suffix=args.with_first_suffix,
        weight_by=args.weight_by,
    )


if __name__ == "__main__":
    main()
