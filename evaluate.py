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
    if not np.isfinite(CI) or CI <= 0:
        return str(f"{round(value, 2):.2f}"), "0.00"

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


def load_method_r_bucket_precision(
    method: str, result_dir: pathlib.Path, suffix: Optional[str]
) -> Dict[str, List[Dict]]:
    fp = resolve_result_file(method, result_dir, suffix)
    if fp is None:
        return {}

    with fp.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out: Dict[str, List[Dict]] = {}
    for r in rows:
        user = r.get("user")
        buckets = r.get("r_bucket_precision")
        if user is None or not isinstance(buckets, list):
            continue
        out[str(user)] = buckets
    return out


def _aggregate_r_bucket_precision(bucket_lists: List[List[Dict]]) -> List[Dict]:
    acc: Dict[Tuple[float, float], Dict[str, float]] = {}

    for buckets in bucket_lists:
        for b in buckets:
            try:
                lo = float(b["bucket_start"])
                hi = float(b["bucket_end"])
                count = float(b["count"])
                mse = float(b["mse_sec"]) if "mse_sec" in b else float("nan")
                mae = float(b["mae_sec"])
                precise = float(b["precise_enough_pct"])
                mean_true = float(b["mean_true_sec"]) if "mean_true_sec" in b else float("nan")
                mean_pred = float(b["mean_pred_sec"]) if "mean_pred_sec" in b else float("nan")
                tol = float(b.get("tolerance_sec", 2.0))
            except Exception:
                continue
            if count <= 0:
                continue
            k = (lo, hi)
            if k not in acc:
                acc[k] = {
                    "count": 0.0,
                    "mean_true_weighted_sum": 0.0,
                    "mean_true_weight": 0.0,
                    "mean_pred_weighted_sum": 0.0,
                    "mean_pred_weight": 0.0,
                    "mse_weighted_sum": 0.0,
                    "mse_weight": 0.0,
                    "mae_weighted_sum": 0.0,
                    "precise_weighted_sum": 0.0,
                    "tolerance_sec": tol,
                }
            acc[k]["count"] += count
            if np.isfinite(mean_true):
                acc[k]["mean_true_weighted_sum"] += mean_true * count
                acc[k]["mean_true_weight"] += count
            if np.isfinite(mean_pred):
                acc[k]["mean_pred_weighted_sum"] += mean_pred * count
                acc[k]["mean_pred_weight"] += count
            if np.isfinite(mse):
                acc[k]["mse_weighted_sum"] += mse * count
                acc[k]["mse_weight"] += count
            acc[k]["mae_weighted_sum"] += mae * count
            acc[k]["precise_weighted_sum"] += precise * count

    out: List[Dict] = []
    for (lo, hi), v in sorted(acc.items(), key=lambda x: x[0][0]):
        cnt = v["count"]
        out.append(
            {
                "bucket_start": round(lo, 2),
                "bucket_end": round(hi, 2),
                "count": int(cnt),
                "mean_true_sec": (
                    round(v["mean_true_weighted_sum"] / v["mean_true_weight"], 6)
                    if v["mean_true_weight"] > 0
                    else None
                ),
                "mean_pred_sec": (
                    round(v["mean_pred_weighted_sum"] / v["mean_pred_weight"], 6)
                    if v["mean_pred_weight"] > 0
                    else None
                ),
                "rmse_sec": (
                    round(float(np.sqrt(v["mse_weighted_sum"] / v["mse_weight"])), 6)
                    if v["mse_weight"] > 0
                    else None
                ),
                "mae_sec": round(v["mae_weighted_sum"] / cnt, 6),
                "precise_enough_pct": round(v["precise_weighted_sum"] / cnt, 6),
                "tolerance_sec": float(v["tolerance_sec"]),
            }
        )
    return out


def _add_ratio_mapping_pct(
    rows: List[Dict],
    ref_bucket_start: float = 0.85,
    ref_bucket_end: float = 0.90,
) -> List[Dict]:
    out = [dict(r) for r in rows]
    ref_row = next(
        (
            r
            for r in out
            if np.isclose(float(r.get("bucket_start", -1.0)), ref_bucket_start)
            and np.isclose(float(r.get("bucket_end", -1.0)), ref_bucket_end)
        ),
        None,
    )
    if ref_row is None:
        for r in out:
            r["ratio_mapping_pct"] = None
            r["pred_ratio_to_ref"] = None
            r["actual_ratio_to_ref"] = None
        return out

    ref_true = ref_row.get("mean_true_sec")
    ref_pred = ref_row.get("mean_pred_sec")
    if ref_true is None or ref_pred is None or float(ref_true) <= 0 or float(ref_pred) <= 0:
        for r in out:
            r["ratio_mapping_pct"] = None
            r["pred_ratio_to_ref"] = None
            r["actual_ratio_to_ref"] = None
        return out

    ref_true_f = float(ref_true)
    ref_pred_f = float(ref_pred)
    eps = 1e-12
    for r in out:
        bt = r.get("mean_true_sec")
        bp = r.get("mean_pred_sec")
        if bt is None or bp is None or float(bt) <= 0 or float(bp) <= 0:
            r["ratio_mapping_pct"] = None
            r["pred_ratio_to_ref"] = None
            r["actual_ratio_to_ref"] = None
            continue
        pred_ratio = float(bp) / ref_pred_f
        actual_ratio = float(bt) / ref_true_f
        denom = max(pred_ratio, actual_ratio, eps)
        mapping_pct = 100.0 * min(pred_ratio, actual_ratio) / denom
        r["ratio_mapping_pct"] = round(mapping_pct, 6)
        r["pred_ratio_to_ref"] = round(pred_ratio, 6)
        r["actual_ratio_to_ref"] = round(actual_ratio, 6)
    return out


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None
    sw = float(np.sum(w))
    if sw <= 0:
        return None
    mx = float(np.sum(w * x) / sw)
    my = float(np.sum(w * y) / sw)
    dx = x - mx
    dy = y - my
    cov = float(np.sum(w * dx * dy) / sw)
    vx = float(np.sum(w * dx * dx) / sw)
    vy = float(np.sum(w * dy * dy) / sw)
    if vx <= 0 or vy <= 0:
        return None
    return cov / float(np.sqrt(vx * vy))


def _bucket_corr_summary(rows: List[Dict]) -> Optional[Dict[str, float]]:
    valid = [
        r
        for r in rows
        if r.get("mean_true_sec") is not None and r.get("mean_pred_sec") is not None and int(r.get("count", 0)) > 0
    ]
    if len(valid) < 2:
        return None

    centers = np.array(
        [(float(r["bucket_start"]) + float(r["bucket_end"])) / 2.0 for r in valid],
        dtype=float,
    )
    mean_true = np.array([float(r["mean_true_sec"]) for r in valid], dtype=float)
    mean_pred = np.array([float(r["mean_pred_sec"]) for r in valid], dtype=float)
    weights = np.array([float(r["count"]) for r in valid], dtype=float)

    pearson_true = _weighted_pearson(centers, mean_true, weights)
    pearson_pred = _weighted_pearson(centers, mean_pred, weights)

    # Rank-based correlation over bucket means (bucket-level Spearman proxy).
    rank_x = np.argsort(np.argsort(centers)).astype(float)
    rank_true = np.argsort(np.argsort(mean_true)).astype(float)
    rank_pred = np.argsort(np.argsort(mean_pred)).astype(float)
    spearman_true = _weighted_pearson(rank_x, rank_true, weights)
    spearman_pred = _weighted_pearson(rank_x, rank_pred, weights)

    return {
        "pearson_true": float(pearson_true) if pearson_true is not None else float("nan"),
        "pearson_pred": float(pearson_pred) if pearson_pred is not None else float("nan"),
        "spearman_true": float(spearman_true) if spearman_true is not None else float("nan"),
        "spearman_pred": float(spearman_pred) if spearman_pred is not None else float("nan"),
    }


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


def _print_aligned_markdown_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: List[str]) -> str:
        padded = [cells[i].ljust(widths[i]) for i in range(len(cells))]
        return "| " + " | ".join(padded) + " |"

    print(_fmt_row(headers))
    print("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    for row in rows:
        print(_fmt_row(row))


def _format_highlights(
    rows: List[Tuple[str, float, Optional[float], Optional[float], str, str, str]]
) -> List[str]:
    valid_rmse = [r for r in rows if r[1] != float("inf")]
    valid_mae = [r for r in rows if r[2] is not None]
    valid_mape = [r for r in rows if r[3] is not None]

    def _best(rows_in: List[Tuple], metric_idx: int) -> Tuple[List[str], float]:
        best_v = min(float(r[metric_idx]) for r in rows_in)
        winners = [str(r[0]) for r in rows_in if np.isclose(float(r[metric_idx]), best_v)]
        return winners, best_v

    rmse_winners, rmse_best = _best(valid_rmse, 1)
    mae_winners, mae_best = _best(valid_mae, 2)
    mape_winners, mape_best = _best(valid_mape, 3)

    lines = [
        f"- Best RMSE: {', '.join(rmse_winners)} ({rmse_best:.4f} s)",
        f"- Best MAE: {', '.join(mae_winners)} ({mae_best:.4f} s)",
        f"- Best MAPE: {', '.join(mape_winners)} ({mape_best:.4f}%)",
    ]

    common = set(rmse_winners) & set(mae_winners) & set(mape_winners)
    if common:
        lines.append(f"- Overall: {', '.join(sorted(common))} is best on RMSE, MAE, and MAPE.")
    else:
        lines.append("- Overall: no single method is best on all three metrics.")
    return lines


def _metric_interpretation_line() -> str:
    return (
        "Interpretation tip: 10s MAE means ~10s average absolute error; "
        "10s RMSE is in seconds too but penalizes large misses more; "
        "MAPE is percentage (e.g., 10% means ~2s error when true time is 20s)."
    )


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
                mae_mean,
                mape_mean,
                fmt_mean_ci(rmse_mean, rmse_ci, "s"),
                fmt_mean_ci(mae_mean, mae_ci, "s"),
                fmt_mean_ci(mape_mean, mape_ci, "%"),
            )
        )

    rows.sort(key=lambda x: x[1])

    table_rows = [[method, rmse, mae, mape] for method, _, _, _, rmse, mae, mape in rows]
    _print_aligned_markdown_table(["Method", "RMSE", "MAE", "MAPE"], table_rows)
    print("\nHighlights:")
    for line in _format_highlights(rows):
        print(line)
    print(_metric_interpretation_line())

    r_bucket_by_method: Dict[str, Dict[str, List[Dict]]] = {}
    for method in method_user_data.keys():
        d = load_method_r_bucket_precision(method, result_dir, suffix_for_match)
        if d:
            r_bucket_by_method[method] = d

    if r_bucket_by_method:
        print("\nR-bucket precision (5% buckets):")
        corr_rows: List[List[str]] = []
        for method, by_user in r_bucket_by_method.items():
            bucket_lists = [by_user[u] for u in common_users if u in by_user]
            if not bucket_lists:
                continue
            agg = _aggregate_r_bucket_precision(bucket_lists)
            if not agg:
                continue
            agg = _add_ratio_mapping_pct(agg, ref_bucket_start=0.85, ref_bucket_end=0.90)
            corr = _bucket_corr_summary(agg)
            if corr is not None:
                corr_rows.append(
                    [
                        method,
                        ("N/A" if not np.isfinite(corr["pearson_true"]) else f"{corr['pearson_true']:.4f}"),
                        ("N/A" if not np.isfinite(corr["pearson_pred"]) else f"{corr['pearson_pred']:.4f}"),
                        ("N/A" if not np.isfinite(corr["spearman_true"]) else f"{corr['spearman_true']:.4f}"),
                        ("N/A" if not np.isfinite(corr["spearman_pred"]) else f"{corr['spearman_pred']:.4f}"),
                    ]
                )
            tol = agg[0].get("tolerance_sec", 2.0)
            print(f"\nMethod: {method} (Within 2s uses |pred-true| <= {tol:.1f}s)")
            table_rows = [
                [
                    f"{b['bucket_start']:.2f}-{b['bucket_end']:.2f}",
                    str(b["count"]),
                    ("N/A" if b.get("mean_true_sec") is None else f"{float(b['mean_true_sec']):.4f}"),
                    ("N/A" if b.get("mean_pred_sec") is None else f"{float(b['mean_pred_sec']):.4f}"),
                    ("N/A" if b.get("rmse_sec") is None else f"{float(b['rmse_sec']):.4f}"),
                    f"{b['mae_sec']:.4f}",
                    ("N/A" if b.get("actual_ratio_to_ref") is None else f"{float(b['actual_ratio_to_ref']):.4f}"),
                    ("N/A" if b.get("pred_ratio_to_ref") is None else f"{float(b['pred_ratio_to_ref']):.4f}"),
                    ("N/A" if b.get("ratio_mapping_pct") is None else f"{float(b['ratio_mapping_pct']):.2f}"),
                    f"{b['precise_enough_pct']:.2f}",
                ]
                for b in agg
            ]
            _print_aligned_markdown_table(
                [
                    "R bucket",
                    "Count",
                    "Mean true (s)",
                    "Mean pred (s)",
                    "RMSE (s)",
                    "MAE (s)",
                    "True ratio vs 0.85-0.90",
                    "Pred ratio vs 0.85-0.90",
                    "Ratio match (%)",
                    "Within 2s (%)",
                ],
                table_rows,
            )
        if corr_rows:
            print("\nR-time correlation summary (bucket-level, weighted by bucket count):")
            _print_aligned_markdown_table(
                [
                    "Method",
                    "Pearson R~true",
                    "Pearson R~pred",
                    "Spearman R~true",
                    "Spearman R~pred",
                ],
                corr_rows,
            )


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
