from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import (
    _aggregate_r_bucket_precision,
    discover_methods,
    load_method_r_bucket_precision,
    load_method_user_metrics,
)


METHOD_HEADER_RE = re.compile(r"^Method:\s+(.+?)\s+\(Within 2s", re.MULTILINE)
BUCKET_RE = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$")
LEADING_FLOAT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")


def safe_name(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", method).strip("_")


def parse_methods_data(text: str) -> dict[str, list[tuple[float, float]]]:
    methods_data: dict[str, list[tuple[float, float]]] = {}

    headers = list(METHOD_HEADER_RE.finditer(text))
    for i, h in enumerate(headers):
        method = h.group(1).strip()
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]

        pairs: list[tuple[float, float]] = []
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) < 4:
                continue
            if not BUCKET_RE.match(cols[0]):
                continue
            if cols[2] == "N/A" or cols[3] == "N/A":
                continue
            try:
                mean_true = float(cols[2])
                mean_pred = float(cols[3])
            except ValueError:
                continue
            pairs.append((mean_true, mean_pred))

        if pairs:
            methods_data[method] = pairs

    return methods_data


def parse_method_mae(text: str) -> dict[str, float]:
    maes: dict[str, float] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not (line.startswith("|") and "Method" in line and "MAE" in line):
            i += 1
            continue

        i += 1
        while i < len(lines):
            row = lines[i].strip()
            if not row.startswith("|"):
                break
            cols = [c.strip() for c in row.split("|")[1:-1]]
            i += 1
            if len(cols) < 4 or cols[0] in {"Method", "---"}:
                continue
            if cols[1].startswith("---"):
                continue
            m = LEADING_FLOAT_RE.match(cols[2])  # MAE column
            if m is None:
                continue
            maes[cols[0]] = float(m.group(1))
        continue

    return maes


def build_from_result_dir(
    result_dir: Path,
    suffix: Optional[str],
    methods_arg: Optional[list[str]],
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, float]]:
    suffix_for_match = (suffix or "").strip() or None
    methods = methods_arg if methods_arg else discover_methods(result_dir, suffix_for_match)

    methods_data: dict[str, list[tuple[float, float]]] = {}
    method_maes: dict[str, float] = {}
    for method in methods:
        user_metrics = load_method_user_metrics(method, result_dir, suffix_for_match)
        user_buckets = load_method_r_bucket_precision(method, result_dir, suffix_for_match)
        if not user_buckets:
            continue

        agg = _aggregate_r_bucket_precision(list(user_buckets.values()))
        pairs: list[tuple[float, float]] = []
        for b in agg:
            mean_true = b.get("mean_true_sec")
            mean_pred = b.get("mean_pred_sec")
            if mean_true is None or mean_pred is None:
                continue
            pairs.append((float(mean_true), float(mean_pred)))
        if not pairs:
            continue
        methods_data[method] = pairs

        total_weight = 0.0
        weighted_sum = 0.0
        for row in user_metrics.values():
            w = float(row.get("size", 0))
            mae = row.get("MAE")
            if mae is None or w <= 0:
                continue
            total_weight += w
            weighted_sum += float(mae) * w
        if total_weight > 0:
            method_maes[method] = weighted_sum / total_weight

    return methods_data, method_maes


def build_ordered_methods(
    methods_data: dict[str, list[tuple[float, float]]],
    maes: dict[str, float],
) -> list[tuple[str, list[tuple[float, float]], float | None, int | None]]:
    methods = list(methods_data.keys())
    methods.sort(key=lambda m: (float("inf") if m not in maes else maes[m], m))
    out: list[tuple[str, list[tuple[float, float]], float | None, int | None]] = []
    for idx, method in enumerate(methods):
        mae = maes.get(method)
        rank = idx + 1 if mae is not None else None
        out.append((method, methods_data[method], mae, rank))
    return out


def plot_method(
    method: str,
    pairs: list[tuple[float, float]],
    out_path: Path,
    axis_max: float,
    dpi: int,
    mae: float | None = None,
    rank: int | None = None,
) -> None:
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=35, alpha=0.85, label="R-bucket means")
    plt.plot(x, y, alpha=0.6)
    plt.plot([0, axis_max], [0, axis_max], "--", linewidth=1.5, label="y = x (perfect calibration)")
    plt.xlim(0, axis_max)
    plt.ylim(0, axis_max)
    plt.xlabel("Real time (Mean true, s)")
    plt.ylabel("Predicted time (Mean pred, s)")
    if rank is not None and mae is not None:
        plt.title(f"Calibration: {method} (MAE #{rank}: {mae:.2f}s)")
    else:
        plt.title(f"Calibration: {method}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_grid(
    ordered_methods: list[tuple[str, list[tuple[float, float]], float | None, int | None]],
    out_path: Path,
    axis_max: float,
    dpi: int,
) -> None:
    n = len(ordered_methods)
    if n == 0:
        return

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), squeeze=False)

    for idx, (method, pairs, mae, rank) in enumerate(ordered_methods):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        ax.scatter(x, y, s=20, alpha=0.85)
        ax.plot(x, y, alpha=0.6)
        ax.plot([0, axis_max], [0, axis_max], "--", linewidth=1.2)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        if rank is not None and mae is not None:
            ax.set_title(f"#{rank} {method}\nMAE {mae:.2f}s")
        else:
            ax.set_title(method)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Mean true (s)")
        ax.set_ylabel("Mean pred (s)")

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Create calibration plots from evaluate.py text output.")
    p.add_argument("--input", default=None, help="Text file containing evaluate.py output (legacy mode)")
    p.add_argument("--result-dir", default="./result", help="Directory with result/*.jsonl files")
    p.add_argument("--suffix", default="NO_FIRST_REVIEWS", help="Result suffix to read from result dir mode")
    p.add_argument("--methods", nargs="*", default=None, help="Optional method names to include")
    p.add_argument("--out-dir", default="calibration_plots", help="Directory for per-method PNG files")
    p.add_argument("--axis-max", type=float, default=25.0, help="Axis max for both x and y")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--grid", action="store_true", help="Also save one combined grid image")
    args = p.parse_args()

    if args.input:
        input_path = Path(args.input)
        text = input_path.read_text(encoding="utf-8")
        methods_data = parse_methods_data(text)
        method_maes = parse_method_mae(text)
        print(f"Source: {input_path}")
    else:
        result_dir = Path(args.result_dir)
        methods_data, method_maes = build_from_result_dir(
            result_dir=result_dir,
            suffix=args.suffix,
            methods_arg=args.methods,
        )
        print(f"Source: {result_dir} (suffix={args.suffix})")
    ordered_methods = build_ordered_methods(methods_data, method_maes)
    print(f"Parsed methods: {len(methods_data)}")
    if ordered_methods and any(mae is not None for _, _, mae, _ in ordered_methods):
        print("Order: ascending MAE")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, pairs, mae, rank in ordered_methods:
        out_path = out_dir / f"{safe_name(method)}.png"
        plot_method(
            method,
            pairs,
            out_path=out_path,
            axis_max=float(args.axis_max),
            dpi=int(args.dpi),
            mae=mae,
            rank=rank,
        )

    if args.grid and ordered_methods:
        plot_grid(
            ordered_methods,
            out_path=out_dir / "all_methods_grid.png",
            axis_max=float(args.axis_max),
            dpi=int(args.dpi),
        )

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
