# Calibration Plots from `evaluate.py` Output

Use `calibration_plots.py` to generate calibration plots directly from `result/*.jsonl` (recommended), or from text output captured from `evaluate.py` (legacy mode).

## Usage (Recommended)

Generate per-method plots directly from result files (default suffix is `NO_FIRST_REVIEWS`):

```bash
python3 calibration_plots.py --result-dir ./result --grid
```

Pick another suffix:

```bash
python3 calibration_plots.py --result-dir ./result --suffix WITH_FIRST_REVIEWS --grid
```

Restrict to selected methods:

```bash
python3 calibration_plots.py --result-dir ./result --methods FSRS7_R_LINEAR MOVING_AVG --grid
```

## Legacy text mode

The parser expects blocks like:

- `Method: <NAME> (Within 2s uses |pred-true| <= 2.0s)`
- a markdown table containing `Mean true (s)` and `Mean pred (s)`

Generate one PNG per method from `results.txt`:

```bash
python3 calibration_plots.py --input results.txt --out-dir calibration_plots
```

Generate per-method plots plus one grid image from `results.txt`:

```bash
python3 calibration_plots.py --input results.txt --out-dir calibration_plots --grid
```

Adjust axes and rendering:

```bash
python3 calibration_plots.py --input results.txt --axis-max 25 --dpi 150 --grid
```

## Outputs

- `calibration_plots/<METHOD>.png`
- `calibration_plots/all_methods_grid.png` (when `--grid` is enabled)

## Ordering

- Methods are sorted by ascending MAE from the main `evaluate.py` summary table in `results.txt`.
- Plot titles indicate this order (`MAE #<rank>: <value>s`).
