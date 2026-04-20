# Anki Time Benchmark

## Introduction

This benchmark reuses some of the code from the [SRS benchmark](https://github.com/open-spaced-repetition/srs-benchmark). However, it has a different goal: 
instead of measuring how good spaced repetition algorithms are at predicting the probability of recall, it measures how well certain methods can predict the amount of time spent on a single review. This can be useful for improving Anki's simulator, or any spaced repetition simulator that is similar to Anki's.

## Dataset

The dataset for the this benchmark comes from 10 thousand Anki users. In total, this dataset contains information about ~727 million reviews of flashcards. The full dataset is hosted on Hugging Face Datasets: [open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k).

## Evaluation

### Data Split

In the SRS benchmark, we use a tool called `TimeSeriesSplit`. This is part of the [sklearn](https://scikit-learn.org/) library used for machine learning. The tool helps us split the data by time: older reviews are used for training and newer reviews for testing. That way, we don't accidentally cheat by peeking into the future. In practice, we use past study sessions to predict future ones. This makes `TimeSeriesSplit` a good fit for our benchmark.

### Metrics

We use three metrics in the time benchmark to evaluate how well these methods perform: [RMSE](https://en.wikipedia.org/wiki/Root_mean_square_deviation), [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) and [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error). For all three metrics lower values are better.

### Methods

1) const: a hard-coded constant time. Currently, 7 seconds is used.
2) user_median: median time. In this benchmark it's estimated from the train split and then used in the test split. One parameter per user.
3) grade_median_4: median time for Again/Hard/Good/Easy. 4 parameters per user.
4) grade_median_4_4: same as above, but the previous grade also matters. For example, time for Again->Again is different from time for Again->Hard. 16 parameters per user.
5) poor_mans_fsrs: `t = a0 + a1 * ln(number of Agains) + a2 * ln(total number of all reps) + a3 * e^(-a4 * interval length) + a5 * G`. This is akin to using a very crude version of FSRS.
6) moving_avg: [similar to the one used in the main benchmark](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/model_processors.py#L119-L157), it predicts time of the next review based on time of recent reviews. Interval lengths and grades don't matter. Roughly speaking, the idea is that if recent reviews took around 10 seconds, then the next review will probably also take around 10 seconds, and if recent reviews took 5 seconds, then the next review will probably also take around 5 seconds.
7) fsrs_r_linear: `t = b + a * R`. Here R is probability of recall predicted by FSRS-7. `a` and `b` are estimated based on each user's review history.
8) fsrs_r_ridge: `t = b + a * R` with Ridge regularization.
9) fsrs_r_linear_by_grades: `t = b_g + a_g * R`, two linear regression parameters per grade for a total of 8 parameters.
10) fsrs_r_grade_interact: `t = a0 + a1 * G + a2 * R + a3 * G * R`, where G (grade) can take values 1, 2, 3, 4 for Again, Hard, Good and Easy respectively. `a0`, `a1`, `a2` and `a3` are estimated based on each user's review history.
11) fsrs_one_minus_r_s_reps_d_linear: `t = a + b * (1 - R) + c * S + d * reps + e * D`, where `R`, `S`, and `D` come from FSRS-7 and `reps` is the number of previous reviews for the card. Coefficients are fitted per user from the train split.
12) fsrs_one_minus_r_s_reps_d_linear_by_grade: same as above, but fitted separately per grade.
13) fsrs_one_minus_r_s_reps_d_ridge: same features as above, with Ridge regularization.
14) fsrs_dsr_grade_nn: a simple feedforward neural network is used. It takes the grade and difficulty (D), stability (S), retrievability (R) from FSRS-7 as input. The neural network is first pretrained on 250 users, and then fine-tuned on each user individually. During fine-tuning only the last layer is optimized, to avoid overfitting, while other parameters remain frozen. This way it can learn the general pattern from 250 users while still being able to adapt to each user individually.

### Running

Run one method:

```bash
python3 script.py --data ../anki-revlogs-10k --method grade_median_4 --processes 1
```

Run with saved fitted parameters in `result/*.jsonl`:

```bash
python3 script.py --data ../anki-revlogs-10k --method fsrs_r_linear --save-weights
```

Result rows include `r_bucket_precision` (all methods) with 5% `R` buckets, including mean true time, mean predicted time, RMSE, MAE, and `% precise enough` (`|pred-true| <= 2.0s`). `evaluate.py` also reports a ratio-mapping score vs the `0.85-0.90` bucket.

`evaluate.py` additionally prints a per-method correlation summary between `R` and response time (bucket-level weighted Pearson and Spearman for true/predicted means).

Run all methods in one command:

```bash
python3 script.py --data ../anki-revlogs-10k --all-methods --processes 1
```

Run linear methods with MAE fitting instead of MSE:

```bash
python3 script.py --data ../anki-revlogs-10k --method fsrs_r_linear --linear-loss mae
```

FSRS optimization cache (enabled by default):

```bash
python3 script.py --data ../anki-revlogs-10k --all-methods --fsrs-weights-cache-dir .cache/fsrs_weights
```

Disable FSRS cache:

```bash
python3 script.py --data ../anki-revlogs-10k --all-methods --no-cache-fsrs-weights
```

Run only one exact user:

```bash
python3 script.py --data ../anki-revlogs-10k --user-id 100001 --method grade_median_4 --processes 1
```

### Script CLI options

Main run selection:

- `--data <path>`: dataset root (expects `<path>/revlogs/user_id=...` partitions)
- `--method <name>`: run one method
- `--all-methods`: run all methods in one invocation
- `--user-id <id>`: run only one exact user id
- `--max-user-id <id>`: run users with `user_id <= id`
- `--processes <n>`: number of worker processes
- `--with_first_reviews`: include first reviews in metrics (default excludes)

Newly added method names:

- `fsrs_one_minus_r_s_reps_d_linear`
- `fsrs_r_ridge`
- `fsrs_r_linear_by_grades`
- `fsrs_one_minus_r_s_reps_d_ridge`
- `fsrs_one_minus_r_s_reps_d_linear_by_grade`

Outputs / saved metadata:

- `--save-evaluation-file`: writes per-user TSV files under `evaluation/`
- `--save-raw`: writes raw `t_pred` and `t_true` under `raw/`
- `--save-weights`: stores fitted parameters in `result/*.jsonl`
- `r_bucket_precision` is included for all methods (5% `R` buckets)
- For `fsrs_r_linear`, `result/*.jsonl` includes:
- `regression_parameters` (`a`, `b`)
- For `fsrs_r_linear_by_grades`, `regression_parameters` includes per-grade coefficients
- (`again_a`, `again_b`, `hard_a`, `hard_b`, `good_a`, `good_b`, `easy_a`, `easy_b`)
- `r_bucket_precision` (5% R buckets, with `% precise enough` where `|pred-true| <= 2.0s`)
- For `fsrs_one_minus_r_s_reps_d_linear`, `result/*.jsonl` includes:
- `regression_parameters` (`a`, `b`, `c`, `d`, `e`)
- For `fsrs_one_minus_r_s_reps_d_linear_by_grade`, `regression_parameters` includes per-grade
- (`*_a`, `*_b`, `*_c`, `*_d`, `*_e`)
- For ridge variants, `regression_parameters` also include `ridge_alpha`

FSRS optimization cache:

- `--fsrs-weights-cache-dir <path>`: cache directory for fitted FSRS weights
- `--no-cache-fsrs-weights`: disable cache reads/writes
- Default behavior: FSRS weight cache is enabled

Ridge option:

- `--ridge-alpha <float>`: regularization strength for Ridge methods (`fsrs_r_ridge`, `fsrs_one_minus_r_s_reps_d_ridge`)

Linear fit option:

- `--linear-loss <mse|mae>`: fitting loss for linear variants (`fsrs_r_linear`, `fsrs_r_linear_by_grades`, `fsrs_r_grade_interact`, `fsrs_one_minus_r_s_reps_d_linear`, `fsrs_one_minus_r_s_reps_d_linear_by_grade`). Default: `mae`.

NN options (`fsrs_dsr_grade_nn`):

- `--nn_ckpt <path>`
- `--nn_pretrain_users <n>`
- `--nn_pretrain_epochs <n>`
- `--nn_pretrain_lr <float>`
- `--nn_pretrain_batch_size <n>`
- `--nn_pretrain_max_samples_per_user <n>`
- `--nn_finetune_epochs <n>`
- `--nn_finetune_lr <float>`
- `--nn_finetune_batch_size <n>`

Checkpoint note:

- Existing `checkpoints/review_time_pretrained.pth` files are loaded with full checkpoint deserialization (needed for stored normalizer arrays).

Calibration plots:

- Build per-method calibration plots directly from `result/*.jsonl`:

```bash
python3 calibration_plots.py --result-dir ./result --out-dir calibration_plots --grid
```

- Methods in calibration plots are ordered by ascending MAE and titles indicate MAE rank/value.

- Optional clean rebuild (remove old images first):

```bash
rm -rf calibration_plots
python3 calibration_plots.py --result-dir ./result --out-dir calibration_plots --grid
```

- Legacy mode (from `evaluate.py` text export):

```bash
python3 evaluate.py --result-dir ./result > results.txt
python3 calibration_plots.py --input results.txt --out-dir calibration_plots --grid
```

- See detailed usage in [`docs/CALIBRATION_PLOTS.MD`](docs/CALIBRATION_PLOTS.MD).


## Result

Total number of collections (each from one Anki user): ____.

Total number of reviews for evaluation: _________.

Due to Anki's ["Maximum answer seconds"](https://docs.ankiweb.net/deck-options.html#timers) setting capping review time, reviews with capped time are excluded. 6 users were excluded due to not having valid data after filtering. Additionally, firts reviews are excluded. Time spent on reviewing a card for the first time appears to be sufficiently different from time of the following reviews.

The best result for each metric is highlighted in **bold**.

| Method | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
