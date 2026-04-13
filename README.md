# Anki Time Benchmark

## Introduction

This benchmark reuses some of the code from the [SRS benchmark](https://github.com/open-spaced-repetition/srs-benchmark). However, it has a different goal: 
instead of measuring how good spaced repetition algorithms are at predicting the probability of recall, it measures how well certain methods can predict the amount of time spent on a single review. This can be useful for improving Anki's simulator, or any spaced repetition simulator that is similar to Anki's.

## Dataset

The dataset for the this benchmark comes from 10 thousand Anki users. In total, this dataset contains information about ~727 million reviews of flashcards. The full dataset is hosted on Hugging Face Datasets: [open-spaced-repetition/anki-revlogs-10k](https://huggingface.co/datasets/open-spaced-repetition/anki-revlogs-10k).

## Evaluation

### Data Split

In the SRS benchmark, we use a tool called `TimeSeriesSplit`. This is part of the [sklearn](https://scikit-learn.org/) library used for machine learning. The tool helps us split the data by time: older reviews are used for training and newer reviews for testing. That way, we don't accidentally cheat by giving the algorithm future information it shouldn't have. In practice, we use past study sessions to predict future ones. This makes `TimeSeriesSplit` a good fit for our benchmark.

### Metrics

We use three metrics in the time benchmark to evaluate how well these methods perform: [RMSE](https://en.wikipedia.org/wiki/Root_mean_square_deviation), [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) and [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error). For all three metrics lower values are better.

### Methods

1) const: a hard-coded constant time. Currently, 7 seconds is used.
2) user_median: median time. In this benchmark it's estimated from the train split and then used in the test split. One number per user.
3) grade_median_4: median time for Again/Hard/Good/Easy. 4 numbers per user.
4) grade_median_8: median time for Again/Hard/Good/Easy, with the first review receiving special treatment and having its own values. 8 numbers per user. This method is currently used in Anki's simulator.
5) fsrs_r_linear: first review -> first-grade median, else `t=b+a*R`. Here R is probability of recall predicted by FSRS-7.
6) fsrs_r_grade_interact: first review -> first-grade median, else `t=a0+a1*g+a2*R+a3*g*R`, where G (grade) can be 1, 2, 3, 4. 


## Result

Total number of collections (each from one Anki user): 9,994.

Total number of reviews for evaluation: 572,307,825.

Due to Anki's ["Maximum answer seconds"](https://docs.ankiweb.net/deck-options.html#timers) setting capping review time, reviews with capped time are excluded. Additionally, reviews that took >30 minutes are excluded, though they are a small minority of all reviews. 6 users were excluded due to not having valid data after filtering.

The best result for each metric is highlighted in **bold**.

| Method | RMSE | MAE | MAPE |
| --- | --- | --- | --- |
| CONST | 14.64±0.46 s | 8.03±0.18 s | 158%±20% |
