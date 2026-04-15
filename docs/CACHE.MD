# FSRS Weight Cache

`script.py` can cache fitted FSRS weights to avoid re-optimizing the same user/split on later runs.

## Cache files

- Default directory: `.cache/fsrs_weights`
- File format: JSON array of floats (FSRS parameters)

Cache keys include user/split plus relevant config and train-window metadata, so different settings produce separate cache files.

## CLI options

- Enabled by default.
- Disable cache reads/writes:

```bash
python3 script.py --no-cache-fsrs-weights
```

- Use a custom cache directory:

```bash
python3 script.py --fsrs-weights-cache-dir /tmp/fsrs-cache
```
