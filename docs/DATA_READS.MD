# Data Read Scope

`script.py` reads review logs from:

- `<data_root>/revlogs/user_id=<id>/...parquet`

## User Selection Flags

- `--user-id <id>`: process exactly one user partition (exact match).
- `--max-user-id <id>`: process all user partitions with `user_id <= <id>`.

When both are provided, `--user-id` is used as the exact filter.

## Example

Run only user `100001`:

```bash
python3 script.py --data ../anki-revlogs-10k --user-id 100001
```
