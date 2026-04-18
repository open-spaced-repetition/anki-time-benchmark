import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from script import Config, _fit_algorithm_weights_cached


class FsrsWeightCacheTests(unittest.TestCase):
    def test_cached_weights_are_reused_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = Config(cache_fsrs_weights=True, fsrs_weights_cache_dir=Path(td))
            df = pd.DataFrame({"event_id": [1, 2, 3]})

            with patch("script._fit_algorithm_weights", return_value=[1.0, 2.0]) as mock_fit:
                out1 = _fit_algorithm_weights_cached(df, config, user_id=100001, split_key=50)
                self.assertEqual(out1, [1.0, 2.0])
                self.assertEqual(mock_fit.call_count, 1)

            with patch("script._fit_algorithm_weights", side_effect=RuntimeError("should not refit")) as mock_fit:
                out2 = _fit_algorithm_weights_cached(df, config, user_id=100001, split_key=50)
                self.assertEqual(out2, [1.0, 2.0])
                self.assertEqual(mock_fit.call_count, 0)


if __name__ == "__main__":
    unittest.main()
