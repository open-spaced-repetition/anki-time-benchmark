import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from script import Config, _load_or_pretrain_nn_state


class NNCheckpointLoadingTests(unittest.TestCase):
    def test_load_uses_weights_only_false_when_checkpoint_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "review_time_pretrained.pth"
            ckpt.write_text("dummy", encoding="utf-8")
            cfg = Config(nn_ckpt_path=ckpt)

            fake_state = {"state_dict": {}, "norm_mean": [0.0], "norm_std": [1.0]}
            with patch("script.torch.load", return_value=fake_state) as mock_load:
                out = _load_or_pretrain_nn_state(cfg)

            self.assertEqual(out, fake_state)
            mock_load.assert_called_once_with(ckpt, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    unittest.main()
