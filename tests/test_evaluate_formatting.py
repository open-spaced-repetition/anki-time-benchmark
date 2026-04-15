import unittest

from evaluate import sigdig


class EvaluateFormattingTests(unittest.TestCase):
    def test_sigdig_zero_ci(self) -> None:
        value, ci = sigdig(12.3456, 0.0)
        self.assertEqual(value, "12.35")
        self.assertEqual(ci, "0.00")


if __name__ == "__main__":
    unittest.main()
