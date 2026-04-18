import unittest

from script import ALL_METHODS, _resolve_methods_to_run


class MethodSelectionTests(unittest.TestCase):
    def test_single_method_mode(self) -> None:
        self.assertEqual(_resolve_methods_to_run("const", all_methods=False), ["const"])

    def test_all_methods_mode(self) -> None:
        self.assertEqual(_resolve_methods_to_run("const", all_methods=True), ALL_METHODS)


if __name__ == "__main__":
    unittest.main()
