import tempfile
import unittest
from pathlib import Path

from script import Config, _list_user_ids, _resolve_unprocessed_user_ids


def _mk_user_partitions(root: Path, user_ids: list[int]) -> None:
    revlogs = root / "revlogs"
    revlogs.mkdir(parents=True, exist_ok=True)
    for uid in user_ids:
        (revlogs / f"user_id={uid}").mkdir()


class UserSelectionTests(unittest.TestCase):
    def test_list_user_ids_without_filters(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_user_partitions(root, [100003, 100001, 100002])

            got = _list_user_ids(root, max_user_id=None, user_id=None)

            self.assertEqual(got, [100001, 100002, 100003])

    def test_list_user_ids_with_max_user_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_user_partitions(root, [100001, 100002, 100003])

            got = _list_user_ids(root, max_user_id=100002, user_id=None)

            self.assertEqual(got, [100001, 100002])

    def test_list_user_ids_with_exact_user_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_user_partitions(root, [100001, 100002, 100003])

            got = _list_user_ids(root, max_user_id=None, user_id=100002)

            self.assertEqual(got, [100002])

    def test_user_id_takes_precedence_over_max_user_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_user_partitions(root, [100001, 100002, 100003])

            got = _list_user_ids(root, max_user_id=100001, user_id=100003)

            self.assertEqual(got, [100003])

    def test_resolve_unprocessed_user_ids_filters_processed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_user_partitions(root, [100001, 100002, 100003])

            config = Config(data_path=root, user_id=None, max_user_id=None)
            got = _resolve_unprocessed_user_ids(config, processed_users={100002})

            self.assertEqual(got, [100001, 100003])


if __name__ == "__main__":
    unittest.main()
