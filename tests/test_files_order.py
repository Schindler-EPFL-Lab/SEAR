import unittest

from sear.scripts.eval.files_order import find_files_order


class TestFilesOrder(unittest.TestCase):
    """
    Tests that files_order function works properly
    """

    def test_files_order_raises(self) -> None:
        """Tests that files_order raises an error"""
        with self.assertRaises(RuntimeError):
            find_files_order(original_files=["1", "2"], found_files=["a"])

    def test_files_order_increasing(self) -> None:
        """Tests that files_order works in a simple case"""
        original_files = ["a", "e", "b", "c"]
        files = ["a", "e"]
        expected_order = [0, 1]
        found_order = find_files_order(
            original_files=original_files,
            found_files=files,
        )
        self.assertEqual(found_order, expected_order)

    def test_files_order(self) -> None:
        """Tests that files_order works in a difficult case case when eve"""
        original_files = ["90", "aaa", "e", "b", "c", "a", "cdf"]
        files = ["aaa", "c", "e", "b"]
        expected_order = [0, 2, 3, 1]
        found_order = find_files_order(
            original_files=original_files,
            found_files=files,
        )
        self.assertEqual(found_order, expected_order)
