import re
import unittest

from sear.models.thermal_aggregators.custom_patterns import (
    CustomPatterns,
)


class TestAblationModels(unittest.TestCase):
    """
    Tests that Ablation models work properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        cls.blocks_names = [
            # frame blocks
            "frame_blocks.0.norm1.weight",
            "frame_blocks.1.fc1.bias",
            "frame_blocks.2.gamma",
            "frame_blocks.3.attn",
            "frame_blocks.4.cam",
            "frame_blocks.5",
            "frame_blocks.6.cam",
            "frame_blocks.7.11",
            "frame_blocks.8.11",
            "frame_blocks.9.bb",
            "frame_blocks.10.aaa",
            "frame_blocks.11.cde",
            "frame_blocks.12.qwerty",
            "frame_blocks.13.norm1.weight",
            "frame_blocks.14.cam",
            "frame_blocks.15",
            "frame_blocks.16.cam",
            "frame_blocks.17.11",
            "frame_blocks.18.11",
            "frame_blocks.19.bb",
            "frame_blocks.20.aaa",
            "frame_blocks.21.cde",
            "frame_blocks.22.qwerty",
            "frame_blocks.23.qwerty",
            # global blocks
            "global_blocks.0.coeff",
            "global_blocks.1",
            "global_blocks.2.fc",
            "global_blocks.3.anything",
            "global_blocks.4.hello",
            "global_blocks.5",
            "global_blocks.6.world",
            "global_blocks.7.howareyou",
            "global_blocks.8.layer5",
            "global_blocks.9.44",
            "global_blocks.10.cube",
            "global_blocks.11.donkey",
            "global_blocks.12.qwerty",
            "global_blocks.13.norm5.weight",
            "global_blocks.14.norm5.bias",
            "global_blocks.15",
            "global_blocks.16.cam",
            "global_blocks.17.11",
            "global_blocks.18.11",
            "global_blocks.19.bb",
            "global_blocks.20.aaa",
            "global_blocks.21.cde",
            "global_blocks.22.qwerty",
            "global_blocks.23.qwerty",
            # should not be marked by any
            "patch1.frame_blocks.0.norm1.weight",
            "patch1.frame_blocks.234",
            # should not be marked by any
            "patch2.global_blocks.0.norm1.weight",
            "patch2.global_blocks.234",
        ]

    @staticmethod
    def _find_matches(string_array: list[str], pattern: str) -> list[bool]:
        """
        For each string from `string_array` finds a whether the `pattern` matches to it.

        :return: a list of bool containing True if the pattern matches and False
            otherwise.
        """
        result: list[bool] = []
        pattern_re = re.compile(pattern)
        for el in string_array:
            result.append(bool(re.match(pattern_re, el)))
        return result

    def test_custom_pattern_frame_only(self) -> None:
        """
        Tests that CustomPatterns.FRAME_ONLY works properly.
        """
        matches = [True] * 24 + [False] * 28

        self.assertEqual(
            self._find_matches(
                self.blocks_names, pattern=CustomPatterns.FRAME_ONLY.value
            ),
            matches,
        )

    def test_custom_pattern_global_only(self) -> None:
        """
        Tests that CustomPatterns.GLOBAL_ONLY works properly.
        """
        matches = [False] * 24 + [True] * 24 + [False] * 4

        self.assertEqual(
            self._find_matches(
                self.blocks_names, pattern=CustomPatterns.GLOBAL_ONLY.value
            ),
            matches,
        )

    def test_custom_pattern_original(self) -> None:
        """
        Tests that CustomPatterns.ALL_BLOCKS works properly.
        """
        matches = [True] * 48 + [False] * 4

        self.assertEqual(
            self._find_matches(
                self.blocks_names, pattern=CustomPatterns.ALL_BLOCKS.value
            ),
            matches,
        )

    def test_custom_pattern_first_quater(self) -> None:
        """
        Tests that CustomPatterns.FIRST_QUATER works properly.
        """
        matches = [True] * 6 + [False] * 18 + [True] * 6 + [False] * 18 + [False] * 4

        self.assertEqual(
            self._find_matches(
                self.blocks_names, pattern=CustomPatterns.FIRST_QUATER.value
            ),
            matches,
        )

    def test_custom_pattern_first_two_quaters(self) -> None:
        """
        Tests that CustomPatterns.FIRST_TWO_QUATERS works properly.
        """
        matches = [True] * 12 + [False] * 12 + [True] * 12 + [False] * 12 + [False] * 4

        self.assertEqual(
            self._find_matches(
                self.blocks_names, pattern=CustomPatterns.FIRST_TWO_QUATERS.value
            ),
            matches,
        )

    def test_custom_pattern_first_three_quaters(self) -> None:
        """
        Tests that CustomPatterns.FIRST_THREE_QUATERS works properly.
        """
        matches = [True] * 18 + [False] * 6 + [True] * 18 + [False] * 6 + [False] * 4

        self.assertEqual(
            self._find_matches(
                self.blocks_names, pattern=CustomPatterns.FIRST_THREE_QUATERS.value
            ),
            matches,
        )
