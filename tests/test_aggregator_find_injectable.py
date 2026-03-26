import re
import unittest

import torch
import torch.nn as nn

from sear.models.thermal_aggregators.base import ThermalAggregatorBase


class ModelNoneInjectable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.buffer1 = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.buffer2 = nn.Parameter(torch.tensor([0.1, 2.2, 1.22]))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(1, 1)
        self.attn1 = nn.MultiheadAttention(1, num_heads=1)
        self.model_inner = nn.Sequential(
            nn.Linear(1, 2),
            nn.Linear(2, 1),
            nn.LeakyReLU(),
            nn.Linear(1, 1),
        )


class TestFindInjectableLayers(unittest.TestCase):
    """
    Tests that find_injectable_layers class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        cls.model_none = ModelNoneInjectable()
        cls.model = Model()

    def test_finds_injectable_layers_none(self) -> None:
        """
        Tests that the for the model without any injectable layers the function
        `find_injectable_layers` returns an empty list.
        """

        expected_result = set()

        pattern = re.compile(r".*")
        result = ThermalAggregatorBase._find_injectable_layers(
            pattern=pattern, module=self.model_none
        )
        result_set = set(result)

        self.assertEqual(len(result), len(result_set))
        self.assertEqual(expected_result, result_set)

    def test_finds_injectable_layers(self) -> None:
        """
        Tests that the for the model with injectable layers the function
        `find_injectable_layers` returns a correct list.
        """

        test_cases = [
            {
                "pattern": re.compile(r".*"),
                "expected_result": [
                    "linear1",
                    "attn1",
                    "model_inner.0",
                    "model_inner.1",
                    "model_inner.3",
                ],
            },
            {
                "pattern": re.compile(r"model_inner.*"),
                "expected_result": [
                    "model_inner.0",
                    "model_inner.1",
                    "model_inner.3",
                ],
            },
        ]

        for i in range(len(test_cases)):
            with self.subTest(i=i):
                expected_result = set(test_cases[i]["expected_result"])
                pattern = test_cases[i]["pattern"]
                result = ThermalAggregatorBase._find_injectable_layers(
                    pattern=pattern, module=self.model
                )
                result_set = set(result)

                self.assertEqual(len(result), len(result_set))
                self.assertEqual(expected_result, result_set)
