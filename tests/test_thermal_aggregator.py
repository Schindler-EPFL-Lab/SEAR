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


class ModelGrandParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
        self.mlp = nn.Linear(1, 1)


class ModelParent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grand_parent = ModelGrandParent()
        self.buffer = nn.Parameter(torch.tensor([0.4, 0.5, 0.6]))
        self.mlp = nn.Linear(1, 1)


class ModelSon(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.grand_parent = ModelGrandParent()
        self.parent = ModelParent()

        self.mlp = nn.Linear(1, 1)


class TestFindInjectableLayers(unittest.TestCase):
    """
    Tests that find_injectable_layers class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        # for find injectable layers
        cls.model_none = ModelNoneInjectable()
        cls.model = Model()

        # for get state dict part
        cls.model_son = ModelSon()
        cls.model_parent = ModelParent()
        cls.state_dict_son = {
            "grand_parent.buffer": torch.tensor([0.1, 0.2, 0.3]),
            "grand_parent.mlp.weight": torch.tensor([[0.5]]),
            "grand_parent.mlp.bias": torch.tensor([-0.5]),
            "parent.buffer": torch.tensor([0.4, 0.5, 0.6]),
            "parent.grand_parent.buffer": torch.tensor([0.1, 0.2, 0.3]),
            "parent.grand_parent.mlp.weight": torch.tensor([[0.5]]),
            "parent.grand_parent.mlp.bias": torch.tensor([-0.5]),
            "parent.mlp.weight": torch.tensor([[0.99]]),
            "parent.mlp.bias": torch.tensor([2.28]),
            "mlp.weight": torch.tensor([[1.234]]),
            "mlp.bias": torch.tensor([-0.321]),
        }
        cls.state_dict_parent = {
            "buffer": torch.tensor([0.4, 0.5, 0.6]),
            "grand_parent.buffer": torch.tensor([0.1, 0.2, 0.3]),
            "grand_parent.mlp.weight": torch.tensor([[0.5]]),
            "grand_parent.mlp.bias": torch.tensor([-0.5]),
            "mlp.weight": torch.tensor([[0.99]]),
            "mlp.bias": torch.tensor([2.28]),
        }

        cls.state_dict_grand_parent = {
            "buffer": torch.tensor([0.1, 0.2, 0.3]),
            "mlp.weight": torch.tensor([[0.5]]),
            "mlp.bias": torch.tensor([-0.5]),
        }

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

    def test_get_state_dict_empty(self) -> None:
        """
        Tests that the method `get_state_dict_part` returns nothing for keys not
        presented in the model.
        """

        test_cases = [
            {
                "expected_state_dict": {},
                "starts_with": "qwerty",
                "state_dict": self.state_dict_son,
            },
            {
                "expected_state_dict": {},
                "starts_with": "parent.parent",
                "state_dict": self.state_dict_son,
            },
            {
                "expected_state_dict": {},
                "starts_with": "_grand_parent",
                "state_dict": self.state_dict_parent,
            },
        ]

        for i in range(len(test_cases)):
            with self.subTest(i=i):
                expected_state_dict = test_cases[i]["expected_state_dict"]
                state_dict_part = ThermalAggregatorBase.get_state_dict_part(
                    state_dict=test_cases[i]["state_dict"],
                    starts_with=test_cases[i]["starts_with"],
                )

                self.assertEqual(
                    set(expected_state_dict.keys()),
                    set(state_dict_part.keys()),
                )

                for k in expected_state_dict:
                    self.assertTrue(
                        torch.allclose(expected_state_dict[k], state_dict_part[k])
                    )

    def test_get_state_dict_part(self) -> None:
        """
        Tests that the method `get_state_dict_part` works properly for various models.
        """

        test_cases = [
            {
                "expected_state_dict": self.state_dict_parent,
                "starts_with": "parent",
                "state_dict": self.state_dict_son,
            },
            {
                "expected_state_dict": self.state_dict_grand_parent,
                "starts_with": "grand_parent",
                "state_dict": self.state_dict_son,
            },
            {
                "expected_state_dict": self.state_dict_grand_parent,
                "starts_with": "parent.grand_parent",
                "state_dict": self.state_dict_son,
            },
            {
                "expected_state_dict": self.state_dict_grand_parent,
                "starts_with": "grand_parent",
                "state_dict": self.state_dict_parent,
            },
        ]

        for i in range(len(test_cases)):
            with self.subTest(i=i):
                expected_state_dict = test_cases[i]["expected_state_dict"]
                state_dict_part = ThermalAggregatorBase.get_state_dict_part(
                    state_dict=test_cases[i]["state_dict"],
                    starts_with=test_cases[i]["starts_with"],
                )

                self.assertEqual(
                    set(expected_state_dict.keys()),
                    set(state_dict_part.keys()),
                )

                for k in expected_state_dict:
                    self.assertTrue(
                        torch.allclose(expected_state_dict[k], state_dict_part[k])
                    )

    def test_load_part(self) -> None:
        """
        Tests that it is possible to load a part of the model using a part of the model
        from the method `get_state_dict_part`
        """

        test_cases = [
            {
                "state_dict": self.state_dict_son,
                "starts_with": "parent",
                "module_load": self.model_son.parent,
            },
            {
                "state_dict": self.state_dict_son,
                "starts_with": "grand_parent",
                "module_load": self.model_son.grand_parent,
            },
            {
                "state_dict": self.state_dict_son,
                "starts_with": "parent.grand_parent",
                "module_load": self.model_son.parent.grand_parent,
            },
            {
                "state_dict": self.state_dict_parent,
                "starts_with": "grand_parent",
                "module_load": self.model_parent.grand_parent,
            },
        ]

        for i in range(len(test_cases)):
            with self.subTest(i=i):
                state_dict_part = ThermalAggregatorBase.get_state_dict_part(
                    state_dict=test_cases[i]["state_dict"],
                    starts_with=test_cases[i]["starts_with"],
                )
                test_cases[i]["module_load"].load_state_dict(state_dict_part)
