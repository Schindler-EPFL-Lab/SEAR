import unittest
from datetime import datetime
from typing import Any

import numpy as np

from sear.data_processing.synchronize import synchronize
from sear.data_processing.topics import BaseTopic


class TestTopic(BaseTopic):
    """A topic that contains int values measured at timestamp"""

    def __init__(self, timestamp: datetime, value: int) -> None:
        """Initializes the TestTopic with `value` measured at `timestamp`"""
        super().__init__(timestamp=timestamp)
        self._value = value

    @property
    def value(self) -> int:
        return self._value


class TestSyncronization(unittest.TestCase):
    """
    Tests that functions to syncronize two sequences.
    """

    def test_synchronize(self) -> None:
        """
        Tests that synchronize works properly on a few examples.
        """

        test_cases = [
            {
                "master": {"2025-10-20T12:01:00.000000": 0},
                "data": {"2025-10-20T14:37:00.000000": 1},
                "expected_data": [("2025-10-20T14:37:00.000000", 1)],
            },
            {
                "master": {
                    "2025-10-20T12:01:00.000000": 0,
                    "2025-10-20T12:02:00.000000": 1,
                    "2025-10-20T12:03:00.000000": 2,
                    "2025-10-20T12:04:00.000000": 3,
                    "2025-10-20T12:05:00.000000": 4,
                    "2025-10-20T12:06:00.000000": 5,
                    "2025-10-20T12:07:00.000000": 6,
                    "2025-10-20T12:08:00.000000": 7,
                    "2025-10-20T12:09:00.000000": 8,
                    "2025-10-20T12:10:00.000000": 9,
                },
                "data": {"2025-10-20T10:22:00.000000": 5},
                "expected_data": [("2025-10-20T10:22:00.000000", 5)] * 10,
            },
            {
                "master": {
                    "2025-10-20T12:01:00.000000": 0,
                    "2025-10-20T12:02:00.000000": 1,
                    "2025-10-20T12:03:00.000000": 2,
                    "2025-10-20T12:04:00.000000": 3,
                    "2025-10-20T12:05:00.000000": 4,
                    "2025-10-20T12:06:00.000000": 5,
                    "2025-10-20T12:07:00.000000": 6,
                    "2025-10-20T12:08:00.000000": 7,
                    "2025-10-20T12:09:00.000000": 8,
                    "2025-10-20T12:10:00.000000": 9,
                },
                "data": {"2025-10-20T21:02:00.000000": 7},
                "expected_data": [("2025-10-20T21:02:00.000000", 7)] * 10,
            },
            {
                "master": {
                    "2025-10-20T12:01:00.000000": 0,
                    "2025-10-20T12:02:00.000000": 1,
                    "2025-10-20T12:03:00.000000": 2,
                    "2025-10-20T12:04:00.000000": 3,
                    "2025-10-20T12:05:00.000000": 4,
                    "2025-10-20T12:06:00.000000": 5,
                    "2025-10-20T12:07:00.000000": 6,
                    "2025-10-20T12:08:00.000000": 7,
                    "2025-10-20T12:09:00.000000": 8,
                    "2025-10-20T12:10:00.000000": 9,
                },
                "data": {
                    "2025-10-20T12:00:30.000000": 9,
                    "2025-10-20T12:00:45.000000": 5,
                    "2025-10-20T12:01:20.000000": 8,
                    "2025-10-20T12:02:40.000000": 1,
                    "2025-10-20T12:03:15.000000": 2,
                    "2025-10-20T12:05:30.000000": 3,
                    "2025-10-20T12:07:00.000000": 4,
                    "2025-10-20T12:07:12.000000": 6,
                    "2025-10-20T12:07:47.000000": 7,
                    "2025-10-20T12:08:38.000000": 12,
                    "2025-10-20T12:09:30.000000": 10,
                    "2025-10-20T12:10:10.000000": 11,
                    "2025-10-20T12:11:55.000000": 27,
                    "2025-10-20T12:17:01.000000": 15,
                },
                "expected_data": [
                    ("2025-10-20T12:00:45.000000", 5),
                    ("2025-10-20T12:01:20.000000", 8),
                    ("2025-10-20T12:03:15.000000", 2),
                    ("2025-10-20T12:03:15.000000", 2),
                    ("2025-10-20T12:05:30.000000", 3),
                    ("2025-10-20T12:05:30.000000", 3),
                    ("2025-10-20T12:07:00.000000", 4),
                    ("2025-10-20T12:07:47.000000", 7),
                    ("2025-10-20T12:08:38.000000", 12),
                    ("2025-10-20T12:10:10.000000", 11),
                ],
            },
        ]

        for subtest_index in range(len(test_cases)):
            with self.subTest(i=subtest_index):
                master = [
                    TestTopic(timestamp=datetime.fromisoformat(t), value=v)
                    for (t, v) in test_cases[subtest_index]["master"].items()
                ]
                data = [
                    TestTopic(timestamp=datetime.fromisoformat(t), value=v)
                    for (t, v) in test_cases[subtest_index]["data"].items()
                ]
                expected_result = [
                    TestTopic(timestamp=datetime.fromisoformat(t), value=v)
                    for (t, v) in test_cases[subtest_index]["expected_data"]
                ]
                result = synchronize(
                    master=master,
                    data=data,
                )

                self.assertEqual(len(result), len(expected_result))
                for i in range(len(result)):
                    self.assertEqual(result[i].timestamp, expected_result[i].timestamp)
                    self.assertEqual(result[i].value, expected_result[i].value)

    @staticmethod
    def _naive_synchronize(
        master: list[BaseTopic],
        data: list[BaseTopic],
    ) -> list[BaseTopic]:
        """
        For each item in `master` it finds the closest element in `data` in terms of
        timestep difference. This implementation is naive and works in O(len(master) *
        len(data)).

        :return: samples from `data` which are the closest in time to the samples in
        `master`.
        """

        master_items = sorted(master, key=lambda x: x.timestamp)
        data_items = sorted(data, key=lambda x: x.timestamp)

        result: list[tuple[str, Any]] = []
        for master_index in range(len(master_items)):
            master_time = master_items[master_index].timestamp

            min_item = None
            min_distance = None
            for data_index in range(len(data_items)):
                data_time = data_items[data_index].timestamp
                current_distance = (master_time - data_time).total_seconds()
                current_distance = abs(current_distance)
                if min_distance is None:
                    min_distance = current_distance
                    min_item = data_items[data_index]

                if current_distance < min_distance:
                    min_distance = current_distance
                    min_item = data_items[data_index]
            result.append(min_item)

        return result

    def test_synchronize_vs_naive(self) -> None:
        """
        Tests that naive implementation of synchronize works exactly the same the
        original implementation.
        """

        base_time = datetime.fromisoformat("2025-10-20T12:00:00.000000").timestamp()
        for test_index in range(100):
            np.random.seed(test_index)
            len_master, len_data = np.random.randint(
                low=1, high=100, size=(2,)
            ).tolist()
            master_date_int = np.random.uniform(low=0, high=1000, size=(len_master,))
            data_date_int = np.random.uniform(low=-500, high=1500, size=(len_data,))

            master_date_str = [
                datetime.fromtimestamp(base_time + date) for date in master_date_int
            ]
            master = [
                TestTopic(timestamp=k, value=np.random.randint(1, 50, (1,)).item())
                for k in master_date_str
            ]
            data_date_str = [
                datetime.fromtimestamp(base_time + date) for date in data_date_int
            ]
            data = [
                TestTopic(timestamp=k, value=np.random.randint(1, 50, (1,)).item())
                for k in data_date_str
            ]

            naive_result = self._naive_synchronize(master=master, data=data)
            result = synchronize(master=master, data=data)

            self.assertEqual(len(naive_result), len(result))
            for i in range(len(naive_result)):
                self.assertEqual(naive_result[i].timestamp, result[i].timestamp)
                self.assertEqual(naive_result[i].value, result[i].value)
