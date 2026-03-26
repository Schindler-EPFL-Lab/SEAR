from typing import Sequence

from sear.data_processing.topics import BaseTopic


def synchronize(
    master: Sequence[BaseTopic],
    data: Sequence[BaseTopic],
) -> Sequence[BaseTopic]:
    """
    For each item in `master` finds the closest (in terms of time) item in `data`.

    Let n = len(master), m = len(data), then implemented algorithm works in O(n * logn +
    m * logm) which happen because the sorting the arrays. The naive implementation
    works in O(n * m).

    :return: a list of neighboring items.
    """

    if len(master) == 0:
        return []
    if len(data) == 0:
        raise RuntimeError("The `data` is empty.")

    master_items = sorted(master, key=lambda x: x.timestamp)
    data_items = sorted(data, key=lambda x: x.timestamp)

    # In the algorithm we aim that current data time <= master time <= next data time.
    # The lists with items contain non-decreasing time after sorting which simplifies
    # the algorithm.
    data_index = 0
    result: list[BaseTopic] = []
    for master_index in range(len(master_items)):
        master_time = master_items[master_index].timestamp

        # We might need to adjust the interval such that master time is in between of
        # the data time and next data time
        while data_index < len(data_items):
            # If the last element is presented then there is no next data time and
            # anyway we must took the current data item
            if data_index == len(data_items) - 1:
                result.append(data_items[data_index])
                break

            next_data_index = data_index + 1
            current_data_time = data_items[data_index].timestamp
            next_data_time = data_items[next_data_index].timestamp

            # If the master time is in between then we check which timestamp is closer.
            # Otherwise we need to change the interval by updating the `data_index``.
            if next_data_time >= master_time:
                selected = next_data_index
                if (master_time - current_data_time) <= (next_data_time - master_time):
                    selected = data_index

                result.append(data_items[selected])
                break

            # Since the arrays are non-decreasing the desired data_index on the next
            # step cannot be smaller than the current data index, because the next
            # master time >= current master time >= current data time
            data_index += 1

        data_index = min(data_index, len(data_items) - 1)
    return result
