def find_files_order(
    original_files: list[str],
    found_files: list[str],
) -> list[int]:
    """
    Given the lists `original_files` and `found_files`, compute an ordering of indices
    for `found_files` such that applying this ordering will sort `found_files` according
    to the order in which the same filenames appear in `original_files`.

    :return: the order
    """
    if not set(found_files).issubset(set(original_files)):
        raise RuntimeError(
            "The found_files must be a subset of original_files but got "
            + f"{found_files} and {original_files} respectively"
        )

    original_files_positions = {
        file_name: i for i, file_name in enumerate(original_files)
    }
    found_order = list(range(len(found_files)))
    return sorted(
        found_order, key=lambda index: original_files_positions[found_files[index]]
    )
